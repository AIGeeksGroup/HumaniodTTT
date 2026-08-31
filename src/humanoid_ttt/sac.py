from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self, state_dim: int, action_dim: int, capacity: int, allowed_seeds: set[int]
    ):
        self.state = np.zeros((capacity, state_dim), np.float32)
        self.action = np.zeros((capacity, action_dim), np.float32)
        self.reward = np.zeros((capacity, 1), np.float32)
        self.next_state = np.zeros((capacity, state_dim), np.float32)
        self.done = np.zeros((capacity, 1), np.float32)
        self.seed = np.zeros(capacity, np.int64)
        self.capacity, self.size, self.cursor = int(capacity), 0, 0
        self.allowed_seeds = set(map(int, allowed_seeds))

    def add(self, state, action, reward, next_state, done, seed) -> None:
        if int(seed) not in self.allowed_seeds:
            raise ValueError("seed namespace violation")
        index = self.cursor
        self.state[index] = state
        self.action[index] = action
        self.reward[index, 0] = reward
        self.next_state[index] = next_state
        self.done[index, 0] = float(done)
        self.seed[index] = int(seed)
        self.cursor = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator, device: torch.device):
        if self.size < 1:
            raise ValueError("empty replay")
        index = rng.integers(0, self.size, size=int(batch))
        return tuple(
            torch.as_tensor(x[index], device=device)
            for x in (self.state, self.action, self.reward, self.next_state, self.done)
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "cursor": self.cursor,
            "seeds": sorted(set(map(int, self.seed[: self.size]))),
            "allowed_seeds": sorted(self.allowed_seeds),
        }


def _mlp(input_dim: int, output_dim: int, hidden: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, output_dim),
    )


class Actor(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 64,
        initial_std: float = 0.05,
    ):
        super().__init__()
        self.net = _mlp(state_dim, 2 * action_dim, hidden)
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[action_dim:] = float(np.log(initial_std))
        self.action_dim = action_dim

    def forward(self, state):
        mean, log_std = self.net(state).chunk(2, -1)
        return mean, log_std.clamp(-5.0, -1.0)

    def sample(self, state, deterministic: bool = False):
        mean, log_std = self(state)
        raw = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        action = torch.tanh(raw)
        logp = -0.5 * (
            ((raw - mean) / log_std.exp()) ** 2 + 2 * log_std + np.log(2 * np.pi)
        )
        logp = logp.sum(-1, keepdim=True) - torch.log(1 - action.square() + 1e-6).sum(
            -1, keepdim=True
        )
        return action, logp


class Critic(torch.nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = _mlp(state_dim + action_dim, 1, hidden)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], -1))


@dataclass
class SACAgent:
    state_dim: int
    action_dim: int
    device: torch.device
    hidden: int = 64
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    gamma: float = 0.97
    tau: float = 0.005
    initial_std: float = 0.05

    def __post_init__(self):
        self.actor = Actor(
            self.state_dim, self.action_dim, self.hidden, self.initial_std
        ).to(self.device)
        self.q1 = Critic(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.q2 = Critic(self.state_dim, self.action_dim, self.hidden).to(self.device)
        self.tq1, self.tq2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), self.actor_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), self.critic_lr
        )
        self.log_alpha = torch.tensor(
            np.log(0.05), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], self.actor_lr)
        self.target_entropy = -0.5 * self.action_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state, deterministic=False, seed=None):
        if seed is not None:
            torch.manual_seed(int(seed))
        tensor = torch.as_tensor(
            np.asarray(state)[None], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            action, _ = self.actor.sample(tensor, deterministic)
        return action[0].cpu().numpy().astype(np.float32)

    def q_values(self, state, action):
        s = torch.as_tensor(
            np.asarray(state)[None], dtype=torch.float32, device=self.device
        )
        a = torch.as_tensor(
            np.asarray(action)[None], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            return float(self.q1(s, a).item()), float(self.q2(s, a).item())

    def update(self, replay: ReplayBuffer, rng, batch=32, grad_clip=1.0):
        state, action, reward, next_state, done = replay.sample(batch, rng, self.device)
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state)
            target = reward + self.gamma * (1 - done) * (
                torch.minimum(
                    self.tq1(next_state, next_action), self.tq2(next_state, next_action)
                )
                - self.alpha.detach() * next_logp
            )
        q1, q2 = self.q1(state, action), self.q2(state, action)
        critic_loss = torch.nn.functional.mse_loss(
            q1, target
        ) + torch.nn.functional.mse_loss(q2, target)
        self.q_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = float(
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), grad_clip
            )
        )
        self.q_opt.step()
        sampled, logp = self.actor.sample(state)
        actor_loss = (
            self.alpha.detach() * logp
            - torch.minimum(self.q1(state, sampled), self.q2(state, sampled))
        ).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = float(
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
        )
        self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for target_net, net in ((self.tq1, self.q1), (self.tq2, self.q2)):
                for tp, p in zip(
                    target_net.parameters(), net.parameters(), strict=True
                ):
                    tp.mul_(1 - self.tau).add_(p, alpha=self.tau)
        if not all(
            torch.isfinite(p).all()
            for n in (self.actor, self.q1, self.q2)
            for p in n.parameters()
        ):
            raise RuntimeError("nonfinite SAC parameter")
        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
            "actor_grad": actor_grad,
            "critic_grad": critic_grad,
        }

    def snapshot(self):
        return copy.deepcopy(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "tq1": self.tq1.state_dict(),
                "tq2": self.tq2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            }
        )

    def restore(self, snapshot):
        for name in ("actor", "q1", "q2", "tq1", "tq2"):
            getattr(self, name).load_state_dict(snapshot[name])
        self.log_alpha.data.copy_(snapshot["log_alpha"].to(self.device))

    def save(self, path: Path, metadata: dict[str, Any]):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({**self.snapshot(), "metadata": metadata}, path)

    def load(self, path: Path):
        row = torch.load(path, map_location=self.device, weights_only=False)
        self.restore(row)
        return row.get("metadata", {})


@dataclass(frozen=True)
class ConservativeGate:
    disagreement_threshold: float
    value_tolerance: float
    minimum_action_norm: float = 1e-6

    def decide(
        self, agent: SACAgent, state: np.ndarray, proposed: np.ndarray
    ) -> dict[str, Any]:
        proposal = np.clip(np.asarray(proposed, np.float32), -1.0, 1.0)
        zero = np.zeros_like(proposal)
        p1, p2 = agent.q_values(state, proposal)
        z1, z2 = agent.q_values(state, zero)
        disagreement = abs(p1 - p2)
        accept = bool(
            np.linalg.norm(proposal) > self.minimum_action_norm
            and disagreement <= self.disagreement_threshold
            and min(p1, p2) >= min(z1, z2) - self.value_tolerance
        )
        reason = (
            "ACCEPT"
            if accept
            else (
                "ZERO_PROPOSAL"
                if np.linalg.norm(proposal) <= self.minimum_action_norm
                else "CRITIC_DISAGREEMENT"
                if disagreement > self.disagreement_threshold
                else "VALUE_BELOW_ZERO"
            )
        )
        return {
            "proposed": proposal,
            "executed": proposal if accept else zero,
            "accepted": accept,
            "reason": reason,
            "q1": p1,
            "q2": p2,
            "q_zero_1": z1,
            "q_zero_2": z2,
            "disagreement": disagreement,
        }
