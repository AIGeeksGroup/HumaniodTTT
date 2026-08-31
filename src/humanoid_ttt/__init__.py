"""Core algorithms for HumanoidTTT capability reuse and adaptation."""

from .certificates import build_certificate, certificate_membership, evaluate_membership
from .consolidation import ConsolidationDecision, ConsolidationPolicy, Scorer
from .entry_features import FEATURE_NAMES, capture_entry_features
from .latent import CompactState, LatentBasis, LatentSteering, PerformanceBand
from .memory import (
    StoreAction,
    actions,
    apply_action,
    immediate_rows,
    probe_bank,
    utility,
)
from .motion_features import descriptor_16
from .sac import ConservativeGate, ReplayBuffer, SACAgent
from .types import CanonicalReference, StoreEntry

__all__ = [
    "CanonicalReference",
    "CompactState",
    "ConservativeGate",
    "ConsolidationDecision",
    "ConsolidationPolicy",
    "FEATURE_NAMES",
    "LatentBasis",
    "LatentSteering",
    "PerformanceBand",
    "ReplayBuffer",
    "SACAgent",
    "Scorer",
    "StoreAction",
    "StoreEntry",
    "actions",
    "apply_action",
    "build_certificate",
    "capture_entry_features",
    "certificate_membership",
    "descriptor_16",
    "evaluate_membership",
    "immediate_rows",
    "probe_bank",
    "utility",
]
