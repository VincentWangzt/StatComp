from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from models.reverse_model import BaseReverseConditionalModel


@dataclass
class ReverseProposalFit:
    reverse_model: BaseReverseConditionalModel
    proposal_type: str
    fit_mode: str
    diagnostics: dict[str, Any]
    cache: dict[str, torch.Tensor] | None = None
    resolved_config: dict[str, Any] | None = None


@dataclass
class LogQEstimate:
    log_q_values: torch.Tensor
    stderr: float
    diagnostics: dict[str, Any]
    ess_values: torch.Tensor | None = None


@dataclass
class ELMEstimate:
    value: float
    stderr: float
    log_q_values: torch.Tensor
    diagnostics: dict[str, Any]
    ess_values: torch.Tensor | None = None
