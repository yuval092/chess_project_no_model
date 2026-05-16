from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np

from mujoco_chess.health.checks import HealthCheckFailureError, HealthCheckResult, write_failure_report


@dataclass
class HealthCheckContext:
    transaction: object | None = None
    active_stage: object | None = None
    target_piece_body: str | None = None
    held_piece_body: str | None = None
    expected_positions_override: dict[str, np.ndarray] = field(default_factory=dict)
    ignored_bodies_for_drift: set[str] = field(default_factory=set)


class CheckContext(Enum):
    STARTUP = "STARTUP"
    SETTLE_WINDOW_BEFORE = "SETTLE_WINDOW_BEFORE"
    SETTLE_WINDOW_AFTER = "SETTLE_WINDOW_AFTER"
    POST_GRASP = "POST_GRASP"
    DURING_CARRY = "DURING_CARRY"
    POST_RELEASE = "POST_RELEASE"
    START_OF_TURN = "START_OF_TURN"
    END_OF_TURN = "END_OF_TURN"


class HealthCheckRunner:
    def __init__(self, env=None, registry=None, config=None) -> None:
        self.env = env
        self.registry = registry
        self.config = config
        self._checks: dict[CheckContext, list[Callable]] = defaultdict(list)

    def register(self, context: CheckContext, check_fn: Callable) -> None:
        self._checks[context].append(check_fn)

    def run_checks(self, context: CheckContext, hc_context: HealthCheckContext | None = None, **kwargs) -> list[HealthCheckResult]:
        return [fn(self.env, self.registry, self.config, hc_context=hc_context, **kwargs) for fn in self._checks[context]]

    def all_passed(self, results: list[HealthCheckResult]) -> bool:
        return all(result.passed for result in results)

    def handle_failure(self, results: list[HealthCheckResult], context_info: dict) -> None:
        log_dir = self.config.logging.log_dir if self.config is not None and hasattr(self.config, "logging") else "logs"
        report_path = write_failure_report(results, context_info, log_dir)
        raise HealthCheckFailureError(f"Health check failure report written to {report_path}")
