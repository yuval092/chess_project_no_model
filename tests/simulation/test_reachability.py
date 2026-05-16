from __future__ import annotations

from mujoco_chess.app.startup import bootstrap
from mujoco_chess.debug.scenarios import ReachabilityChecker
from mujoco_chess.mujoco_env.env import MuJoCoEnv


def test_real_fetch_level1_reachability_reports_all_targets() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    results = ReachabilityChecker(env, config).check_level1()
    assert len(results) == 64
    failed = [r for r in results if not r.passed]
    assert not failed, f"{len(failed)}/64 board squares failed: {[r.target for r in failed[:10]]}"


def test_real_fetch_level2_reachability_reports_sample_targets() -> None:
    config, xml_path = bootstrap()
    env = MuJoCoEnv(xml_path, config)
    env.load()
    results = ReachabilityChecker(env, config).check_level2()
    assert results
    failed = [r for r in results if not r.passed]
    assert not failed, f"{len(failed)} level-2 targets failed: {[r.target for r in failed]}"
