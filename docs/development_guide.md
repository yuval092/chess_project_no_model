# Development Guide

Create and use a local virtual environment: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -e . pytest`.

Generate XML with `.venv/bin/python -m mujoco_chess.app.startup`. Run tests with `.venv/bin/pytest tests/ -q`. Run debug commands with `.venv/bin/python -m mujoco_chess.debug.cli run-env`.

**UI thread model**: `BoardUI.run()` is designed to run in a secondary thread (call `BoardUI.start_thread()` to launch it as a daemon thread). The main thread owns the `GameLoop` and communicates with the UI via two `queue.Queue` objects: `event_queue` (UI → game loop, carries `MoveRequest`, `AIPlayRequest`, `ResetRequest`, `QuitRequest`) and `state_queue` (game loop → UI, carries `chess.Board` snapshots). Pygame is never called from the main thread. In headless/CI environments, set `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy` before any `pygame.init()` call.

**Automated games** (AI vs AI): pass `human_color=None` to `GameLoop.__init__()`. With `human_color=None`, `_is_human_turn()` always returns `False`, so the `move_selector` handles every turn. This is used for `run-random-game` and the stress test scaffolds.

Known v1 limitation from the plan remains: promotion reserve supports at most two promoted pieces of each type per color.
