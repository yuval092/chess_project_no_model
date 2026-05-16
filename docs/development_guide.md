# Development Guide

Create and use a local virtual environment: `python3 -m venv .venv`, then `.venv/bin/python -m pip install -e . pytest`.

Generate XML with `.venv/bin/python -m mujoco_chess.app.startup`. Run tests with `.venv/bin/pytest tests/ -q`. Run debug commands with `.venv/bin/python -m mujoco_chess.debug.cli run-env`.

Current UI thread model is pending full game-loop integration. Known v1 limitation from the plan remains: promotion reserve supports at most two promoted pieces of each type per color.
