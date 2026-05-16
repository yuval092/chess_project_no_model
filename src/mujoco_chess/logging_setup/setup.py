from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from mujoco_chess.utils.config_loader import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if root.handlers:
        return
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler = RotatingFileHandler(log_dir / "app.log", maxBytes=config.max_bytes, backupCount=config.backup_count)
    handler.setFormatter(formatter)
    root.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(getattr(logging, config.app_log_level.upper(), logging.INFO))
    root.addHandler(console)
