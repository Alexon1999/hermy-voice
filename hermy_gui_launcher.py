"""Point d'entrée GUI empaqueté pour macOS."""
from __future__ import annotations

import os
from pathlib import Path
import sys


if sys.stdout is None or sys.stderr is None:
    data_dir = Path.home() / "Library" / "Application Support" / "Hermy"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / "hermy.log"
    if log_path.exists() and log_path.stat().st_size > 2_000_000:
        backup_path = data_dir / "hermy.log.1"
        try:
            backup_path.unlink(missing_ok=True)
            log_path.replace(backup_path)
        except OSError:
            pass
    log = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

import hermy_voice

hermy_voice.GUI_MODE = True
hermy_voice.run_gui()
