#!/usr/bin/env python3
"""Lance Hermy et écrit ses sorties dans un journal à taille plafonnée."""
from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "hermy-launchd.log"

logger = logging.getLogger("hermy")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(
    LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(handler)

parser = argparse.ArgumentParser(description="Lance Hermy avec rotation des journaux.")
parser.add_argument("--gui", action="store_true", help="lancer la mascotte graphique (macOS)")
args = parser.parse_args()

process: subprocess.Popen[str] | None = None


def stop_child(signum: int, _frame: object) -> None:
    if process is not None and process.poll() is None:
        process.send_signal(signum)


signal.signal(signal.SIGTERM, stop_child)
signal.signal(signal.SIGINT, stop_child)

process = subprocess.Popen(
    [sys.executable, "-u", str(ROOT / "hermy_voice.py")] + (["--gui"] if args.gui else []),
    cwd=ROOT,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
    bufsize=1,
)
assert process.stdout is not None
for line in process.stdout:
    logger.info(line.rstrip("\n"))

sys.exit(process.wait())
