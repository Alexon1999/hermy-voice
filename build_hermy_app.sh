#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
RUNNER="$PROJECT_DIR/run_hermy_background.py"
APP_DEST="/Applications/Hermy.app"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python introuvable ou non exécutable : $PYTHON" >&2
  echo "Créez d’abord l’environnement du projet et installez ses dépendances." >&2
  exit 1
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "Lanceur Python introuvable : $RUNNER" >&2
  exit 1
fi
if ! command -v osacompile >/dev/null 2>&1; then
  echo "osacompile est introuvable. Ce script doit être exécuté sur macOS." >&2
  exit 1
fi

if [[ -e "$APP_DEST" ]]; then
  printf 'Une application existe déjà à %s. La remplacer ? [o/N] ' "$APP_DEST"
  read -r answer
  case "$answer" in
    o|O|oui|Oui|OUI) ;;
    *) echo "Installation annulée."; exit 0 ;;
  esac
fi

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hermy-app.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
JS_SOURCE="$WORK_DIR/Hermy.js"
STAGED_APP="$WORK_DIR/Hermy.app"

python3 - "$PROJECT_DIR" "$JS_SOURCE" <<'PY'
import json
import sys
from pathlib import Path

project = json.dumps(sys.argv[1])
source = f'''var app = Application.currentApplication();
app.includeStandardAdditions = true;
var projectPath = {project};
function shellQuote(value) {{ return "'" + value.replace(/'/g, "'\\\\''") + "'"; }}
var command = "/bin/zsh -c " + shellQuote(
  "cd " + shellQuote(projectPath) +
  " && nohup " + shellQuote(projectPath + "/.venv/bin/python") +
  " " + shellQuote(projectPath + "/run_hermy_background.py") +
  " --gui </dev/null >/dev/null 2>&1 &"
);
try {{
  app.doShellScript(command);
}} catch (error) {{
  app.displayDialog("Impossible de lancer Hermy : " + error.message, {{
    buttons: ["OK"], defaultButton: "OK"
  }});
}}
'''
Path(sys.argv[2]).write_text(source, encoding="utf-8")
PY

osacompile -l JavaScript -o "$STAGED_APP" "$JS_SOURCE"

if [[ -e "$APP_DEST" ]]; then
  if [[ -w /Applications ]]; then
    /bin/rm -rf "$APP_DEST"
  else
    sudo /bin/rm -rf "$APP_DEST"
  fi
fi
if [[ -w /Applications ]]; then
  /usr/bin/ditto "$STAGED_APP" "$APP_DEST"
else
  sudo /usr/bin/ditto "$STAGED_APP" "$APP_DEST"
fi

printf 'Application installée : %s\n' "$APP_DEST"
printf 'Ouvrez Hermy depuis Applications pour la démarrer.\n'
