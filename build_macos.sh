#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Erreur : le bundle macOS doit être construit sur un Mac." >&2
    exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
    echo "PyInstaller est absent. Installe les dépendances de build avec :" >&2
    echo "  $PYTHON_BIN -m pip install -r requirements-build-macos.txt" >&2
    exit 1
fi

mkdir -p build/macos/gui build/macos/console build/macos/specs dist/macos
export PYINSTALLER_CONFIG_DIR="$PROJECT_DIR/build/pyinstaller-config"
mkdir -p "$PYINSTALLER_CONFIG_DIR"
"$PYTHON_BIN" create_macos_icon.py

echo "Construction de l’application GUI Hermy.app…"
"$PYTHON_BIN" -m PyInstaller \
    --noconfirm --clean \
    --distpath dist/macos --workpath build/macos/gui \
    hermy_gui.spec

echo "Construction du lanceur console HermyConsole…"
"$PYTHON_BIN" -m PyInstaller \
    --noconfirm --clean --onedir --console \
    --distpath dist/macos --workpath build/macos/console \
    --specpath build/macos/specs \
    --name HermyConsole \
    --collect-all faster_whisper \
    --collect-all ctranslate2 \
    hermy_voice.py

echo
echo "Bundles créés dans : $PROJECT_DIR/dist/macos"
echo "- GUI : dist/macos/Hermy.app (double-clic dans le Finder)"
echo "- Console : dist/macos/HermyConsole/HermyConsole (à lancer depuis Terminal)"
