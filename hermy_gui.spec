# macOS mascot app. Build with: python -m PyInstaller hermy_gui.spec
from PyInstaller.utils.hooks import collect_all

faster_datas, faster_binaries, faster_hidden = collect_all("faster_whisper")
ctranslate_datas, ctranslate_binaries, ctranslate_hidden = collect_all("ctranslate2")

a = Analysis(
    ["hermy_gui_launcher.py"],
    pathex=[],
    binaries=faster_binaries + ctranslate_binaries,
    datas=faster_datas + ctranslate_datas,
    hiddenimports=faster_hidden + ctranslate_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Hermy",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Hermy",
)
app = BUNDLE(
    coll,
    name="Hermy.app",
    icon="assets/hermy.icns",
    bundle_identifier="dev.ualexon.hermyvoice",
    info_plist={
        "CFBundleName": "Hermy",
        "CFBundleDisplayName": "Hermy",
        "NSMicrophoneUsageDescription": (
            "Hermy utilise le micro pour écouter tes commandes vocales."
        ),
    },
)
