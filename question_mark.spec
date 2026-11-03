# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build for the RLBot botpack (bob.toml): one folder, QuestionMark/QuestionMark(.exe) + _internal/ with
# the ONNX policy.
a = Analysis(
    ['bot.py'],
    pathex=['.'],
    binaries=[],
    datas=[('policy', 'policy'), ('LICENSE', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='QuestionMark', debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='QuestionMark')
