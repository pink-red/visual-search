import argparse

from PyInstaller.utils.hooks import collect_data_files


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
options = parser.parse_args()


datas = []
datas += collect_data_files('gradio_client')
datas += collect_data_files('gradio')


main_a = Analysis(
    ['main_win.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
#    noarchive=False,
    noarchive=True,
    module_collection_mode={
        # См.: https://github.com/pyinstaller/pyinstaller/issues/8108
        'gradio': 'py',
    },
)
main_pyz = PYZ(main_a.pure)
main_splash = Splash(
    'splash.png',
    binaries=main_a.binaries,
    datas=main_a.datas,
    max_img_size=(760, 480),
    text_pos=(10, 480 - 10),
    text_size=12,
    text_color='white',
    text_default='Загрузка...',
    always_on_top=False,
)
main_exe = EXE(
    main_pyz,
    main_splash,
    main_a.scripts,
    [],
    exclude_binaries=True,
    name='Search',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
#    upx=True,
    upx=False,
    console=options.debug,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['main.ico'],
)

create_index_a = Analysis(
    ['create_index_win.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
#    noarchive=False,
    noarchive=True,
    module_collection_mode={
        # См.: https://github.com/pyinstaller/pyinstaller/issues/8108
        'gradio': 'py',
    },
)
create_index_pyz = PYZ(create_index_a.pure)
create_index_splash = Splash(
    'splash.png',
    binaries=create_index_a.binaries,
    datas=create_index_a.datas,
    max_img_size=(760, 480),
    text_pos=(10, 480 - 10),
    text_size=12,
    text_color='white',
    text_default='Загрузка...',
    always_on_top=False,
)
create_index_exe = EXE(
    create_index_pyz,
    create_index_splash,
    create_index_a.scripts,
    [],
    exclude_binaries=True,
    name='Index Creator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
#    upx=True,
    upx=False,
    console=options.debug,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['create_index.ico'],
)


coll = COLLECT(
    main_exe,
    main_splash.binaries,
    main_a.binaries,
    main_a.datas,

    create_index_exe,
    create_index_splash.binaries,
    create_index_a.binaries,
    create_index_a.datas,

    strip=False,
#    upx=True,
    upx=False,
    upx_exclude=[],
    name='Search',
)
