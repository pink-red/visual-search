if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    from pathlib import Path
    import sys

    import pyi_splash
    import webview

    pyi_splash.update_text("[1/2] Загрузка библиотек...")
    from create_index import make_app

    app_dir = Path(sys.executable).parent
    app = make_app(
        model_dir=app_dir / "model",
        output_dir=app_dir / "created index",
        use_cuda=False,
        batch_size=1,
    )

    pyi_splash.update_text("[2/2] Загрузка интерфейса...")
    url = app.launch(
        prevent_thread_lock=True,
        show_error=True,
        show_api=False,
        server_port=7871,
    )[1]

    webview.create_window("Visual Index Creator", url, maximized=True)
    pyi_splash.close()
    webview.start(gui="edgechromium")
