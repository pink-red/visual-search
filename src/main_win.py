from pathlib import Path
import sys

import pyi_splash
import webview

pyi_splash.update_text("[1/3] Загрузка библиотек...")
import torch
from main import make_app


app_dir = Path(sys.executable).parent
pyi_splash.update_text("[2/3] Загрузка модели и индексов...")
app = make_app(
    model_dir=app_dir / "model",
    device="cpu",
    dtype=torch.float32,
    indices_dir=app_dir / "index",
)
pyi_splash.update_text("[3/3] Загрузка интерфейса...")
url = app.launch(
    prevent_thread_lock=True, show_error=True, show_api=False, server_port=7870
)[1]

webview.create_window("Visual Search", url, maximized=True, text_select=True)
pyi_splash.close()
webview.start(gui="edgechromium")
