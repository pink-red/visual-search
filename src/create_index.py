import argparse
from datetime import datetime, timezone
import json
from tempfile import TemporaryDirectory
import torch
from tqdm import tqdm
import shutil
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_STORED

import gradio as gr
import pandas as pd
import safetensors.numpy

from data_types import InputImage
from make_thumbnails import extract_video_frames
from model_v3 import ModelV3
from video_metadata import extract_metadata
import utils


def save_index(
    path: Path, embeddings, input_images: list[InputImage], metadata
):
    with open(path / "thumbnail_paths.json", "w") as f:
        json.dump(
            [
                str(x.relative_path.with_suffix("").as_posix())
                for x in input_images
            ],
            f,
        )
    safetensors.numpy.save_file(
        {"embeddings": embeddings},
        path / "embeddings.safetensors",
    )

    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    index_file_path = path / (metadata["name"] + ".vindex")
    with ZipFile(
        index_file_path,
        "x",
        # Бинарный индекс и уже сжатые картинки практически не сжимаются и
        # сжатие замедляет извлечение, поэтому не используем его.
        compression=ZIP_STORED,
    ) as archive:
        archive.write(path / "metadata.json", arcname="metadata.json")
        archive.write(
            path / "embeddings.safetensors", arcname="embeddings.safetensors"
        )
        archive.write(
            path / "thumbnail_paths.json", arcname="thumbnail_paths.json"
        )
        thumbnails_dir = path / "thumbnails"
        for x in thumbnails_dir.glob("**/*"):
            archive.write(x, arcname=str(x.relative_to(path)))

    lossless_thumnails_zip_path = index_file_path.with_name(
        index_file_path.stem + "-lossless-thumbnails.zip"
    )
    with ZipFile(
        lossless_thumnails_zip_path, "x", compression=ZIP_STORED
    ) as archive:
        lossless_thumbnails_dir = path / "thumbnails-lossless"
        for x in lossless_thumbnails_dir.glob("**/*"):
            archive.write(
                x, arcname=str(x.relative_to(lossless_thumbnails_dir))
            )

    return index_file_path, lossless_thumnails_zip_path


def create_index(
    input_dir: str | Path,
    output_dir: str | Path,
    model_dir: str | Path,
    url: str,
    interval_seconds: int = 10,
    num_workers: int = 1,
    num_metadata_workers: int = 1,
    use_cuda: bool = False,
    batch_size: int = 1,
    progress = None,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    model_dir = Path(model_dir)

    index_name = input_dir.name

    output_index_dir = output_dir / index_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_index_dir.mkdir(parents=True, exist_ok=True)
    is_empty = not any(output_index_dir.iterdir())
    if not is_empty:
        raise ValueError(f"Папка непуста: {output_index_dir}")

    if url is not None:
        url = url.strip()

    if progress is not None:
        progress((0, 1), desc="Загрузка модели")
    model = ModelV3(
        path=model_dir,
        device="cuda" if use_cuda else "cpu",
        # TODO: можно ли использовать half-тип на cpu?
        dtype=torch.float16 if use_cuda else torch.float32,
    )
    if progress is not None:
        progress((1, 1), desc="Загрузка модели")

    files_metadata = extract_metadata(
        videos_dir=input_dir,
        video_paths=utils.find_animated(
            input_dir, include_gifs=False  # FIXME: захардкожено
        ),
        url=url,
        num_workers=num_metadata_workers,
        progress=progress,
    )
    if not files_metadata:
        raise ValueError(f"Не найдено ни одного видео в папке {input_dir}")

    with TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        input_images = extract_video_frames(
            videos_dir=input_dir,
            include_gifs=False,  # FIXME: захардкожено
            index_dir=tmp_dir,
            interval_seconds=interval_seconds,
            side_size=448,  # FIXME: захардкожено
            num_workers=num_workers,
            progress=progress,
        )
        input_images.sort(key=lambda x: x.source)
        image_paths = [str(x.path) for x in input_images]

        embeddings, loaded_paths = model.batch_embed_images(
            image_paths=image_paths, batch_size=batch_size, progress=progress,
        )

        metadata = {
            "version": 1,
            "name": index_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "index_type": "videos",
            "thumbnails_ext": "jpg",
            "model_name": model.name,
            "settings": {
                "interval_seconds": interval_seconds,
            },
            "files": files_metadata,
        }

        index_file_path, lossless_thumnails_zip_path = save_index(
            path=tmp_dir,
            embeddings=embeddings,
            input_images=input_images,
            metadata=metadata,
        )

        shutil.copy(index_file_path, output_index_dir)
        shutil.copy(lossless_thumnails_zip_path, output_index_dir)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--interval-seconds", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-metadata-workers", type=int, default=1)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    create_index(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        url=args.url,
        interval_seconds=args.interval_seconds,
        num_workers=args.num_workers,
        num_metadata_workers=args.num_metadata_workers,
        use_cuda=args.use_cuda,
        batch_size=args.batch_size,
    )


def make_app(
    model_dir: str | Path,
    output_dir: str | Path,
    use_cuda: bool,
    batch_size: int,
):
    def do_create_index(
        input_dir,
        url,
        interval_seconds,
        num_workers,
        num_metadata_workers,
        progress = gr.Progress(),
    ):
        input_dir = input_dir.strip()
        if not input_dir:
            raise ValueError("Укажите путь к папке с видео")

        url = url.strip()
        if not url:
            url = None

        create_index(
            model_dir=model_dir,
            use_cuda=use_cuda,
            batch_size=batch_size,
            input_dir=input_dir,
            output_dir=output_dir,
            url=url,
            interval_seconds=int(interval_seconds),
            num_workers=int(num_workers),
            num_metadata_workers=int(num_metadata_workers),
            progress=progress,
        )

        return """
            <div style="text-align:center;">
                <h2>Готово!</h2>
                Индекс находится в папке "created index".
            </div>
        """

    with gr.Blocks(
        analytics_enabled=False,
        css="""
            #form-wrapper {
                width: 500px;
                margin: auto;
            }

            /* Скрываем дефолтный прогресс Gradio */
            #progress-el .progress-text {
                display: none;
            }
        """,
    ) as app:
        with gr.Column(elem_id="form-wrapper", variant="panel"):
            input_dir = gr.Textbox(label="Путь к папке с видео", max_lines=1)
            url = gr.Textbox(
                label="Ссылка на раздачу-источник видео",
                info="Будет отображаться для всех видео из этого индекса. Опционально.",
                max_lines=1,
            )
            with gr.Column():
                interval_seconds = gr.Slider(
                    label="Интервал, сек.",
                    info=(
                        'Для "обычных" видео, оставьте значение по умолчанию.'
                        + ' Для более динамичных, уменьшите интервал.'
                    ),
                    minimum=1,
                    maximum=10,
                    value=10,
                    step=1,
                )
                with gr.Accordion("Подробнее про интервал", open=False):
                    gr.Markdown(
                        "10 секунд - хороший интервал для большинства видео,"
                        + " потому что за это время ситуация/ракурс камеры"
                        + " мало успевают поменяться. Если же видео более"
                        + " динамичные, то интервал стоит уменьшить.\n"
                        + "\n"
                        + "При этом, пропорционально увеличится размер"
                        + " индекса. Например, индекс с интервалом 5 с. будет"
                        + " весить в 2 раза больше индекса с интервалом 10 с."
                        + " (кадров будет в 2 раза больше)."
                    )
            num_metadata_workers = gr.Slider(
                label="Кол-во потоков для извлечения метаданных",
                info=(
                    "Если видео лежат на HDD, ставьте равным 1."
                    + " Если на SSD, ставьте равным количеству ядер."
                ),
                minimum=1,
                maximum=16,
                value=1,
                step=1,
            )
            num_workers = gr.Slider(
                label="Кол-во обработчиков для извлечения кадров",
                info=(
                    "Если видео лежат на HDD, ставьте равным 1."
                    + " Если на SSD, то ставьте в 4 раза меньше, чем у вас"
                    + " ядер, поскольку один обработчик загружает 3-4 ядра."
                ),
                minimum=1,
                maximum=8,
                value=1,
                step=1,
            )
            process = gr.Button("Создать индекс", variant="primary")
            progress_el = gr.HTML(elem_id="progress-el")

        process.click(
            do_create_index,
            inputs=[
                input_dir,
                url,
                interval_seconds,
                num_workers,
                num_metadata_workers,
            ],
            outputs=progress_el,
        )

    return app.queue()


def main():
    _, model_dir, output_dir = sys.argv
    app = make_app(
        model_dir=model_dir,
        output_dir=output_dir,
        use_cuda=True,
        batch_size=16,
    )
    app.launch(show_error=True, show_api=False)


if __name__ == "__main__":
    main()
