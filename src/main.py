import html
from pathlib import Path
import sys

import gradio as gr
import torch

from model_v3 import ModelV3
import pandas as pd
from search import Search
import utils


class SearchFrontend:
    def __init__(self, search: Search):
        self.search = search
        self.current_matches = None

    def search_by_tags(self, *args, **kwargs):
        self.current_matches, results = self.search.search_by_tags(*args, **kwargs)
        return results

    def search_by_image(self, *args, **kwargs):
        self.current_matches, results = self.search.search_by_image(*args, **kwargs)
        return results

    def get_metadata(self, evt: gr.SelectData):
        match = self.current_matches.iloc[evt.index]

        def h(x):
            return html.escape(str(x))
        return (
            f"<b>Путь:</b> {h(match.source_path)}<br/>"
            + f"<b>Время в видео:</b> {utils.format_timestamp(match.source_time)}<br/>"
            + f"<b>Название индекса:</b> {h(match.index_name)}<br/>"
            + f"<b>PHash:</b> {h(match.source_phash)}<br/>"
            + (
                f'<b>URL:</b> <a href="{h(match.url)}">{h(match.url)}</a><br/>'
                if not pd.isnull(match.url)  # pandas category
                else ""
            )
        )


def make_app(
    model_dir: Path, device: str, dtype: torch.dtype, indices_dir: Path
):
    search_frontend = SearchFrontend(
        Search(
            model=ModelV3(path=model_dir, device=device, dtype=dtype),
            indices_dir=indices_dir,
        )
    )

    with gr.Blocks(
        analytics_enabled=False,
        delete_cache=(86400, 86400),
        css="""
            .thumbnail-item {
                aspect-ratio: auto;
                display: flex;
                flex-direction: column;
            }
            .thumbnail-item > img {
                aspect-ratio: 16/9;
            }
            .thumbnail-item > .caption-label {
                position: unset;
                max-width: 100%;
            }
            .thumbnail-item:hover > .caption-label {
                opacity: 1.0;
            }

            #upload-image .upload-container {
                width: 100%;
            }
            #upload-image .upload-container .wrap {
                flex-direction: row;
                padding: 0;
                gap: 5px;
            }
            #upload-image .upload-container .wrap .icon-wrap {
                margin-bottom: unset;
            }
            #upload-image .upload-container .image-frame img {
                object-fit: contain;
            }
        """,
        js="""
            function () {
                function doReplace() {
                    const el = document.querySelector("#upload-image .upload-container > button > .wrap")
                    const iconHtml = el.querySelector(".icon-wrap").outerHTML
                    el.innerHTML = iconHtml + "Поиск по картинке"
                    el.classList.add("replaced")
                }

                doReplace()

                const observer = new MutationObserver(mutations => {
                    if (document.querySelector("#upload-image .upload-container > button > .wrap:not(.replaced)")) {
                        doReplace()
                    }
                })
                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                })
            }
        """,
    ) as app:
        with gr.Row():
            query_textbox = gr.Textbox(
                show_label=False,
                container=False,
                max_lines=1,
                scale=67,
            )
            search_btn = gr.Button(
                "🔍", scale=3, min_width=20, size="sm"
            )
            upload_image = gr.Image(
                show_label=False,
                type="filepath",
                sources=["upload"],
                elem_id="upload-image",
                height=45,
                scale=15,
            )
            group_by_video_checkbox = gr.Checkbox(
                label="Группировать по видео",
                value=True,
                scale=15,
            )
        found_images = gr.Gallery(
            # по умолчанию, отображаем кадры, в которых есть люди
            value=search_frontend.search_by_tags(
                query="-no_humans", group_by_video=True
            ),
            show_label=False,
            columns=5,
            height=650,
            object_fit="contain",
        )
        metadata_el = gr.HTML()

        upload_image.upload(
            search_frontend.search_by_image,
            inputs=[upload_image, group_by_video_checkbox],
            outputs=found_images,
        )
        for on_event in [query_textbox.submit, search_btn.click]:
            on_event(
                search_frontend.search_by_tags,
                inputs=[query_textbox, group_by_video_checkbox],
                outputs=found_images,
            )
        found_images.select(
            search_frontend.get_metadata,
            outputs=metadata_el,
            show_progress="hidden",
        )
    return app


def main():
    _, model_dir, indices_dir = sys.argv
    model_dir = Path(model_dir)
    indices_dir = Path(indices_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    app = make_app(
        model_dir=model_dir,
        device=device,
        dtype=dtype,
        indices_dir=indices_dir,
    )
    app.launch(show_error=True, show_api=False)


if __name__ == "__main__":
    main()
