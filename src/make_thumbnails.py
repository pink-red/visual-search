from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import random
import subprocess
from tqdm import tqdm

from future_map import future_map

from data_types import InputImage, Source
import utils
from utils import log


def extract_video_frames_single(
    path: Path,
    relative_path: Path,
    index_dir: Path,
    interval_seconds: int,
    side_size: int,
):
    jpg_q4_path = index_dir / "thumbnails" / relative_path
    jpg_q4_path.mkdir(parents=True)
    png_path = index_dir / "thumbnails-lossless" / relative_path
    png_path.mkdir(parents=True)

    thumb_args = [
        "-vf", (
            f"fps=1/{interval_seconds}:round=up"
            + ",scale=iw*sar:ih,setsar=1"  # фикс неквадратных пикселей
            + f",scale='min({side_size},iw)':min'({side_size},ih)':force_original_aspect_ratio=decrease"
        ),
        "-sws_flags", "area",
        "-start_number", "0",
    ]

    try:
        subprocess.run(
            [
                utils.get_ffmpeg_command("ffmpeg"),
                "-hide_banner",

                "-i", str(path),

                "-q", "4",
                *thumb_args,
                str(jpg_q4_path / "%d.jpg"),

                *thumb_args,
                "-compression_level", "9",
                "-pred", "mixed",
                str(png_path / "%d.png"),
            ],
            check=True,
            capture_output=True,
            creationflags=utils.no_window_flag(),
        )
        return True, path, png_path, index_dir / "thumbnails-lossless"
    except subprocess.CalledProcessError as e:
        if e.returncode == 255:
            # Gradio поймала KeyboardInterrupt и останавливает процесс,
            # завершаемся.
            raise ValueError
        else:
            print(e.stderr.decode(errors="replace"))
            return False, path, png_path, index_dir / "thumbnails-lossless"


def extract_video_frames(
    videos_dir: Path,
    video_paths: list[Path],
    index_dir: Path,
    interval_seconds: int,
    side_size: int,
    num_workers: int,
    progress = None,
):
    video_paths = video_paths.copy()
    random.shuffle(video_paths)

    input_images = []
    with (
        ThreadPoolExecutor(max_workers=num_workers) as executor,
        tqdm(desc="Извлечение кадров", total=len(video_paths), smoothing=0) as tq,
    ):
        if progress is not None:
            progress(
                (tq.n, tq.total),
                desc=(
                    "[2/3] Извлечение кадров "
                    + utils.get_eta_from_tqdm(tq)
                )
            )
        for success, path, out_path, thumbnails_dir in future_map(
            lambda x: executor.submit(
                extract_video_frames_single,
                path=x,
                relative_path=x.relative_to(videos_dir),
                index_dir=index_dir,
                interval_seconds=interval_seconds,
                side_size=side_size,
            ),
            video_paths,
            buffersize=num_workers,
        ):
            if not success:
                log(f"Ошибка извлечения кадров, пропускаем {path}")
            else:
                for frame_path in utils.find_files(out_path, ["png"]):
                    time = timedelta(
                        seconds=int(frame_path.stem) * interval_seconds
                    )
                    input_images.append(
                        InputImage(
                            dir_path=thumbnails_dir,
                            path=frame_path,
                            source=Source(
                                dir_path=videos_dir,
                                path=path,
                                time=time,
                            ),
                        )
                    )
            tq.update(1)
            if progress is not None:
                progress(
                    (tq.n, tq.total),
                    desc=(
                        "[2/3] Извлечение кадров "
                        + utils.get_eta_from_tqdm(tq)
                    )
                )
    return input_images
