import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PosixPath
import random
import re
import subprocess
from urllib.parse import urljoin

from future_map import future_map
from natsort import natsorted
from oshash import oshash as make_oshash

from stash_phash import video_phash, get_video_duration
from tqdm import tqdm
import utils
from utils import log


def make_md5(file_path) -> str:
    with open(file_path, "rb", buffering=0) as f:
        return hashlib.file_digest(f, "md5").hexdigest()


def map_optional(f, x):
    if x is None:
        return None
    else:
        return f(x)


def approximate_video_bit_rate(ffprobe_info) -> int:
    video = next(
        x for x in ffprobe_info["streams"] if x["codec_type"] == "video"
    )

    total_bit_rate = int(ffprobe_info["format"]["bit_rate"])
    other_streams_bit_rate = 0
    for stream in ffprobe_info["streams"]:
        if stream["index"] == video["index"]:
            continue
        other_streams_bit_rate += int(stream["bit_rate"])

    return total_bit_rate - other_streams_bit_rate


def extract_media_metadata(path: Path):
    res = subprocess.run(
        [
            utils.get_ffmpeg_command("ffprobe"),
            "-loglevel", "error",
            "-output_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        creationflags=utils.no_window_flag(),
    )
    res = json.loads(res.stdout)
    video = next(x for x in res["streams"] if x["codec_type"] == "video")
    audio = next(x for x in res["streams"] if x["codec_type"] == "audio")

    try:
        return {
            "duration": get_video_duration(path),
            "video": {
                "width": video["width"],
                "height": video["height"],
                "display_aspect_ratio": video.get("display_aspect_ratio"),
                "fps": (
                    float(Fraction(video["avg_frame_rate"]))
                    if video["avg_frame_rate"] != "0/0"
                    else None
                ),
                "bit_rate": (
                    int(video["bit_rate"])
                    if "bit_rate" in video
                    else approximate_video_bit_rate(res)
                ),
                "codec": video["codec_name"],
            },
            "audio": {
                "codec": audio["codec_name"],
                "channels": audio["channels"],
                "sample_rate": int(audio["sample_rate"]),
                "bit_rate": int(audio["bit_rate"]),
            },
        }
    except KeyError:
        raise ValueError(path)


def extract_file_metadata(path: Path):
    metadata = {
        "file_size": path.stat().st_size,
        "hashes": {
            "oshash": make_oshash(path),
            "md5": make_md5(path),
            "phash": video_phash(path),
        },
        "media": extract_media_metadata(path),
    }
    return path, metadata


def extract_metadata(
    videos_dir: Path,
    video_paths: list[Path],
    url: str | None,
    num_workers: int,
    progress = None,
):
    if url is not None:
        url = url.strip()

    video_paths = video_paths.copy()
    # для более точного оставшегося времени прогресс-бара
    random.shuffle(video_paths)

    metadata_by_file = {}
    with (
        ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(
            desc="Извлечение метаданных", total=len(video_paths), smoothing=0
        ) as tq,
    ):
        if progress is not None:
            progress(
                (tq.n, tq.total),
                desc=(
                    "[1/3] Извлечение метаданных "
                    + utils.get_eta_from_tqdm(tq)
                ),
            )
        for path, file_metadata in future_map(
            lambda path: executor.submit(extract_file_metadata, path),
            video_paths,
            buffersize=num_workers,
        ):
            rel_path = path.relative_to(videos_dir)
            metadata_by_file[str(rel_path.as_posix())] = {
                **file_metadata,
                "url": url,
            }
            tq.update(1)
            if progress is not None:
                progress(
                    (tq.n, tq.total),
                    desc=(
                        "[1/3] Извлечение метаданных "
                        + utils.get_eta_from_tqdm(tq)
                    ),
                )
    metadata_by_file = dict(
        natsorted(metadata_by_file.items(), key=lambda x: x[0].lower())
    )
    return metadata_by_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos_dir", type=Path)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()

    video_paths = utils.find_animated(
        args.videos_dir,
        include_gifs=False,  # FIXME: захардкожено
    )
    metadata = extract_metadata(
        videos_dir=args.videos_dir,
        video_paths=video_paths,
        url=args.url,
        num_workers=args.num_workers,
    )
    print(json.dumps(metadata, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
