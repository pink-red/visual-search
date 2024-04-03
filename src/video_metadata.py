import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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

from exceptions import MetadataExtractionError
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


def extract_media_metadata(path: Path):
    try:
        res = subprocess.run(
            [
                utils.get_ffmpeg_command("ffprobe"),
                "-hide_banner",
                "-output_format", "json",
                "-show_streams",
                "-show_format",
                str(path),
            ],
            check=True,
            capture_output=True,
            creationflags=utils.no_window_flag(),
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr.decode(errors="replace"))
        raise MetadataExtractionError(path)
    res = json.loads(res.stdout)
    video = next(x for x in res["streams"] if x["codec_type"] == "video")
    audio = next(
        (x for x in res["streams"] if x["codec_type"] == "audio"), None
    )

    try:
        if audio is not None:
            audio_md = {
                "codec": audio["codec_name"],
                "channels": audio["channels"],
                "sample_rate": int(audio["sample_rate"]),
                "bit_rate": map_optional(int, audio.get("bit_rate")),
            }
        else:
            audio_md = None

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
                "bit_rate": map_optional(int, video.get("bit_rate")),
                "codec": video["codec_name"],
            },
            "audio": audio_md,
        }
    except KeyError:
        raise ValueError(path)


@dataclass
class FileMetadata:
    metadata: dict


@dataclass
class SkippedFileMetadata:
    metadata: dict


def extract_file_metadata(path: Path):
    try:
        oshash = make_oshash(path)
    except ValueError:
        oshash = None

    try:
        phash = video_phash(path)
    except MetadataExtractionError:
        phash = None

    try:
        media_metadata = extract_media_metadata(path)
    except MetadataExtractionError:
        media_metadata = None

    metadata = {
        "file_size": path.stat().st_size,
        "hashes": {
            "oshash": oshash,
            "md5": make_md5(path),
            "phash": phash,
        },
        "media": media_metadata,
    }
    if phash is None:
        metadata["reason"] = "PhashError"
        return path, SkippedFileMetadata(metadata)
    elif media_metadata is None:
        metadata["reason"] = "MediaMetadataError"
        return path, SkippedFileMetadata(metadata)
    else:
        return path, FileMetadata(metadata)


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
    skipped_files = {}
    ok_paths = []
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
            rel_path = str(path.relative_to(videos_dir).as_posix())
            if isinstance(file_metadata, FileMetadata):
                metadata_by_file[rel_path] = {
                    **file_metadata.metadata,
                    "url": url,
                }
                ok_paths.append(path)
            else:
                skipped_files[rel_path] = {
                    **file_metadata.metadata,
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
    return metadata_by_file, skipped_files, ok_paths
