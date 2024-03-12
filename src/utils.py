from datetime import timedelta
from pathlib import Path
import subprocess
import sys

from PIL import Image


VIDEO_EXTS = [
    "asf",
    "avi",
    "divx",
    "flv",
    "m4v",
    "mkv",
    "mov",
    "mp4",
    "mpeg",
    "mpg",
    "ogv",
    "rm",
    "rmvb",
    "webm",
    "wmv",
]


def find_files(path: Path, exts: list[str]):
    image_paths = []
    ignored_exts = set()
    for p in path.glob("**/*"):
        if not p.is_file():
            continue

        if p.suffix[1:].lower() in exts:
            image_paths.append(p)
        else:
            if p.suffix:
                ignored_exts.add(p.suffix[1:])
            else:
                log(
                    "WARNING: У файла нет расширения, поэтому он пропущен: "
                    + str(p)
                )

    if ignored_exts:
        log(
            f"Файлы со следующими расширениями были проигнорированы: "
            f"{', '.join(sorted(ignored_exts))}"
        )

    return image_paths


def find_animated(path: Path, include_gifs: bool):
    if include_gifs:
        exts = VIDEO_EXTS + ["gif"]
    else:
        exts = VIDEO_EXTS
    video_paths = []
    for p in find_files(path, exts):
        if p.suffix[1:].lower() == "gif":
            with Image.open(p) as image:
                if image.is_animated:
                    video_paths.append(p)
        else:
            video_paths.append(p)
    return video_paths


def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def format_timestamp(t: timedelta) -> str:
    hours, remainder = divmod(t.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    hours = round(hours)
    minutes = round(minutes)
    seconds = round(seconds)

    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_ffmpeg_command(command):
    if command not in ["ffmpeg", "ffprobe"]:
        raise ValueError(command)

    if getattr(sys, "frozen", False):
        return (
            Path(sys.executable).parent / "ffmpeg" / "bin" / f"{command}.exe"
        )
    else:
        return command


def no_window_flag():
    if getattr(sys, "frozen", False):
        return subprocess.CREATE_NO_WINDOW
    else:
        return 0


def get_eta_from_tqdm(tq):
    elapsed = tq.format_dict["elapsed"]
    elapsed_str = tq.format_interval(elapsed)

    if tq.n and elapsed:
        rate = tq.n / elapsed
        rate_str = (
            f"{rate:.2f}it/s"
            if rate >= 1
            else f"{1 / rate:.2f}s/it"
        )
    else:
        rate = None
        rate_str = "?it/s"

    remaining = (tq.total - tq.n) / rate if rate and tq.total else None
    remaining_str = (
        tq.format_interval(remaining)
        if remaining is not None
        else "?"
    )

    total = tq.total if tq.total else "?"

    return f"| {tq.n}/{total} [{elapsed_str}<{remaining_str}, {rate_str}]"
