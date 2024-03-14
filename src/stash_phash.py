from io import BytesIO
import json
import math
from pathlib import Path
import subprocess

import imagehash
from PIL import Image

import utils


SCREENSHOT_SIZE = 160
COLUMNS = 5
ROWS = 5


def get_video_duration(video_path: Path) -> float:
    def do_get_duration(seek_to_end: bool):
        command = [
            utils.get_ffmpeg_command("ffprobe"),

            "-hide_banner",
            "-loglevel", "error",

            "-output_format", "json",
            "-show_entries", "packet=pts_time",

            *(["-read_intervals", "9999999%+#1000"] if seek_to_end else []),

            str(video_path),
        ]
        res = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            creationflags=utils.no_window_flag(),
        )
        packet_infos = json.loads(res.stdout)["packets"]
        packet_infos.reverse()
        last_pts_time = next(
            x["pts_time"]
            for x in packet_infos
            if "pts_time" in x
        )
        return float(last_pts_time)

    try:
        try:
            return do_get_duration(seek_to_end=True)
        except (ValueError, StopIteration):
            return do_get_duration(seek_to_end=False)
    except Exception:
        raise ValueError(video_path)


def video_phash(video_path: str | Path) -> str:
    sprite = generate_sprite(str(video_path))
    return str(imagehash.phash(sprite))


def generate_sprite_screenshot(video_path: Path, t: float) -> Image.Image:
    command = [
        utils.get_ffmpeg_command("ffmpeg"),

        "-hide_banner",
#        "-loglevel", "error",

        "-ss", str(t),
        "-i", str(video_path),

        "-frames:v", "1",
        "-vf", f"scale={SCREENSHOT_SIZE}:{-2}",

        "-c:v", "bmp",
        "-f", "image2",
        "-",
    ]
    res = subprocess.run(
        command,
        check=True,
        capture_output=True,
        creationflags=utils.no_window_flag(),
    )
    bio = BytesIO(res.stdout)
    return Image.open(bio)


def combine_images(images: list[Image.Image]) -> Image.Image:
    width, height = images[0].size
    canvas_width = width * COLUMNS
    canvas_height = height * ROWS
    montage = Image.new("RGB", (canvas_width, canvas_height))
    for i, img in enumerate(images):
        x = width * (i % COLUMNS)
        y = height * math.floor(i / ROWS)
        montage.paste(img, (x, y))
    return montage


def generate_sprite(video_path: Path) -> Image.Image:
    # Generate sprite image offset by 5% on each end to avoid intro/outros
    chunk_count = COLUMNS * ROWS
    video_duration = get_video_duration(video_path)
    offset = 0.05 * video_duration
    step_size = (0.9 * video_duration) / chunk_count
    images = []
    for i in range(chunk_count):
        time = offset + (i * step_size)
        img = generate_sprite_screenshot(video_path, time)
        images.append(img)
    # Combine all of the thumbnails into a sprite image
    if len(images) == 0:
        raise ValueError(f"images list is empty, failed to generate phash sprite for {video_path}")
    return combine_images(images)


def phash_distance(a: str, b: str) -> int:
    a = imagehash.hex_to_hash(a)
    b = imagehash.hex_to_hash(b)
    return a - b


def main():
    import sys
    if len(sys.argv) == 3:
        _, video_path, target_phash = sys.argv
    elif len(sys.argv) == 2:
        _, video_path = sys.argv
        target_phash = None
    else:
        raise ValueError(sys.argv)

    phash = video_phash(video_path)
    print(phash)
    if target_phash is not None:
        print(phash_distance(phash, target_phash))


if __name__ == "__main__":
    main()
