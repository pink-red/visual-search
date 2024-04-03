from io import BytesIO
import json
import math
from pathlib import Path
import subprocess

import imagehash
from PIL import Image

from exceptions import MetadataExtractionError
import utils
from utils import log


SCREENSHOT_SIZE = 160
COLUMNS = 5
ROWS = 5


def get_video_duration(video_path: Path) -> float:
    def do_get_duration(seek_to_end: bool):
        command = [
            utils.get_ffmpeg_command("ffprobe"),
            "-hide_banner",

            "-output_format", "json",
            "-show_packets",

            "-select_streams", "v",
            # перематываем в конец файла и читаем 1000 последних пакетов
            # (ffmpeg не дает перемотать в конец напрямую, только через такой
            # хак с огромным таймстампом)
            *(["-read_intervals", "1000:00:00%+#1000"] if seek_to_end else []),

            str(video_path),
        ]
        try:
            res = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                creationflags=utils.no_window_flag(),
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr)
            raise
        packet_infos = json.loads(res.stdout)["packets"]
        packet_infos.reverse()

        # Длина видео - момент отображения последнего кадра (его pts) +
        # длительность отображения этого кадра.
        last_pts_time = next(
            (
                float(x["pts_time"]) + float(x.get("duration_time", 0))
                for x in packet_infos
                if "pts_time" in x
            ),
            None
        )
        if last_pts_time is not None:
            return last_pts_time
        else:
            # Берем примерную длину исходя из dts последнего кадра (момента,
            # когда он должен быть декодирован).
            last_dts_time = next(
                float(x["dts_time"]) + float(x.get("duration_time", 0))
                for x in packet_infos
                if "dts_time" in x
            )
            return last_dts_time

    try:
        try:
            return do_get_duration(seek_to_end=True)
        except (ValueError, StopIteration):
            return do_get_duration(seek_to_end=False)
    except Exception:
        raise MetadataExtractionError(video_path)


def video_phash(video_path: str | Path) -> str:
    sprite = generate_sprite(str(video_path))
    return str(imagehash.phash(sprite))


def generate_sprite_screenshot(video_path: Path, t: float) -> Image.Image:
    def do_generate(fast_seek: bool):
        command = [
            utils.get_ffmpeg_command("ffmpeg"),

            *(
                ["-ss", str(t), "-i", str(video_path)]
                if fast_seek
                else ["-i", str(video_path), "-ss", str(t)]
            ),

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
        if len(res.stdout) == 0:
            log(f"WARNING: Не удалось извлечь кадр из {video_path} t={t}")
            return None
        bio = BytesIO(res.stdout)
        return Image.open(bio)

    try:
        try:
            return do_generate(fast_seek=True)
        except subprocess.CalledProcessError:
            return do_generate(fast_seek=False)
    except Exception:
        raise MetadataExtractionError(video_path)


def combine_images(images: list[Image.Image]) -> Image.Image:
    width, height = images[0].size
    canvas_width = width * COLUMNS
    canvas_height = height * ROWS
    montage = Image.new("RGB", (canvas_width, canvas_height))
    for i, img in enumerate(images):
        x = width * (i % COLUMNS)
        y = height * math.floor(i / ROWS)
        if img is not None:
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
    if len([x for x in images if x is not None]) == 0:
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
