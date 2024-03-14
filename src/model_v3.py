import json
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
from PIL import Image
import timm
from timm.data import create_transform, resolve_data_config
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from extract_single_classifier_torch import extract_single_classifier
import utils


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


class ImageDataset(Dataset):
    def __init__(self, image_paths: list[Path | str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path)
        except PIL.UnidentifiedImageError:
            return None
        image = pil_ensure_rgb(image)
        image = pil_pad_square(image)
        image = self.transform(image)
        # Convert PIL-native RGB to BGR
        image = image[[2, 1, 0]]

        return image_path, image


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)


class ModelV3:
    name = "hf_hub:SmilingWolf/wd-swinv2-tagger-v3@v1"

    def __init__(
        self, path: Path | str, device = "cpu", dtype = torch.float32
    ):
        self.device = device
        self.dtype = dtype

        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        self.model = timm.create_model(
            config["architecture"],
            pretrained=True,
            pretrained_cfg=config["pretrained_cfg"],
            pretrained_cfg_overlay=dict(file=path / "model.safetensors"),
            global_pool=config["global_pool"],
            **config["model_args"],
        ).to(device, dtype)
        self.model.eval()

        self.tags = pd.read_csv(path / "selected_tags.csv")
        with open(path / "aliases.json") as f:
            self.aliases = json.load(f)

        self.embs_dim = self.model.num_features
        self.model_height = self.model.pretrained_cfg["input_size"][-1]

        # TODO
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    @torch.no_grad()
    def images_to_embeddings(self, x):
        x = self.model.forward_features(x)
        x = self.model.head.global_pool(x)
        return x.cpu().numpy()

    @torch.no_grad()
    def embeddings_to_probs(self, x):
        x = self.model.head.fc(x)
        x = self.model.head.flatten(x)
        # почему-то в timm нет сигмоиды в конце модели
        x = torch.nn.functional.sigmoid(x)
        return x.cpu().numpy()

    @torch.no_grad()
    def embeddings_to_single_tag_probs(self, tag_name, embeddings):
        # TODO: cuda?
        ts = self.tags["name"]
        single_tag_model = extract_single_classifier(
            self.model,
            ts.index.get_loc(ts.index[ts == tag_name][0]),
        )
        # TODO: batches
        probs = single_tag_model(torch.from_numpy(embeddings))
        probs = torch.nn.functional.sigmoid(probs)
        return probs.numpy()[:, 0]

    def batch_embed_images(
        self, image_paths: list[Path | str], batch_size: int, progress = None
    ):
        loaded_paths = []
        def embeddings_generator():
            data_loader = DataLoader(
                ImageDataset(
                    image_paths=image_paths, transform=self.transform
                ),
                batch_size=batch_size,
                num_workers=batch_size,  # FIXME
                collate_fn=collate_fn,
            )
            with tqdm(
                desc="Вычисление эмбеддингов", total=len(image_paths)
            ) as tq:
                if progress is not None:
                    progress(
                        (tq.n, tq.total),
                        desc=(
                            "[3/3] Вычисление эмбеддингов "
                            + utils.get_eta_from_tqdm(tq)
                        ),
                    )
                for paths, images in data_loader:
                    loaded_paths.extend(paths)
                    yield from self.images_to_embeddings(
                        images.to(self.device, self.dtype)
                    )
                    tq.update(len(paths))
                    if progress is not None:
                        progress(
                            (tq.n, tq.total),
                            desc=(
                                "[3/3] Вычисление эмбеддингов "
                                + utils.get_eta_from_tqdm(tq)
                            ),
                        )

        embeddings = np.fromiter(
            embeddings_generator(), dtype=(np.float32, self.embs_dim)
        )

        return embeddings, loaded_paths

    def preprocess_single(self, image_path):
        dataset = ImageDataset(
            image_paths=[image_path], transform=self.transform
        )
        _image_path, image = dataset[0]

        image = image.unsqueeze(0)
        image = image.to(self.device, self.dtype)
        return image
