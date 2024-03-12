from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import ZipFile

from lru import LRU
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from PIL import Image
import safetensors.numpy
from tqdm import tqdm

from model_v3 import ModelV3
import utils


def is_webp(file_path):
    with open(file_path, "rb") as f:
        header = f.read(12)

    riff_header = header[0:4]
    webp_header = header[8:12]

    is_riff = riff_header == b"RIFF"
    is_fourcc_webp = webp_header == b"WEBP"
    return is_riff and is_fourcc_webp


def cosine_similarity(a, b):
#    return np.dot(query, items.T) / (np.linalg.norm(query) * np.linalg.norm(items, axis=1))
    return np.dot(a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b))


def euclidean_distance(embeddings_matrix, embeddings_vector):
    return np.linalg.norm(embeddings_matrix - embeddings_vector, axis=1)


class IndexError(Exception):
    pass


class UnsupportedIndexType(IndexError):
    pass


class UnsupportedIndexVersion(IndexError):
    pass


class UnsupportedModel(IndexError):
    pass


def load_index(index_path: str | Path):
    index_path = Path(index_path)

    with ZipFile(index_path) as archive:
        with archive.open("metadata.json") as f:
            metadata = json.load(f)

            if metadata["version"] != 1:
                raise UnsupportedIndexVersion(
                    f"Индекс {index_path.name} имеет неподерживаемую версию"
                    + f" ({metadata['version']}) и будет пропущен."
                )

            if metadata["index_type"] != "videos":
                raise UnsupportedIndexType(
                    f"Индекс {index_path.name} имеет неподерживаемый тип"
                    + f" ({metadata['index_type']}) и будет пропущен."
                )

            if metadata["model_name"] != ModelV3.name:
                raise UnsupportedModel(
                    f"Индекс {index_path.name} был создан через"
                    + f" неподдерживаемую модель ({metadata['model_name']})"
                    + f" и будет пропущен."
                )

        with archive.open("embeddings.safetensors") as f:
            embeddings = safetensors.numpy.load(f.read())["embeddings"]
        # пути к превью в том же порядке, что и эмбеддинги
        with archive.open("thumbnail_paths.json") as f:
            thumbnail_paths = json.load(f)

    df = pd.DataFrame(
        {
            "path": thumbnail_path + "." + metadata["thumbnails_ext"],
            "source_path": str(PurePosixPath(thumbnail_path).parent),
            "source_time": timedelta(
                seconds=(
                    int(PurePosixPath(thumbnail_path).name)
                    * metadata["settings"]["interval_seconds"]
                )
            ),
        }
        for thumbnail_path in tqdm(
            thumbnail_paths,
            desc="Загрузка данных о кадрах",
            total=len(thumbnail_paths),
        )
    )

    df["index_path"] = pd.Series(
        str(index_path), index=df.index, dtype="category"
    )
    df["index_name"] = pd.Series(
        metadata["name"], index=df.index, dtype="category"
    )

    df["url"] = None
    for filename, file_md in metadata["files"].items():
        df.loc[df.source_path == filename, "url"] = file_md["url"]
    df["url"] = df["url"].astype("category")

    df["source_phash"] = None
    for filename, file_md in metadata["files"].items():
        df.loc[df.source_path == filename, "source_phash"] = (
            file_md["hashes"]["phash"]
        )
    df["source_phash"] = df["source_phash"].astype("category")

    return df, embeddings, metadata


@dataclass
class QueryTag:
    name: str
    is_positive: bool


@dataclass
class QueryFilter:
    is_positive: bool

    def filter(self, df):
        condition = self._calculate_condition(df)
        if self.is_positive:
            return df[condition]
        else:
            return df[~condition]


#@dataclass
#class QueryFilterFilename(QueryFilter):
#    substring: str
#
#    def _calculate_condition(self, df):
#        return df.source_filename.str.contains(
#            self.substring, case=False, regex=False
#        )


@dataclass
class QueryFilterPath(QueryFilter):
    substring: str

    def _calculate_condition(self, df):
        return df.source_path.str.contains(
            self.substring, case=False, regex=False
        )


@dataclass
class QueryFilterIndexName(QueryFilter):
    substring: str

    def _calculate_condition(self, df):
        return df.index_name.str.contains(
            self.substring, case=False, regex=False
        )


@dataclass
class Query:
    tags: list[QueryTag]
    filters: list[QueryFilter]


def parse_query(query: str) -> Query:
    names = query.strip().split()

    filters_by_name = {
#        "filename": QueryFilterFilename,
        "path": QueryFilterPath,
        "index": QueryFilterIndexName,
    }

    tags = []
    filters = []
    for x in names:
        if x[0] == "-":
            is_positive = False
            x = x[1:]
        else:
            is_positive = True

        if ":" in x and x.split(":")[0].lower() in filters_by_name.keys():
            flt, val = x.split(":")
            if not val:
                raise ValueError(x)
            filters.append(
                filters_by_name[flt](is_positive=is_positive, substring=val)
            )
        else:
            tags.append(
                QueryTag(name=x.lower(), is_positive=is_positive)
            )

    return Query(tags=tags, filters=filters)


class Search:
    def __init__(self, model: ModelV3, indices_dir: Path):
        self.model = model
        self.tag_probs_cache = LRU(20)  # FIXME: захардкожено

        dataframes = []
        embedding_matrices = []
        for index_path in indices_dir.glob("**/*"):
            if index_path.suffix.lower() != ".vindex":
                continue
            try:
                df, embeddings, _metadata = load_index(index_path)
            except IndexError as e:
                print(f"WARNING: {e.message}")
                continue
            dataframes.append(df)
            embedding_matrices.append(embeddings)

        # объединяем категории чтобы снизить потребление оперативки
        for col_name in dataframes[0].select_dtypes("category").columns:
            uc = union_categoricals([d[col_name] for d in dataframes])
            for d in dataframes:
                d[col_name] = pd.Categorical(
                    d[col_name],
                    categories=uc.categories,
                )

        self.df = pd.concat(dataframes, axis=0, ignore_index=True, copy=False)
        self.embeddings = np.concatenate(embedding_matrices, axis=0)

    def search_by_image(self, image_path: str, group_by_video: bool):
        image_path = Path(image_path)

        try:
            if is_webp(image_path):
                with NamedTemporaryFile(suffix=".bmp") as f:
                    with tqdm(
                        desc="Конвертация картинки из webp", total=1
                    ) as tq:
                        tmp_img = Image.open(image_path)
                        tmp_img.save(f.name)
                        tq.update(1)
                    with tqdm(desc="Препроцессинг картинки", total=1) as tq:
                        image = self.model.preprocess_single(f.name)
                        tq.update(1)
            else:
                with tqdm(desc="Препроцессинг картинки", total=1) as tq:
                    image = self.model.preprocess_single(image_path)
                    tq.update(1)
        finally:
            image_path.unlink()
            # удаляем папку, если там был только наш файл
            try:
                image_path.parent.rmdir()
            except OSError:
                # в папке не только наш файл, не удаляем её
                pass

        with tqdm(desc="Конвертируем картинку в эмбеддинги", total=1) as tq:
            query_embeddings = self.model.images_to_embeddings(image)[0]
            tq.update(1)
        score = cosine_similarity(self.embeddings, query_embeddings)

        matches = self.df.copy(deep=False)
        matches["score"] = score
        return self._top_results(
            matches=matches, group_by_video=group_by_video
        )

    def search_by_tags(self, query: str, group_by_video: bool):
        matches = self.df.copy(deep=False)

        query = parse_query(query)

        unknown_tags = (
            set(t.name for t in query.tags) - set(self.model.tags["name"])
        )
        if unknown_tags:
            raise ValueError(
                "Неизвестные теги: " + ", ".join(sorted(unknown_tags))
            )

        for flt in query.filters:
            matches = flt.filter(matches)

        if not query.tags:
            score = 1.0
        else:
            target_probs = [1.0 if t.is_positive else 0.0 for t in query.tags]
            tag_names = [t.name for t in query.tags]

            # делаем пустой DataFrame и затем наполняем
            matches_tags = pd.DataFrame({tag_names[0]: []})
            ts = self.model.tags["name"]
            for tag_name in tag_names:
                try:
                    matches_tags[tag_name] = self.tag_probs_cache[tag_name]
                except KeyError:
                    probs = self.model.embeddings_to_single_tag_probs(
                        tag_name, self.embeddings
                    )
                    matches_tags[tag_name] = probs
                    self.tag_probs_cache[tag_name] = probs
            # фильтруем эмбеддинги на основе отфильтрованных результатов
            matches_tags = matches_tags.iloc[matches.index]

            normalized_euclidean_distance = (
                euclidean_distance(matches_tags, target_probs)
                / np.sqrt(len(query.tags))
            )
            score = 1 - normalized_euclidean_distance

        matches["score"] = score
        return self._top_results(
            matches=matches, group_by_video=group_by_video
        )

    @staticmethod
    def _top_results(matches, group_by_video: bool):
        if group_by_video:
            matches = matches.loc[
                # FIXME: убедиться, что df отсортирован заранее
                matches.groupby(["source_phash"], observed=True, sort=False)
                ["score"]
                .idxmax()
            ]

        matches = matches.nlargest(200, "score", keep="all")[:200]

        archives_by_path = {}
        for x in set(matches.index_path):
            archives_by_path[x] = ZipFile(x)
        results = []
        for _, m in matches.iterrows():
            archive = archives_by_path[m.index_path]
            with archive.open("thumbnails/" + m.path) as f:
                img = Image.open(f)
                img.load()
            caption = (
                f"{m.source_path} {utils.format_timestamp(m.source_time)}"
                f" ({m.index_name})"
            )
            results.append((img, caption))
        for x in archives_by_path.values():
            x.close()
        return matches, results
