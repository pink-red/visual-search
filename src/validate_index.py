from pathlib import Path
import sys

from search import load_index


def main():
    _, index_path = sys.argv

    df, embeddings, metadata = load_index(index_path)
    print(df.path.iloc[0])
    print(df.source_path.iloc[0])

    print("version" in metadata)
    print("thumbnails_ext" in metadata)
    print("model_name" in metadata)
    print("index_type" in metadata)
    print("name" in metadata)
    print("created_at" in metadata)

    settings = metadata.get("settings", {})
    print("interval_seconds" in settings)


if __name__ == "__main__":
    main()
