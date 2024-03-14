import json
from time import sleep

import requests


aliases = []
page = "1"
while True:
    sleep(1)
    r = requests.get(
        "https://danbooru.donmai.us/tag_aliases.json",
        params={
            "search[order]": "id",
            "search[status]": "Active",
            "limit": "1000",
            "page": page,
        }
    )
    r.raise_for_status()
    aliases_chunk = r.json()

    if not aliases_chunk:
        break

    aliases += aliases_chunk

    last_id = aliases_chunk[-1]["id"]
    page = f"b{last_id}"


print(json.dumps(aliases))
