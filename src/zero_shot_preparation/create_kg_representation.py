import json
from collections import defaultdict
from csv import DictReader
from pathlib import Path

import jsonlines
import tqdm

labels = {}
descriptions = {}
aliases = defaultdict(set)

labels_path = Path("../kdwd/labels_descriptions.json")
aliases_path = Path("../kdwd/aliases.json")

if not labels_path.exists():
    labels_reader = DictReader(open("../kdwd/item.csv"))
    for item in tqdm.tqdm(labels_reader, total=51450317):
        labels[item["item_id"]] = item["en_label"]
        descriptions[item["item_id"]] = item["en_description"]
    json.dump({"labels": labels,
               "descriptions": descriptions}, labels_path.open("w"))
else:
    content = json.load(labels_path.open())
    labels = content["labels"]
    descriptions = content["descriptions"]

if not aliases_path.exists():
    aliases_reader = DictReader(open("../kdwd/item_aliases.csv"))
    for item in tqdm.tqdm(aliases_reader, total=6823025):
        aliases[item["item_id"]].add(item["en_alias"])
    aliases = {key: list(value) for key, value in aliases.items()}
    json.dump(aliases, aliases_path.open("w"))
else:
    aliases = json.load(aliases_path.open())

claims = defaultdict(set)

claims_path = Path("../kdwd/claims.json")

if not claims_path.exists():
    for item in tqdm.tqdm(DictReader(open("../kdwd/statements.csv")), total=141206854):
        # if item["edge_property_id"] in {"31", "106", "641", "17"}:
        claims[item["source_item_id"]].add((f'P{item["edge_property_id"]}', f'Q{item["target_item_id"]}'))
    claims = {key: list(value) for key, value in claims.items()}
    json.dump(claims, claims_path.open("w"))
else:
    claims = json.load(claims_path.open())


kg_file_path = Path("../kdwd/kdwd_kg.jsonl")
kg_file = jsonlines.open(kg_file_path, "w")

for item in tqdm.tqdm(labels.keys()):
    full_item = {
        "id": f"Q{item}",
        "labels": labels[item],
        "descriptions": descriptions.get(item, ""),
        "aliases": aliases.get(item, []),
        "claims": claims.get(item, [])
    }
    kg_file.write(full_item)
