import csv
import json
from collections import defaultdict
from pathlib import Path

import jsonlines
from tqdm import tqdm

existing_qids = set()
for item in tqdm(jsonlines.open("../kdwd/kdwd_kg_onetwohop_zero_shot.jsonl")):
    existing_qids.add(item["id"])


qid_mapping_file = Path("../kdwd/qid_mapping.json")
if not qid_mapping_file.exists():
    reader = csv.DictReader(open("../kdwd/page.csv"))

    qid_mapping = {}
    for item in tqdm(reader, total=5362175):
        qid_mapping[item["page_id"]] = item
    json.dump(qid_mapping, qid_mapping_file.open("w"))
else:
    qid_mapping = json.load(qid_mapping_file.open())
    qid_mapping = {int(key): value for key, value in qid_mapping.items()}


mention_dict = defaultdict(dict)
for item in tqdm(jsonlines.open("../kdwd/link_annotated_text.jsonl"), total=5343564):
    for section in item["sections"]:
        for offset, length, page_id in zip(section["link_offsets"], section["link_lengths"],
                                           section["target_page_ids"]):
            item_qid = f'Q{qid_mapping[page_id]["item_id"]}'
            if item_qid in existing_qids:
                mention = section["text"][offset: offset + length].lower()
                if item_qid not in mention_dict[mention]:
                    mention_dict[mention][item_qid] = 0
                mention_dict[mention][item_qid] += 1

new_mention_dict = {}

for key, value in mention_dict.items():
    overall_sum = sum(value.values())
    candidate_list = []
    for key_ in value:
        candidate_list.append({
            "qid": key_,
            "prior": value[key_] / overall_sum
        })
    new_mention_dict[key] = candidate_list

json.dump(new_mention_dict, open("../kdwd/mention_dict.json", "w"), indent=4)



mention_dict = defaultdict(dict)
for example in tqdm(jsonlines.open("../kdwd/parsed_dataset.jsonl"), total=5343564):
    for entity in example["entities"]:
        item_qid = entity["qid"]
        if item_qid in existing_qids:
            mention = entity["mention"].lower()
            if item_qid not in mention_dict[mention]:
                mention_dict[mention][item_qid] = 0
            mention_dict[mention][item_qid] += 1

new_mention_dict = {}
for key, value in mention_dict.items():
    overall_sum = sum(value.values())
    candidate_list = []
    for key_ in value:
        candidate_list.append({
            "qid": key_,
            "prior": value[key_] / overall_sum
        })

    new_mention_dict[key] = candidate_list

json.dump(new_mention_dict, open("../kdwd/mention_dict_weakly_annotated.json", "w"), indent=4)
