import csv
import json
import re
from pathlib import Path

import jsonlines
from tqdm import tqdm

reader = csv.DictReader(open("../kdwd/statements.csv"))

is_person_file = Path("../kdwd/is_person_list.json")
if not is_person_file.exists():
    is_person = set()
    for item in tqdm(reader, total=141206854):
        if item["edge_property_id"] == "31" and item["target_item_id"] == "5":
            is_person.add(item["source_item_id"])

    json.dump(list(is_person), is_person_file.open("w"))
else:
    is_person = set(json.load(is_person_file.open()))

reader = csv.DictReader(open("../kdwd/page.csv"))

qid_mapping_file = Path("../kdwd/qid_mapping.json")
if not qid_mapping_file.exists():
    qid_mapping = {}
    for item in tqdm(reader, total=5362175):
        qid_mapping[item["page_id"]] = item
    json.dump(qid_mapping, qid_mapping_file.open("w"))
else:
    qid_mapping = json.load(qid_mapping_file.open())
    qid_mapping = {int(key): value for key, value in qid_mapping.items()}

examples = jsonlines.open("../kdwd/parsed_dataset.jsonl", "w")
for item in tqdm(jsonlines.open("../kdwd/link_annotated_text.jsonl"), total=5343564):
    item_qid = qid_mapping[item["page_id"]]["item_id"]
    page_title = qid_mapping[item["page_id"]]["title"]
    surname = None
    if item_qid in is_person:
        split_page_title = page_title.split()
        surname = split_page_title[-1]
    for section in item["sections"]:
        text = section["text"]

        entities = []
        for offset, length, page_id in zip(section["link_offsets"], section["link_lengths"],
                                           section["target_page_ids"]):
            entities.append({
                "mention": text[offset: offset + length],
                "offset": offset,
                "length": length,
                "qid": f'Q{qid_mapping[page_id]["item_id"]}',
                "out_of_kg": False
            })

        # Get weak labels
        occurrences_title = {m.start(): (m.start(), m.end()) for m in re.finditer(re.escape(page_title.lower()), text.lower())}
        if surname is not None:
            occurrences_surname = [(m.start(), m.end()) for m in re.finditer(re.escape(surname.lower()), text.lower())]
            for span in occurrences_surname:
                if span[0] not in occurrences_title:
                    occurrences_title[span[0]] = span


        for span in occurrences_title.values():
            valid = True
            for entity in entities:
                true_span = (entity["offset"], entity["offset"] + entity["length"])
                if (span[0] < true_span[0] and span[1] <= true_span[0]) or (span[0] >= true_span[1]):
                    continue
                else:
                    valid = False
                    break
            if valid:
                entities.append({
                    "mention": text[span[0]: span[1]],
                    "offset": span[0],
                    "length": span[1] - span[0],
                    "qid": f"Q{item_qid}",
                    "out_of_kg": False
                })
        entities = sorted(entities, key=lambda x: x["offset"])
        if entities:
            examples.write({
                "text": text,
                "entities": entities
            })