import jsonlines
from tqdm import tqdm

existing_qids = set()

for item in tqdm(jsonlines.open("../kdwd/kdwd_kg_onetwohop_zero_shot.jsonl")):
    existing_qids.add(item["id"])


out = jsonlines.open("../kdwd/parsed_dataset_with_entities_removed_not_being_in_kg.jsonl", "w")

for example in tqdm(jsonlines.open("../kdwd/parsed_dataset.jsonl")):
    new_entities = []
    for entity in example["entities"]:
        if entity["qid"] in existing_qids:
            if entity["length"] > 0 and len(entity["mention"]) > 0:
                new_entities.append(entity)

    example["entities"] = new_entities
    if new_entities:
        out.write(example)