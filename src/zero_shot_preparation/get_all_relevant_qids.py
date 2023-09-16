import json

import jsonlines

relevant_qids = set()

mention_dict = json.load(open("../kdwd/mention_dict_weakly_annotated.json"))

for value in mention_dict.values():
    for cand in value:
        relevant_qids.add(cand["qid"])

for example in jsonlines.open("../kdwd/parsed_dataset.jsonl"):
    for entity in example["entities"]:
        relevant_qids.add(entity["qid"])





json.dump(list(relevant_qids), open("../kdwd/relevant_qids.json", "w"))