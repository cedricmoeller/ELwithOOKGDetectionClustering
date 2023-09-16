import json
from argparse import ArgumentParser
from pathlib import Path


def calculate_overlap(entities_a: set, entities_b: set):
    return len(entities_a.intersection(entities_b))/len(entities_a), len(entities_a.intersection(entities_b))/len(entities_b)

def calculate_overlaps(entities_a: list, entities_b: list):
    mention_qids_a = set()
    qids_a = set()
    qids_a_out_of_kg = set()
    qids_a_non_out_of_kg = set()
    for entity in entities_a:
        mention_qids_a.add((entity["qid"], entity["mention"]))
        if entity["qid"]:
            qids_a.add(entity["qid"])
            if entity["out_of_kg"]:
                qids_a_out_of_kg.add(entity["qid"])
            else:
                qids_a_non_out_of_kg.add(entity["qid"])

    mention_qids_b = set()
    qids_b = set()
    qids_b_out_of_kg = set()
    qids_b_non_out_of_kg = set()
    for entity in entities_b:
        mention_qids_b.add((entity["qid"], entity["mention"]))
        if entity["qid"]:
            qids_b.add(entity["qid"])
            if entity["out_of_kg"]:
                qids_b_out_of_kg.add(entity["qid"])
            else:
                qids_b_non_out_of_kg.add(entity["qid"])
    overlap_a, overlap_b = calculate_overlap(qids_a, qids_b)
    overlap_out_of_kg_a, overlap_out_of_kg_b = calculate_overlap(qids_a_out_of_kg, qids_b_out_of_kg)
    overlap_non_out_of_kg_a, overlap_non_out_of_kg_b = calculate_overlap(qids_a_non_out_of_kg, qids_b_non_out_of_kg)
    overlap_mentions_a, overlap_mentions_b = calculate_overlap(mention_qids_a, mention_qids_b)
    return (overlap_a, overlap_b,
            overlap_out_of_kg_a, overlap_out_of_kg_b,
            overlap_non_out_of_kg_a, overlap_non_out_of_kg_b,
            overlap_mentions_a, overlap_mentions_b)

def main(file: Path):
    data = json.load(file.open())
    train, dev, test = (data["train"], data["dev"], data["test"])

    entities_in_train = [entity for x in train for example in x for entity in example["entities"]]
    entities_in_dev = [entity for x in dev for example in x for entity in example["entities"]]
    entities_in_test = [entity for example in test for entity in example["entities"]]
    print(calculate_overlaps(entities_in_train, entities_in_test))
    print(calculate_overlaps(entities_in_train, entities_in_dev))
    print(calculate_overlaps(entities_in_dev, entities_in_test))

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("filepath", type=Path)
    main(argparser.parse_args().filepath)