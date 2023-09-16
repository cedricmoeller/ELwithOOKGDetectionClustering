import gzip
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import jsonlines
from tqdm import tqdm


def enrich_subclasses(superclasses: set, class_dict: dict, already_seen: dict = None):
    if already_seen is None:
        already_seen = set()
    other_superclasses = set()
    for superclass in set(superclasses):
        if superclass not in already_seen:
            already_seen.add(superclass)
            other_superclasses.update(class_dict[superclass])
            enrich_subclasses(class_dict[superclass], class_dict, already_seen)
    superclasses.update(other_superclasses)


def extract_from_wikidata_dump(filepath: Path):
    wikidata_dump = gzip.open(filepath, "rt")
    class_dict = defaultdict(set)
    for idx, line in tqdm(enumerate(wikidata_dump)):
        try:
            content = json.loads(str(line).rstrip(",\n"))
        except json.JSONDecodeError:
            continue

        qid = content["id"]
        subclass_of = set()
        instance_of = []
        for statements in content["claims"].values():
            for statement in statements:
                mainsnak = statement["mainsnak"]
                if (
                        mainsnak.get("snaktype") == "value"
                        and mainsnak.get("datatype") == "wikibase-item"
                ):
                    object_id = mainsnak["datavalue"]["value"]["id"]
                    if mainsnak["property"] == "P31":
                        instance_of.append(object_id)
                    if mainsnak["property"] == "P279":
                        subclass_of.add(object_id)
        for x in instance_of:
            class_dict[x].update(set())
        for x in subclass_of:
            class_dict[x].update(set())
        if subclass_of:
            class_dict[qid] = subclass_of
    return class_dict

def extract_from_parsed_wikidata_dump(filepath: Path):
    wikidata_dump = jsonlines.open(filepath)
    class_dict = defaultdict(set)
    for idx, content in tqdm(enumerate(wikidata_dump)):
        qid = content["id"]
        subclass_of = set()
        instance_of = []
        for claim in content["claims"]:
            property_id, object_id = claim
            if property_id == "P31":
                instance_of.append(object_id)
            if property_id == "P279":
                subclass_of.add(object_id)
        for x in instance_of:
            class_dict[x].update(set())
        for x in subclass_of:
            class_dict[x].update(set())
        if subclass_of:
            class_dict[qid] = subclass_of
    return class_dict

def extract_class_hierarchy(args):
    class_dict = extract_from_parsed_wikidata_dump(args.filepath)
    #json.dump({key: list(value) for key,value in class_dict.items()}, args.output_path.open("w"), indent=4)
    # class_dict = json.load(args.output_path.open())
    # class_dict = {key: set(value) for key,value in class_dict.items()}
    for key, value in tqdm(class_dict.items()):
        enrich_subclasses(value, class_dict)
    json.dump({key: list(value) for key, value in class_dict.items()}, args.output_path.open("w"), indent=4)



if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("filepath", type=Path)
    arg_parser.add_argument("output_path", type=Path)

    args = arg_parser.parse_args()
    extract_class_hierarchy(args)