import json
from argparse import ArgumentParser
from pathlib import Path

import jsonlines
from tqdm import tqdm


def filter_reduced_dump(filepath: Path, filter_set: Path, output_path: Path):

    filter_set = set(json.load(filter_set.open()))

    dump = jsonlines.open(filepath)

    type_and_label_only_filter_set = set()
    for item in tqdm(dump, total=81976036):
        if item["id"] in filter_set:
            for claim in item["claims"]:
                type_and_label_only_filter_set.add(claim[0])
                type_and_label_only_filter_set.add(claim[1])
        else:
            for claim in item["claims"]:
                if claim[0] in {"P31", "P106", "P641", "P17"}:
                    type_and_label_only_filter_set.add(claim[1])
                if claim[1] in filter_set:
                    type_and_label_only_filter_set.add(claim[0])
                    type_and_label_only_filter_set.add(item["id"])

    dump = jsonlines.open(filepath)

    new_dump = jsonlines.open(output_path, "w")
    for item in tqdm(dump, total=81976036):
        if item["id"] in filter_set:
            new_dump.write(item)
        elif item["id"] in type_and_label_only_filter_set:
            new_dump.write({
                "id": item["id"],
                "labels": item["labels"],
                "claims": [claim for claim in item["claims"] if claim[0] in {"P31", "P106", "P641", "P17"}],
                "descriptions": item["descriptions"],
                "aliases": item["aliases"]
            })


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("filepath", type=Path)
    arg_parser.add_argument("filter_set", type=Path)
    arg_parser.add_argument("output_path", type=Path)

    args = arg_parser.parse_args()

    filter_reduced_dump(args.filepath, args.filter_set, args.output_path)
