import argparse
import gzip
import json
from pathlib import Path
from typing import Optional

import jsonlines
from tqdm import tqdm


def reduce_content(content: dict, labels_only):
    for key in list(content.keys()):
        if key not in {"aliases", "labels", "descriptions", "claims", "id"}:
            del content[key]
    content["aliases"] = [x["value"] for x in content["aliases"]["en"]] if 'en' in content["aliases"] else []
    if "en" in content["labels"]:
        content["labels"] = content["labels"]["en"]["value"]
    else:
        new_label = ""
        for value in content["labels"].values():
            new_label = value["value"]
            break
        content["labels"] = new_label

    if labels_only:
        del content["descriptions"]
        del content["claims"]
    else:
        content["descriptions"] = content["descriptions"]["en"]["value"] if 'en' in content["descriptions"] else ""
        all_statements = []
        for statements in content["claims"].values():
            for statement in statements:
                mainsnak = statement["mainsnak"]
                if (
                        mainsnak.get("snaktype") == "value"
                        and mainsnak.get("datatype") == "wikibase-item"
                ):
                    property = mainsnak["property"]
                    object_id = mainsnak["datavalue"]["value"]["id"]
                    all_statements.append([property, object_id])
        content["claims"] = all_statements




def filter_wikidata_dump(wikidata_dump, filter_types: Optional[set], output_name):
    out = jsonlines.open(output_name, "w")
    batch_limit = 1000000
    batch = []
    for idx, line in tqdm(enumerate(wikidata_dump)):
        if len(batch) > batch_limit:
            out.write_all(batch)
            batch = []

        try:
            content = json.loads(str(line).rstrip(",\n"))
        except json.JSONDecodeError:
            continue

        qid = content["id"]
        # if 'en' not in content["labels"]:
        #     continue
        instance_of = []
        labels_only = False
        if not qid.startswith("P"):
            if not qid.startswith("Q"):
                continue
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

            labels_only = filter_types is not None and not any([x in filter_types for x in instance_of])

        reduce_content(content, labels_only)
        batch.append(
            content
        )
    if batch:
        out.write_all(batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument(
        "filename", type=str
    )
    parser.add_argument("output_name")
    parser.add_argument("--filter_path", default=None)

    args = parser.parse_args()

    file_path = Path(args.filename)

    wikidata_dump = gzip.open(file_path, "rt")

    if args.filter_path is not None:
        filter_types = set(json.load(open(args.filter_path)))
    else:
        filter_types = None
    filter_wikidata_dump(wikidata_dump, filter_types, args.output_name)