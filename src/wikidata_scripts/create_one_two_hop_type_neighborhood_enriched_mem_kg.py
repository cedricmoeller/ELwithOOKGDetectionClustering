import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import smart_open
import ujson
from jsonlines import jsonlines
from tqdm import tqdm


def get_types_from_statements(statements: list, type_extension: dict) -> Dict[str, int]:
    properties_to_consider = {"P31", "P106", "P641", "P17"}
    types_dict = defaultdict(int)
    for x in statements:
        object_id = x[1]
        if x[0] in properties_to_consider:
            extended_types = type_extension.get(object_id, []) + [object_id]
            for type_ in extended_types:
                types_dict[type_] += 1
    return types_dict


def insert_one_two_hop_type_neighborhoods(wikidata_dump_file: str, filter_list_files: List[str],
                                          type_extension_file: str, output_file: str, minimize=True,
                                          remove_claims=False):
    filter_list = set()
    for file in filter_list_files:
        data = json.load(open(file))
        if isinstance(data, dict):
            data = [x["qid"] if isinstance(x, dict) else x for value in data.values() for x in value]
        tmp_filter_list = set(data)
        filter_list.update(tmp_filter_list)
    if not filter_list:
        filter_list = None

    type_extension = json.load(open(type_extension_file))

    wikidata_dump = smart_open.open(wikidata_dump_file)

    dependent_nodes_file = Path(f"dependent_nodes{suffix}.json")

    if not dependent_nodes_file.exists():
        dependent_nodes = defaultdict(set)
        for idx, content in tqdm(enumerate(wikidata_dump)):
            content = ujson.loads(content)
            qid = content["id"]
            if not qid.startswith("Q"):
                continue
            if filter_list is None or qid in filter_list:
                for x in content["claims"]:
                    dependent_nodes[x[1]].add(qid)

        json.dump({x: list(y) for x, y in dependent_nodes.items()}, dependent_nodes_file.open("w"))
    else:
        dependent_nodes = json.load(dependent_nodes_file.open())
        dependent_nodes = {x: set(y) for x, y in dependent_nodes.items()}

    def tmp_function():
        return {"onehop": {},
         "twohop": {}}
    types_vectors = defaultdict(tmp_function)
    wikidata_dump = smart_open.open(wikidata_dump_file)
    for idx, content in tqdm(enumerate(wikidata_dump)):
        content = ujson.loads(content)
        qid = content["id"]
        if not qid.startswith("Q"):
            continue
        if filter_list is None or qid in filter_list or qid in dependent_nodes:
            types_dict = get_types_from_statements(content["claims"], type_extension)

            if filter_list is None or qid in filter_list:
                types_vectors[qid]["onehop"] = types_dict

            if qid in dependent_nodes:
                for dependent_qid in dependent_nodes[qid]:
                    for key, value in types_dict.items():
                        if key not in types_vectors[dependent_qid]["twohop"]:
                            types_vectors[dependent_qid]["twohop"][key] = value
                        else:
                            types_vectors[dependent_qid]["twohop"][key] += value


    name = Path(wikidata_dump_file).stem
    new_file = jsonlines.open(output_file, "w")
    wikidata_dump = smart_open.open(wikidata_dump_file)
    for idx, content in tqdm(enumerate(wikidata_dump)):
        content = ujson.loads(content)
        qid = content["id"]
        if qid.startswith("P"):
            new_file.write(content)
        elif not qid.startswith("Q"):
            continue
        if filter_list is None or qid in filter_list:
            content["one_hop_types"] = types_vectors[qid]["onehop"]
            content["two_hop_types"] = types_vectors[qid]["twohop"]
            content["num_claims"] = len(content["claims"])
        if not minimize or filter_list is None or qid in filter_list:
            if remove_claims or (qid in dependent_nodes and qid not in filter_list):
                content["claims"] = []
            if qid in dependent_nodes and qid not in filter_list:
                del content["descriptions"]
            if qid in filter_list or qid in dependent_nodes:
                new_file.write(content)


def minimized_version_aida_2019():
    insert_one_two_hop_type_neighborhoods("/data1/anonym/wikidata-old/parsed_dump_2019.jsonl",
                                          ["/data1/anonym/aida_titov/le_titov_aida_all_qids_2019.json"],
                                          "/data1/anonym/wikidata-old/class_superclasses_2019.json", output_file="data1/anonym/wikidata-old/parsed_dump_2019_onetwohop_filtered_aida_with_degrees_2019.jsonl")

def minimized_version_aida_2019_es():
    insert_one_two_hop_type_neighborhoods("/data1/anonym/wikidata-old/parsed_dump_2019.jsonl",
                                          ["/data1/anonym/aida_titov/aida_es_all_entities.json"],
                                          "/data1/anonym/wikidata-old/class_superclasses_2019.json", output_file="data1/anonym/wikidata-old/parsed_dump_2019_onetwohop_filtered_aida_with_degrees_2019_es.jsonl")


# Without one_hop neighborhood
def minimized_version_zero_shot():
    insert_one_two_hop_type_neighborhoods("../kdwd/kdwd_kg.jsonl",
                                          ["../kdwd/relevant_qids.json"], "class_superclasses.json", remove_claims=True, output_file="zero_shot_filtered.jsonl")

def minimized_version_wikinews_2019():
    insert_one_two_hop_type_neighborhoods("/data1/anonym/wikidata-old/parsed_dump_2019.jsonl",
                                          ["/data1/anonym/wikinews_extended/wikievents_2000-2022_revised_v2_all_entities.json"], "/data1/anonym/wikidata-old/class_superclasses_2019.json", minimize=True, output_file="/data1/anonym/wikidata-old/parsed_dump_2019/wikinews_2019.jsonl",remove_claims=True)

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("parsed_kg_dump", type=str)
    argparser.add_argument("output_file", type=str)
    argparser.add_argument("--qid_to_filter", type=str, nargs="*", default=[])
    argparser.add_argument("--superclasses_file", type=str, default="datasets/class_superclasses_2019.json")

    args = argparser.parse_args()
    insert_one_two_hop_type_neighborhoods(args.parsed_kg_dump, args.qid_to_filter, args.superclasses_file, args.output_file,  minimize=True)

    # minimized_version_wikinews_2019()
    # minimized_version_aida_2019()
    # minimized_version_aida_2019_es()

