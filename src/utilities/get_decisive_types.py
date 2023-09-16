import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Union, List

import jsonlines
import tqdm


def get_entity_types(qid: str, entity_dict: dict, type_dict: dict = None) -> set:
    if type_dict is None:
        type_dict = {}
    types = set(entity_dict.get(qid, set()))
    additional_types = set()
    for x in types:
        additional_types.update(set(type_dict.get(x, set())))
    types.update(additional_types)
    return types

def disambiguate_using_type_set(examples, candidates_dict, num_candidates, with_doc, entity_dict, used_types: set, type_dict=None):
    if type_dict is None:
        type_dict = {}
    not_decisive = 0
    counter = 0
    for idx, example in enumerate(examples):
        for entity in example["entities"]:
            if not entity["out_of_kg"]:
                mention = entity["mention"]
                if with_doc:
                    candidates = set(candidates_dict.get((mention.lower(), entity["docid"]), [])[:num_candidates])
                else:
                    candidates = set(candidates_dict.get(mention.lower(), [])[:num_candidates])
                if entity["qid"] in candidates:
                    candidates.remove(entity["qid"])
                types_correct_entity = get_entity_types(entity["qid"], entity_dict, type_dict).intersection(used_types)
                if candidates:
                    for candidate in candidates:
                        types = get_entity_types(candidate, entity_dict, type_dict)
                        types = types.intersection(used_types)
                        types_correct_entity = types_correct_entity.difference(types)
                if not types_correct_entity:
                    not_decisive += 1
                counter += 1
    return (counter - not_decisive) / counter

def main(entity_dict: dict, examples: list, type_dict: dict, candidates_dict: dict, thresholds: list,
         with_doc:bool= False, num_candidates: int = 30, max_num: int = 1400):
    all_encountered = set()
    not_decisive = 0
    counter = 0
    number_of_types = 0
    decisive_candidates_mention_example = defaultdict(set)
    type_occurrence = defaultdict(int)
    for idx, example in tqdm.tqdm(enumerate(examples)):
        for entity in example["entities"]:
            if not entity["out_of_kg"]:
                mention = entity["mention"]
                if with_doc:
                    candidates = set(candidates_dict.get((mention.lower(), entity["docid"]), [])[:num_candidates])
                else:
                    candidates = set(candidates_dict.get(mention.lower(), [])[:num_candidates])
                if entity["qid"] in candidates:
                    candidates.remove(entity["qid"])
                types_correct_entity = get_entity_types(entity["qid"], entity_dict)
                all_encountered.update(types_correct_entity)
                if candidates:
                    for candidate in candidates:
                        types = get_entity_types(candidate, entity_dict)
                        all_encountered.update(types)
                        types_correct_entity = types_correct_entity.difference(types)
                    if not types_correct_entity:
                        not_decisive += 1
                    number_of_types += len(types_correct_entity)
                    for type_ in types_correct_entity:
                        type_occurrence[type_] += 1
                    decisive_candidates_mention_example[counter] = types_correct_entity
                    counter += 1

    score = disambiguate_using_type_set(examples, candidates_dict, num_candidates, with_doc, entity_dict, set(type_occurrence.keys()))
    already_handled = set()
    before = len(type_occurrence)
    removed = 0
    for type_ in tqdm.tqdm(copy.deepcopy(type_occurrence)):
        current_type = type_
        temp_type_set = copy.deepcopy(type_occurrence)
        del temp_type_set[current_type]
        score_when_replaced = disambiguate_using_type_set(examples, candidates_dict, num_candidates, with_doc, entity_dict, set(temp_type_set.keys()))
        if score_when_replaced >= score:
            type_occurrence = temp_type_set
            removed += 1
    replaced = 0
    for type_ in tqdm.tqdm(copy.deepcopy(type_occurrence)):
        already_handled.add(type_)
        current_type = type_
        super_types = set(type_dict.get(type_, set()))
        was_replaced = False
        for super_type in super_types:
            if super_type not in already_handled:
                already_handled.add(super_type)
                temp_type_set = copy.deepcopy(type_occurrence)
                occurrences = temp_type_set[current_type]
                del temp_type_set[current_type]
                temp_type_set[super_type] += occurrences
                score_when_replaced = disambiguate_using_type_set(examples, candidates_dict, num_candidates, with_doc, entity_dict, set(temp_type_set.keys()), type_dict)
                if score_when_replaced >= score:
                    current_type = super_type
                    type_occurrence = temp_type_set
                    was_replaced = True
        replaced += was_replaced

    print(removed)
    print(replaced)
    print(before)
    print(len(type_occurrence))
    decisive_candidates = set(type_occurrence.keys())

    print(number_of_types / counter)
    print(not_decisive / counter)
    print(len(all_encountered))
    print(len(decisive_candidates))

    print(len([x for x in decisive_candidates_mention_example.values() if not x])/ counter)
    decisive_candidates_list = [list(decisive_candidates)]



    for threshold in thresholds:
        tmp = {key for key in decisive_candidates if type_occurrence[key] >= threshold}

        for value in decisive_candidates_mention_example.values():
            for x in set(value):
                if x not in tmp and x in value:
                    value.remove(x)
        print(len([x for x in decisive_candidates_mention_example.values() if not x]) / counter)

        decisive_candidates_list.append(list(tmp))

    sorted_type_occurrence = list(sorted(type_occurrence.items(), key=lambda x: x[1], reverse=True))
    max_num_types = sorted_type_occurrence[:max_num]
    max_num_types = [x[0] for x in max_num_types]
    decisive_candidates_list.append(max_num_types)

    return decisive_candidates_list


def analyse_one_hop_two_hop_interaction(entity_dict: dict, examples: list, candidates_dict: dict, type_dict: dict,
         with_doc:bool= False, num_candidates: int = 10, window: int = 8, type_restriction_set=None, restrict_only_one_hop= False):

    decisive_candidates = defaultdict(int)
    all_encountered = set()
    not_decisive = 0
    not_decisive_by_two_hop_but_one_hop = 0
    not_decisive_one_hop = 0
    not_decisive_by_one_hop_but_two_hop = 0
    not_decisive_both = 0
    counter = 0
    number_of_types = 0
    decisive_candidates_mention_example = defaultdict(set)
    type_occurrence = defaultdict(int)
    for idx, example in tqdm.tqdm(enumerate(examples)):
        for entity_idx, entity in enumerate(example["entities"]):
            other_entities = example["entities"][max(0, entity_idx -  window): entity_idx]
            other_entities = [x for x in other_entities if x["qid"] != entity["qid"] and x["qid"] is not None]
            other_entities_two_hops = set()
            for x in other_entities:
                other_entities_two_hops.update(set(entity_dict[x["qid"]]["two_hop_types"].keys()))
                other_entities_two_hops.update(set(entity_dict[x["qid"]]["one_hop_types"].keys()))

            if type_restriction_set is not None and not restrict_only_one_hop:
                other_entities_two_hops = other_entities_two_hops.intersection(type_restriction_set)
            if not entity["out_of_kg"]:
                mention = entity["mention"]
                if with_doc:
                    candidates = set(candidates_dict.get((mention.lower(), entity["docid"]), [])[:num_candidates])
                else:
                    candidates = set(candidates_dict.get(mention.lower(), [])[:num_candidates])
                if entity["qid"] in candidates:
                    candidates.remove(entity["qid"])
                types_correct_entity_two_hop = set(entity_dict[entity["qid"]]["two_hop_types"].keys())
                types_correct_entity_one_hop = set(entity_dict[entity["qid"]]["one_hop_types"].keys())
                types_correct_entity_two_hop.update(types_correct_entity_one_hop)
                types_correct_entity_two_hop = types_correct_entity_two_hop.intersection(other_entities_two_hops)
                if type_restriction_set is not None:
                    types_correct_entity_one_hop = types_correct_entity_one_hop.intersection(type_restriction_set)
                all_encountered.update(types_correct_entity_two_hop)
                if candidates:
                    for candidate in candidates:
                        types = set(entity_dict[candidate]["two_hop_types"].keys())
                        types_one_hop = set(entity_dict[candidate]["one_hop_types"].keys())
                        types.update(types_one_hop)
                        types = types.intersection(other_entities_two_hops)
                        if type_restriction_set is not None:
                            types_one_hop = types_one_hop.intersection(
                                type_restriction_set)
                        types_correct_entity_two_hop = types_correct_entity_two_hop.difference(types)
                        all_encountered.update(types)
                        types_correct_entity_two_hop = types_correct_entity_two_hop.difference(types)
                        types_correct_entity_one_hop = types_correct_entity_one_hop.difference(types_one_hop)
                    if not types_correct_entity_two_hop:
                        not_decisive += 1
                    if (not types_correct_entity_two_hop) and types_correct_entity_one_hop:
                        not_decisive_by_two_hop_but_one_hop += 1
                    if not types_correct_entity_one_hop:
                        not_decisive_one_hop += 1
                    if (not types_correct_entity_one_hop) and types_correct_entity_two_hop:
                        not_decisive_by_one_hop_but_two_hop += 1
                    if not types_correct_entity_one_hop and not types_correct_entity_two_hop:
                        not_decisive_both += 1
                    for type_ in copy.deepcopy(types_correct_entity_two_hop):
                        super_types = set(type_dict.get(type_, set()))
                        if super_types.intersection(types_correct_entity_two_hop):
                            types_correct_entity_two_hop.remove(type_)
                    number_of_types += len(types_correct_entity_two_hop)
                    for type_ in types_correct_entity_two_hop:
                        type_occurrence[type_] += 1
                        decisive_candidates[type_] += 1
                    decisive_candidates_mention_example[counter] = types_correct_entity_two_hop
                    counter += 1
        # if idx > 10000:
        #     break

    decisive_candidates = set(decisive_candidates.keys())

    results = {
        "types_per_element": number_of_types / counter,
        "non_decisive_two_hop_type_mentions": not_decisive / counter,
        "non_decisive_by_two_hop_but_one_hop": not_decisive_by_two_hop_but_one_hop/not_decisive,
        "non_decisive_one_hop_type_mentions": not_decisive_one_hop / counter,
        "non_decisive_by_one_hop_but_two_hop": not_decisive_by_one_hop_but_two_hop / not_decisive_one_hop,
        "non_decisive_together": not_decisive_both / counter,
        "number_of_types": len(all_encountered),
        "number_of_decisive_types": len(decisive_candidates)
    }

    return results


def main_two_hop(entity_dict: dict, examples: list, candidates_dict: dict, type_dict: dict, thresholds: List[int],
         with_doc:bool= False, num_candidates: int = 10, max_num: int = 1400, window: int = 8, type_restriction_set=None):

    decisive_candidates = defaultdict(int)
    all_encountered = set()
    not_decisive = 0
    counter = 0
    number_of_types = 0
    decisive_candidates_mention_example = defaultdict(set)
    type_occurrence = defaultdict(int)
    for idx, example in tqdm.tqdm(enumerate(examples)):
        for entity_idx, entity in enumerate(example["entities"]):
            other_entities = example["entities"][max(0, entity_idx -  window): entity_idx]
            other_entities = [x for x in other_entities if x["qid"] != entity["qid"] and x["qid"] is not None]
            other_entities_two_hops = set()
            for x in other_entities:
                other_entities_two_hops.update(set(entity_dict[x["qid"]]["two_hop_types"].keys()))
                other_entities_two_hops.update(set(entity_dict[x["qid"]]["one_hop_types"].keys()))

            if type_restriction_set is not None:
                other_entities_two_hops = other_entities_two_hops.intersection(type_restriction_set)
            if not entity["out_of_kg"] and other_entities:
                mention = entity["mention"]
                if with_doc:
                    candidates = set(candidates_dict.get((mention.lower(), entity["docid"]), [])[:num_candidates])
                else:
                    candidates = set(candidates_dict.get(mention.lower(), [])[:num_candidates])
                if entity["qid"] in candidates:
                    candidates.remove(entity["qid"])
                types_correct_entity_two_hop = set(entity_dict[entity["qid"]]["two_hop_types"].keys())
                types_correct_entity_one_hop = set(entity_dict[entity["qid"]]["one_hop_types"].keys())
                types_correct_entity_two_hop.update(types_correct_entity_one_hop)
                types_correct_entity_two_hop = types_correct_entity_two_hop.intersection(other_entities_two_hops)
                all_encountered.update(types_correct_entity_two_hop)
                if candidates:
                    for candidate in candidates:
                        types = set(entity_dict[candidate]["two_hop_types"].keys())
                        types_one_hop = set(entity_dict[candidate]["one_hop_types"].keys())
                        types.update(types_one_hop)
                        types = types.intersection(other_entities_two_hops)

                        all_encountered.update(types)
                        types_correct_entity_two_hop = types_correct_entity_two_hop.difference(types)

                    if not types_correct_entity_two_hop:
                        not_decisive += 1
                    for type_ in copy.deepcopy(types_correct_entity_two_hop):
                        super_types = set(type_dict.get(type_, set()))
                        if super_types.intersection(types_correct_entity_two_hop):
                            types_correct_entity_two_hop.remove(type_)
                    number_of_types += len(types_correct_entity_two_hop)
                    for type_ in types_correct_entity_two_hop:
                        type_occurrence[type_] += 1
                        decisive_candidates[type_] += 1
                    decisive_candidates_mention_example[counter] = types_correct_entity_two_hop
                    counter += 1
        # if idx > 10000:
        #     break

    decisive_candidates = set(decisive_candidates.keys())

    print(number_of_types / counter)
    print(not_decisive / counter)
    print(len(all_encountered))
    print(len(decisive_candidates))

    print(len([x for x in decisive_candidates_mention_example.values() if not x])/ counter)
    decisive_candidates_list = [list(decisive_candidates)]

    for threshold in thresholds:
        tmp = {key for key in decisive_candidates if type_occurrence[key] >= threshold}

        for value in decisive_candidates_mention_example.values():
            for x in set(value):
                if x not in tmp and x in value:
                    value.remove(x)
        print(len([x for x in decisive_candidates_mention_example.values() if not x]) / counter)

        decisive_candidates_list.append(list(tmp))

    sorted_type_occurrence = list(sorted(type_occurrence.items(), key=lambda x: x[1], reverse=True))
    max_num_types = sorted_type_occurrence[:max_num]
    max_num_types = [x[0] for x in max_num_types]
    decisive_candidates_list.append(max_num_types)

    return decisive_candidates_list


def extract_decisive_types(suffix: str, training_data_file: Union[str, list], mention_mapping_file: str, kg_file: str,
                           thresholds: list, superclasses_file: str = "class_superclasses.json"):
    if isinstance(training_data_file, list):
        examples = training_data_file
    else:
        if Path(training_data_file).suffix == ".jsonl":
            examples = jsonlines.open(training_data_file)
        else:
            examples = json.load(open(training_data_file))

    if Path(f"entity_dict_{suffix}.json").exists():
        entity_dict = json.load(open(f"entity_dict_{suffix}.json"))
    else:
        entity_dict = {}
        for entity in tqdm.tqdm(jsonlines.open(kg_file)):
            types = set()
            if "claims" in entity:
                for claim in entity["claims"]:
                    if claim[0] in {"P31", "P106", "P641", "P17"}:
                        types.add(claim[1])
            entity_dict[entity["id"]] = list(types)
        json.dump(entity_dict, open(f"entity_dict_{suffix}.json", "w"))
    type_dict = json.load(open(superclasses_file))
    candidates_dict = json.load(open(mention_mapping_file))
    candidates_dict = {key: [x["qid"] if isinstance(x, dict) else x for x in value] for key, value in candidates_dict.items()}
    decisive_candidates_lists = main(entity_dict, examples, type_dict, candidates_dict, thresholds, with_doc=False)

    filenames = [f"decisive_candidates_{suffix}.json"]
    json.dump(decisive_candidates_lists[0], open(f"decisive_candidates_{suffix}.json", "w"), indent=4)

    for decisive_candidates in decisive_candidates_lists:
        filename = f"decisive_candidates_{len(decisive_candidates)}_{suffix}.json"
        filenames.append(filename)
        json.dump(decisive_candidates, open(filename, "w"), indent=4)
    return filenames


def get_decisive_one_hop_types(suffix: str, training_data_file: Union[str, list], mention_mapping_file: str, kg_file: str,
                               superclasses_file: str = "class_superclasses.json",
                               user_input: int=-1, thresholds: list = None):


    filenames = extract_decisive_types(suffix, training_data_file, mention_mapping_file, kg_file, thresholds, superclasses_file)

    for idx, filename in enumerate(filenames):
        print(f"{idx}: {filename}")

def analyse_decisive_two_hop_types(suffix: str, training_data_file: Union[str, list], mention_mapping_file: str, kg_file: str,
                               superclasses_file: str = "class_superclasses.json"):
    if isinstance(training_data_file, list):
        examples = training_data_file
    else:
        if Path(training_data_file).suffix == ".jsonl":
            examples = jsonlines.open(training_data_file)
        else:
            examples = json.load(open(training_data_file))
    entity_dict = {}
    for entity in tqdm.tqdm(jsonlines.open(kg_file)):
        entity_dict[entity["id"]] = {
            "one_hop_types": entity["one_hop_types"],
            "two_hop_types": entity["two_hop_types"]
        }

    candidates_dict = json.load(open(mention_mapping_file))
    candidates_dict = {key: [x["qid"] for x in value] for key, value in candidates_dict.items()}

    type_dict = json.load(open(superclasses_file))

    filenames = [f"decisive_candidates_two_hop_{suffix}.json"]

    type_restriction_set = set(json.load(open("decisive_candidates_178_aida_letitov_new.json")))
    # type_restriction_set = None
    results = analyse_one_hop_two_hop_interaction(entity_dict, examples, candidates_dict, type_dict, with_doc=False,
                                             type_restriction_set=type_restriction_set, restrict_only_one_hop=True)

    print(json.dumps(results, indent=4))


def get_decisive_two_hop_types(suffix: str, training_data_file: Union[str, list], mention_mapping_file: str, kg_file: str,
                               superclasses_file: str = "class_superclasses.json", thresholds = None):
    if isinstance(training_data_file, list):
        examples = training_data_file
    else:
        if Path(training_data_file).suffix == ".jsonl":
            examples = jsonlines.open(training_data_file)
        else:
            examples = json.load(open(training_data_file))
    entity_dict = {}
    for entity in tqdm.tqdm(jsonlines.open(kg_file)):
        entity_dict[entity["id"]] = {
            "one_hop_types": entity["one_hop_types"],
            "two_hop_types": entity["two_hop_types"]
        }

    candidates_dict = json.load(open(mention_mapping_file))
    candidates_dict = {key: [x["qid"] for x in value] for key, value in candidates_dict.items()}

    type_dict = json.load(open(superclasses_file))

    # type_restriction_set = None
    decisive_candidates_lists = main_two_hop(entity_dict, examples, candidates_dict, type_dict, thresholds,
                                             with_doc=False)

    json.dump(decisive_candidates_lists[0], open(f"decisive_candidates_two_hops_{suffix}.json", "w"), indent=4)


def zero_shot_wiki():
    get_decisive_one_hop_types("wiki", "../kdwd/parsed_dataset.jsonl",
                  "../kdwd/mention_dict_weakly_annotated.json",
                  "../kdwd/kdwd_kg.jsonl")

def aida_2019(suffix: str = "aida_letitov_2019"):
    get_decisive_one_hop_types(suffix, "aida_datasets_in_use/aida_transformed_train_filtered_ookg_art.json", "/data1/anonym/aida_titov/le_titov_mention_mapping_2019.json",
                           "/data1/anonym/wikidata-old/parsed_dump_2019.jsonl", user_input=5, thresholds=[30, 50, 100],
                               superclasses_file="/data1/anonym/wikidata-old/class_superclasses_2019.json")

def wikievents_2019(suffix: str = "wikievents_2019"):
    get_decisive_one_hop_types(suffix, "/data1/anonym/wikinews_extended/wikievents_2000-2022_train.json", "/data1/anonym/wikinews_extended/wikievents_2000-2022_mention_dictionary.json",
                           "/data1/anonym/wikidata-old/parsed_dump_2019.jsonl", user_input=5, thresholds=[30, 50, 100, 150, 200, 300, 400, 500, 1000],
                               superclasses_file="/data1/anonym/wikidata-old/class_superclasses_2019.json")

if __name__ == '__main__':
    aida_2019()
    wikievents_2019()






