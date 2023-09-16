import itertools
import json
from collections import defaultdict

from elasticsearch import Elasticsearch
from tqdm import tqdm

index_name = "wikinews_2019_extended" # old_dump
es = None


def check_if_qid_in_dataset(qid: str):
    res = es.search(index=index_name, size=2, query={"term": {
      "uri": {
        "value": qid,
      }
    }})
    return len(res["hits"]["hits"])==1


def get_item(qid: str):
    res = es.search(index=index_name, size=2, query={"term": {
      "uri": {
        "value": qid,
      }
    }})
    if not len(res["hits"]["hits"]):
        return None
    return res["hits"]["hits"][0]


def create_alt_query(mention: str, num: int, boost: int = 2):
    head = {'index': index_name}
    nested_querry = {"nested": {
      "path": "labels",
      "query": {
            "match": { "labels.value": {
                            "query": mention,
                            "fuzziness": "AUTO",
                            "minimum_should_match": "50%",}
                          }
        },
      "score_mode": "max",
    }}
    function_score_query = {"function_score": {
                            "field_value_factor": {
                                "field": "num_claims",
                                "factor": 1.0,
                                "modifier": "ln1p",
                            },
        "boost": boost
                        }}
    body = {"query": {"bool": {"must": [nested_querry, function_score_query]}},
    'size': num}
    return head, body


def alt_query(mention: str):
    return es.search(index=index_name, size=150, query={"nested": {
      "path": "labels",
      "query": {
            "bool":
                {
                    "must": [
                        {"multi_match": {
                            "query": mention,
                            "fuzziness": "AUTO",
                            "fields": [
                              "labels.value"
                            ],
                            "minimum_should_match": "100%",
                            "type": "most_fields"
                          }},

                        # {"match": {"labels.value": mention}}
                    ]
                },
        },
      "score_mode": "max",
    }})


def get_candidates(mention: str):
    res = alt_query(mention)
    candidates = []
    # print(len(res["hits"]["hits"]))

    for hit in res['hits']['hits']:
        if hit["_source"]["uri"].startswith("P") or len(candidates)>= 100:
            continue
        candidates.append((hit["_source"]["uri"], hit["_score"], len(hit["_source"]["claims"])))
    return candidates


def filter_dataset(dataset_file, output_file):
    dataset = json.load(open(dataset_file))
    for example in tqdm(dataset):
        for idx in reversed(list(range(len(example["entities"])))):
            entity = example["entities"][idx]
            if not entity["out_of_kg"] and not check_if_qid_in_dataset(entity["qid"]):
                del example["entities"][idx]

    json.dump(dataset, open(output_file, "w"))


def msearch(queries, valid_entity_types = None, dedicated_es = None):
    # as you can see, you just need to feed the <body> parameter,
    # and don't need to specify the <index> and <doc_type> as usual
    if dedicated_es is not None:
        resp = dedicated_es.msearch(body = queries)
    else:
        resp = es.msearch(body = queries)

    candidates_list = []
    for sub_resp in resp["responses"]:
        candidates = []
        # print(len(res["hits"]["hits"]))

        for hit in sub_resp['hits']['hits']:
            if hit["_source"]["uri"].startswith("P") or len(candidates) >= 100:
                continue
            claims = hit["_source"].get("claims", [])
            if valid_entity_types is not None:
                valid = False
                for claim in claims:
                    if claim["predicate"] == "P31":
                        if claim["object"] in valid_entity_types:
                            valid = True
                            break

                if not valid:
                    continue

            candidates.append((hit["_source"]["uri"], hit["_score"], hit["_source"]["num_claims"], hit["_source"]["labels"]))
        candidates_list.append(candidates)
    return candidates_list


def get_candidates_alt(batch, valid_entity_types=None, num: int=150, boost: int = 1, dedicated_es=None):
    queries = []
    for item in batch:
        head, body = create_alt_query(item, num, boost=boost)
        queries.append(head)
        queries.append(body)
    results = msearch(queries, valid_entity_types=valid_entity_types, dedicated_es=dedicated_es)
    return results


def create_candidates_alt(dataset, prefix="", valid_entity_types=None, filter_set=None):
    if filter_set is None:
        filter_set = set()
    hits = 0
    counter = 0
    counter_inkg = 0
    correct = 0
    no_candidates = 0
    out_of_kg = 0
    counter_not_found = 0
    batch = []
    all_ranks = []
    no_candidate_not_included = []
    all_entities = set()
    all_mentions = set()
    mention_dictionary = {}

    for example in tqdm(dataset):
        for entity in example["entities"]:
            all_mentions.add(entity["mention"])

    for mention in tqdm(all_mentions):
        batch.append(mention)
        if len(batch) > 50:
            candidates_list = get_candidates_alt(batch, valid_entity_types=valid_entity_types)
            for mention_, candidates in zip(batch, candidates_list):
                mention_dictionary[mention_] = candidates
            batch = []
    if batch:
        candidates_list = get_candidates_alt(batch, valid_entity_types=valid_entity_types)
        for mention_, candidates in zip(batch, candidates_list):
            mention_dictionary[mention_] = candidates

    final_mention_dictionary = {}
    for example in tqdm(dataset):
        for entity in example["entities"]:
            if not entity["out_of_kg"]:
                all_entities.add(entity["qid"])
            candidates = mention_dictionary.get(entity["mention"], [])
            candidates = [x for x in candidates if x[0] not in filter_set]
            try:
                rank = [item[0] for item in candidates].index(entity["qid"])
            except ValueError:
                rank = -1
            if not candidates:
                no_candidates += 1
            elif rank == 0 and not entity["out_of_kg"]:
                correct += 1
            all_ranks.append(rank)
            if entity["out_of_kg"]:
                out_of_kg += 1
            else:
                counter_inkg += 1
            if rank >= 0 or entity["out_of_kg"]:
                hits += 1
            else:
                item = get_item(entity["qid"])
                if item is None:
                    counter_not_found += 1
                else:
                    no_candidate_not_included.append((entity, item))
                # candidates.add((entity["qid"], 0.0))
            final_mention_dictionary[entity["mention"]] = [item[0] for item in candidates]
            counter += 1

    all_ranks = [x + 1 for x in all_ranks if x>=0]

    print(no_candidates)
    all_ranks = sorted(all_ranks)
    count_ranks = defaultdict(int)
    for rank in all_ranks:
        for i in range(rank, 101):
            count_ranks[i] += 1
    count_ranks = {key: value/(counter - out_of_kg) for key, value in count_ranks.items()}
    json.dump(count_ranks, open(f"{prefix}_count_ranks.json", "w"), indent=4)

    print(sum(all_ranks)/len(all_ranks))
    print(sum([1/x for x in all_ranks])/len(all_ranks))
    print(f"Not found {counter_not_found}")
    print(f"Not found perc {counter_not_found/counter}")
    json.dump(no_candidate_not_included, open(f"{prefix}_no_candidate_not_included.json", "w"), indent=4)
    for key, value in final_mention_dictionary.items():
        all_entities.update(value)
    json.dump(list(all_entities),
              open(f"{prefix}_all_entities.json", "w"))
    json.dump({key: [{"qid": x, "prior": 1.0} for x in value] for key, value in final_mention_dictionary.items()}, open(f"{prefix}_mention_dictionary.json", "w"))

    precision = correct / (counter - no_candidates)
    recall = correct / (counter - no_candidates - out_of_kg)
    print(f"precision {precision}")  # Precision
    print(f"recall {recall}")  # Recall
    print(f"fmeasure {2 * precision * recall / (precision + recall)}")

    print(f"Candidate recall {hits / counter}")
    print(f"Candidate recall {(hits - out_of_kg) / counter_inkg}")
    return final_mention_dictionary


def create_candidates(dataset, suffix="", filter_set = None):
    if filter_set is None:
        filter_set = set()
    hits = 0
    counter = 0
    correct = 0
    no_candidates = 0
    out_of_kg = 0
    mention_dictionary = defaultdict(set)
    counter_not_found = 0
    for example in tqdm(dataset):
        for entity in example["entities"]:
            candidates = get_candidates(mention=entity["mention"])
            if not candidates:
                no_candidates += 1
            elif entity["qid"] == candidates[0][0] and not entity["out_of_kg"]:
                correct += 1
            if entity["out_of_kg"]:
                out_of_kg += 1
            candidates = [x for x in candidates if x[0] not in filter_set]
            candidates = set(candidates)
            if entity["qid"] in [item[0] for item in candidates] or entity["out_of_kg"]:
                hits += 1
            else:
                if not check_if_qid_in_dataset(entity["qid"]):
                    counter_not_found += 1
                # candidates.add((entity["qid"], 0.0))
            mention_dictionary[entity["mention"]].update({item[0] for item in candidates})
            counter += 1
    print(counter_not_found)
    print(counter_not_found/counter)

    json.dump({key: [{"qid": x, "prior": 0.0} for x in value] for key, value in mention_dictionary.items()}, open(f"mention_dictionary{suffix}.json", "w"))

    precision = correct / (counter - no_candidates)
    recall = correct / (counter - no_candidates - out_of_kg)
    print(precision)  # Precision
    print(recall)  # Recall
    print(2 * precision * recall / (precision + recall))

    print(hits / counter)
    return mention_dictionary


def create_wiki_events_candidate_set():
    dataset = json.load(open("data/execution_relevant_data/events_examples_filtered.json"))
    dataset = list(itertools.chain(*[x["examples"] for x in dataset]))
    create_candidates_alt(dataset, "_wikinews_2021")


def create_wiki_events_candidate_set_2019():
    dataset = json.load(open("wikievents_2000-2022_train.json"))
    dataset += json.load(open("wikievents_2000-2022_dev.json"))
    dataset += json.load(open("wikievents_2000-2022_test.json"))
    create_candidates_alt(dataset, "wikievents_2000-2022_revised_v2")

def create_aida_candidate_set():
    dataset = json.load(open("aida_testa_ookg_art_2019.json"))
    dataset += json.load(open("aida_testb_ookg_art_2019.json"))
    dataset += json.load(open("aida_train_ookg_art_2019.json"))
    filter_set = set()
    for example in dataset:
        for entity in example["entities"]:
            if entity["out_of_kg"]:
                filter_set.add(entity["qid"])
    create_candidates_alt(dataset, "aida_es", filter_set=filter_set)

if __name__ == '__main__':
    create_aida_candidate_set()
