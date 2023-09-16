import copy
import json
import random
import re
import urllib.parse
from collections import defaultdict
from csv import reader

import jsonlines
from SPARQLWrapper import SPARQLWrapper, JSON, POST
from elasticsearch import Elasticsearch
from tqdm import tqdm

from src.dataset_creation.wikievents.wikievents_creation import get_wikidata_ids


def map_aida():
    documents = []
    current_document = []
    for line in open("data/datasets/conll-aida/AIDA-YAGO2-dataset.tsv"):
        if "-DOCSTART-" in line:
            if current_document:
                documents.append(current_document)
            current_document = []
        current_document.append(line)

    documents.append(current_document)
    page_ids = set()
    page_titles = set()
    all_documents_train = []
    all_documents_testa = []
    all_documents_testb = []
    for document in documents:
        document_id = document[0]
        normalized_document_id = document_id[12:]
        normalized_document_id = normalized_document_id[0: normalized_document_id.find(" ")]
        if "testa" in normalized_document_id or "testb" in normalized_document_id:
            normalized_document_id = normalized_document_id[:-5]
        document_reader = reader(document[1:], delimiter='\t', quotechar='|')
        document_text = ""
        candidates = []
        current_candidate = []
        start_candidate = -1
        end_candidate = -1
        candidate_identifier = None
        candidate_out_of_kg = None
        mention = None
        page_id = None
        page_title = None
        for element in document_reader:
            if len(element) > 1:
                if element[1] == 'B':
                    if start_candidate >= 0:
                        candidates.append(
                            {
                                "offset": start_candidate,
                                "length": end_candidate-start_candidate,
                                "out_of_kg": candidate_out_of_kg,
                                "qid": candidate_identifier,
                                "page_id": page_id,
                                "page_title": page_title,
                                "docid": normalized_document_id,
                                "mention": mention
                            }
                        )
                    start_candidate = len(document_text)
                    end_candidate = len(document_text) + len(element[0])
                else:
                    end_candidate = len(document_text) + len(element[0])
                mention = element[2]
                if len(element) > 4:
                    candidate_identifier = element[4]
                    page_id = element[5]
                    page_ids.add(element[5])
                    page_title = element[3]
                    page_titles.add(element[3])
                    candidate_out_of_kg=False
                else:
                    candidate_identifier = None
                    page_id = None
                    page_title = None
                    candidate_out_of_kg=True
            else:
                if start_candidate >= 0:
                    candidates.append(
                        {
                            "offset": start_candidate,
                            "length": end_candidate-start_candidate,
                            "out_of_kg": candidate_out_of_kg,
                            "qid": candidate_identifier,
                            "page_id": page_id,
                            "page_title": page_title,
                            "mention": document_text[start_candidate: end_candidate],
                            "docid": normalized_document_id
                        }
                    )
                    start_candidate = -1
                    end_candidate = -1
            if element:
                document_text += element[0] + " "
        if start_candidate >= 0:
            candidates.append(
                {
                    "offset": start_candidate,
                    "length": end_candidate-start_candidate,
                    "out_of_kg": candidate_out_of_kg,
                    "qid": candidate_identifier,
                    "page_id": page_id,
                    "page_title": page_title,
                    "mention": document_text[start_candidate: end_candidate],
                    "docid": normalized_document_id
                }
            )
        if "testa" in document_id:
            all_documents_testa.append({
                "text": document_text,
                "docid": normalized_document_id,
                "entities": candidates
            })
        elif "testb" in document_id:
            all_documents_testb.append({
                "text": document_text,
                "docid": normalized_document_id,
                "entities": candidates
            })
        else:
            all_documents_train.append({
                "text": document_text,
                "docid": document_id,
                "entities": candidates
            })

    url_mapping, page_id_mapping = get_mapping_page_ids_page_titles(page_titles, page_ids)
    counter = 0
    all_counter = 0
    for document in all_documents_train:
        for idx in reversed(list(range(len(document["entities"])))):
            entity = document["entities"][idx]
            if entity['page_id']:
                mapped_qid = page_id_mapping.get(int(entity['page_id']))
                if mapped_qid is None:
                    mapped_qid = url_mapping.get(entity['page_title'])
                all_counter += 1
                if not mapped_qid:
                    counter += 1
                    del document["entities"][idx]
                    continue
                entity["qid"] =mapped_qid
    print(counter)
    print(counter/all_counter)

    counter = 0
    all_counter = 0
    for document in all_documents_testa:
        for idx in reversed(list(range(len(document["entities"])))):
            entity = document["entities"][idx]
            if entity['page_id']:
                mapped_qid = page_id_mapping.get(int(entity['page_id']))
                if mapped_qid is None:
                    mapped_qid = url_mapping.get(entity['page_title'])
                all_counter += 1
                if not mapped_qid:
                    counter += 1
                    del document["entities"][idx]
                    continue
                entity["qid"] = mapped_qid
    print(counter)
    print(counter / all_counter)

    counter = 0
    all_counter = 0
    for document in all_documents_testb:
        for idx in reversed(list(range(len(document["entities"])))):
            entity = document["entities"][idx]
            if entity['page_id']:
                mapped_qid = page_id_mapping.get(int(entity['page_id']))
                if mapped_qid is None:
                    mapped_qid = url_mapping.get(entity['page_title'])
                all_counter += 1
                if not mapped_qid:
                    counter += 1
                    del document["entities"][idx]
                    continue
                entity["qid"] = mapped_qid
    print(counter)
    print(counter / all_counter)

    json.dump(all_documents_train, open("aida_transformed_train.json", "w"), indent=4)
    json.dump(all_documents_testa, open("aida_transformed_testa.json", "w"), indent=4)
    json.dump(all_documents_testb, open("aida_transformed_testb.json", "w"), indent=4)


def hex2int(hexa: str) -> int:
    return int(hexa, 16)


def replace_unicode(u_str):
    matches = set(re.findall("\\\\u....", u_str))
    for match in matches:
        u_str = u_str.replace(match, chr(hex2int(match[2:])))
    return u_str

def replace_unicode_weird(u_str):
    matches = set(re.findall("\\\\\\\\\\\\\\\\u005cu....", u_str))
    for match in matches:
        u_str = u_str.replace(match, chr(hex2int(match[10:])))
    return u_str

def map_yago_names_wikidata(yago_names: list, batch_size= 1):
    sparql_wrapper = SPARQLWrapper("https://yago-knowledge.org/sparql/query")
    sparql_wrapper.setReturnFormat(JSON)
    sparql_wrapper.setMethod(POST)
    batch = []
    mapping = {}
    not_found = 0
    for name in tqdm(yago_names):
        if len(batch) >= batch_size:
            batch_mapping = run_sparql_query(batch, sparql_wrapper)
            not_found += batch_size - len(batch_mapping)
            mapping.update(batch_mapping)
            batch = []
        batch.append(name)
    if batch:
        batch_mapping = run_sparql_query(batch, sparql_wrapper)
        mapping.update(batch_mapping)
        not_found += len(batch) - len(batch_mapping)
    print(not_found/len(yago_names))
    return mapping

def run_sparql_query(yago_names: list, sparql_wrapper: SPARQLWrapper) -> dict:
    yago_names = [f"<http://yago-knowledge.org/resource/{x}>" for x in yago_names]
    concatenated = " ".join(yago_names)
    query = f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX yago: <http://yago-knowledge.org/resource/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT * WHERE {{
      values ?sub {{{concatenated}}}
       ?sub owl:sameAs ?obj .
      FILTER ( strstarts(str(?obj), "http://www.wikidata.org/entity/") )
    }} 
    """
    sparql_wrapper.setQuery(query)
    sparql_wrapper.uri
    result = sparql_wrapper.queryAndConvert()
    mapping = {}
    for r in result["results"]["bindings"]:
        yago_name = r["sub"]["value"]
        yago_name = yago_name[len("http://yago-knowledge.org/resource/"):]
        qid = r["obj"]["value"][len("http://www.wikidata.org/entity/"):]
        mapping[yago_name] = qid

    return mapping



def map_mention_dictionary():
    mention_candidate_mapping = defaultdict(set)
    for line in open("data/datasets/conll-aida/aida_means.tsv"):
        line = line[:-1]
        values = line.split("\t")
        mention = replace_unicode(values[0][1:-1])
        candidate = replace_unicode(values[1]) #.replace("_", " ")
        mention_candidate_mapping[mention].add(candidate)

    return mention_candidate_mapping

def extract_wikidata_mappings():
    mapping = {}
    for line in tqdm(open("data/datasets/conll-aida/yago-wd-sameAs.nt"), total=37091475):
        if not "http://www.wikidata.org/entity" in line:
            continue
        content = line.split("\t")
        yago_name, _, wikidata_qid = content[:3]
        yago_name = yago_name[len('<http://yago-knowledge.org/resource/'):-1]
        wikidata_qid = wikidata_qid[len('<http://www.wikidata.org/entity/'):-1]
        mapping[yago_name] = wikidata_qid
    return mapping


def get_allwikipedia_urls(filename="data/datasets/conll-aida/yago-2.ttl", total=82219056):
    file = open(filename)
    yago_wikipedia_url_mapping = {}
    for line in tqdm(file, total=total):
        if "hasWikipediaUrl" in line:
            split_line = line.split()
            if "<hasWikipediaUrl>" == split_line[1]:
                yago_tag = split_line[0]
                yago_tag = yago_tag[1:-1]
                wikipedia_url = split_line[2]
                wikipedia_url = wikipedia_url[1:-1]
                yago_wikipedia_url_mapping[yago_tag] = urllib.parse.unquote(wikipedia_url)
    return yago_wikipedia_url_mapping


def get_allwikipedia_urls_alt(filename="data/datasets/conll-aida/yago2_core_20101206.n3", total=32962709):
    file = open(filename)
    yago_wikipedia_url_mapping = {}
    for line in tqdm(file, total=total):
        if "hasWikipediaUrl" in line:
            split_line = line.split()
            if "y:hasWikipediaUrl" == split_line[1]:
                yago_tag = split_line[0]
                yago_tag = urllib.parse.unquote(yago_tag[1:-1])
                wikipedia_url = split_line[2]
                wikipedia_url = wikipedia_url[1:-1]
                yago_wikipedia_url_mapping[yago_tag] = urllib.parse.unquote(wikipedia_url)
    return yago_wikipedia_url_mapping

# mention_candidate_mapping = map_mention_dictionary()
# mention_candidate_mapping = {key: list(value) for key, value in mention_candidate_mapping.items()}
# json.dump(mention_candidate_mapping, open("mention_candidate_mapping_yago.json", "w"))


def clean_str(string: str):
    matches = set(re.findall("%\d\d\d\d", string))
    for match in matches:
        string = string.replace(match, chr(hex2int(match[1:])))
    matches = set(re.findall("%..", string))
    for match in matches:
        string = string.replace(match, chr(hex2int("00" + match[1:])))
    if "%" in string and False:
        key = string.replace("%", "\\u")
        try:
            key = key.encode("utf-8").decode("unicode-escape")
        except:
            key = urllib.parse.unquote(string)
    else:
        key = urllib.parse.unquote(string)
    return key

# map_aida()
# yago_wikipedia_url_mapping = get_allwikipedia_urls_alt()


es = Elasticsearch()


def check_if_qid_in_dataset(qid: str):
    res = es.search(index="old_dump", size=2, query={"term": {
      "uri": {
        "value": qid,
      }
    }})
    return len(res["hits"]["hits"])==1

# mention_candidate_mapping: dict = json.load(open("mention_candidate_mapping_yago.json"))
# yago_wikipedia_url_mapping = json.load(open("yago_wikipedia_url_mapping.json"))
#
# new_yago_wikipedia_url_mapping = {}
# for key, value in yago_wikipedia_url_mapping.items():
#     key = clean_str(key)
#     replaced_value = replace_unicode_weird(value)
#     new_yago_wikipedia_url_mapping[key] = replaced_value[replaced_value.rfind("/") + 1:]
#
# yago_wikipedia_url_mapping = new_yago_wikipedia_url_mapping
# new_mention_mapping = {}
#
# counter = 0
# overall = 0
# set_not_found = set()
# for mention, value in tqdm(mention_candidate_mapping.items()):
#     wikidata_qids = []
#     for identifier in value:
#         if identifier.startswith("wikicategory") or identifier.startswith("wordnet"):
#             continue
#         overall += 1
#         if identifier not in yago_wikipedia_url_mapping:
#             counter += 1
#             set_not_found.add(identifier)
#             continue
#         wikidata_qids.append(yago_wikipedia_url_mapping[identifier])
#     new_mention_mapping[mention] = wikidata_qids
#
# print(len(set_not_found))
# print(counter)
# print(overall)
# print(counter/overall)
# wikipedia_urls = {x for value in new_mention_mapping.values() for x in value}

# wikipedia_urls = json.load(open("all_failed.json"))
# url_mapping, all_failed = get_wikidata_ids(wikipedia_urls, parallel=50)
#
# json.dump(all_failed, open("all_failed.json", "w"))

# url_mapping = json.load(open("wikipedia_wikidata_mapping_final.json"))


def get_mentions(dataset_filename: str):
    dataset = json.load(open(dataset_filename))
    mentions = set()
    for example in dataset:
        for entity in example["entities"]:
            if not entity["out_of_kg"]:
                mentions.add(entity["mention"])
    return mentions


mentions_in_datasets = set()
mentions_in_datasets.update(get_mentions("aida_transformed_train.json"))
mentions_in_datasets.update(get_mentions("aida_transformed_testa.json"))
mentions_in_datasets.update(get_mentions("aida_transformed_testb.json"))

# final_candidate_mapping = {}
# counter = 0
# overall = 0
# for key, value in tqdm(new_mention_mapping.items()):
#     if key not in mentions_in_datasets:
#         continue
#     qids = set()
#     for x in value:
#         if x in url_mapping and url_mapping[x] is not None:
#             if check_if_qid_in_dataset(url_mapping[x]):
#                 counter += 1
#                 qids.add(url_mapping[x])
#         overall += 1
#     final_candidate_mapping[key] = qids
# print(counter)
# print(counter/overall)


def load_mention_dict(filter_list=None):
    file = open("data/datasets/conll-aida/prob_yago_crosswikis_wikipedia_p_e_m.txt")
    mention_dict = {}
    for line in tqdm(file, total=21587744):
        split_line = line.split("\t")[:-1]
        mention = split_line[0]
        if filter_list is None or mention in filter_list:
            candidates = split_line[2:]
            # candidates = [x.split(",")[0] for x in candidates]
            candidates = [(*(x.split(",")[:2]), ",".join(x.split(",")[2:])) for x in candidates]
            mention_dict[mention] = candidates
    return mention_dict

def get_mapping_page_ids_page_titles(page_titles:set, page_ids: set):
    url_mapping, all_failed = get_wikidata_ids(page_titles, parallel=30, page_titles=True)
    new_mapping, all_failed = get_wikidata_ids(page_titles, parallel=30, page_titles=True)
    for key, value in new_mapping.items():
        if url_mapping[key] is None:
            url_mapping[key] = value
    new_mapping, all_failed = get_wikidata_ids(page_titles, parallel=30, page_titles=True)
    for key, value in new_mapping.items():
        if url_mapping[key] is None:
            url_mapping[key] = value
    new_mapping, all_failed = get_wikidata_ids(page_titles, parallel=30, page_titles=True)
    for key, value in new_mapping.items():
        if url_mapping[key] is None:
            url_mapping[key] = value

    page_id_mapping, all_failed = get_wikidata_ids(page_ids, parallel=30, page_titles=False)

    return url_mapping, page_id_mapping

def extract_precomputed_mentions(available_qids):
    mention_dict = load_mention_dict(mentions_in_datasets)

    page_titles = set()
    for mention, candidates in mention_dict.items():
        if mention in mentions_in_datasets:
            page_titles.update(set([x[2] for x in candidates]))

    page_ids = set()
    for mention, candidates in mention_dict.items():
        if mention in mentions_in_datasets:
            page_ids.update(set([x[0] for x in candidates]))

    url_mapping, page_id_mapping = get_mapping_page_ids_page_titles(page_titles, page_ids)
    json.dump(url_mapping, open("new_candidate_wikidata_mapping.json", "w"))
    json.dump(page_id_mapping, open("new_candidate_wikidata_mapping2.json", "w"))

    mapping = json.load(open("new_candidate_wikidata_mapping.json"))
    mapping_2 = json.load(open("new_candidate_wikidata_mapping2.json"))

    new_mention_dict = {}
    not_found = set()
    overall = 0
    not_in_dataset = 0
    for mention, candidates in mention_dict.items():
        if mention in mentions_in_datasets:
            new_candidates = []
            for x in candidates:
                if x[2] in mapping and mapping[x[2]] is not None:
                    qid = mapping[x[2]]
                elif int(x[0]) in mapping_2 and mapping_2[int(x[0])] is not None:
                    qid = mapping_2[int(x[0])]
                else:
                    not_found.add(x)
                    qid = None
                if qid is not None:
                    if qid not in available_qids:
                        not_in_dataset += 1
                        qid = None
                new_candidates.append({"qid": qid, "prior": float(x[1])})
                overall += 1
            new_mention_dict[mention] = new_candidates
    print(len(not_found))
    print(len(not_found) / overall)
    print(not_in_dataset)
    print(not_in_dataset / overall)

    difference = mentions_in_datasets.difference(set(mention_dict.keys()))
    mentions_not_found = len(difference)
    print(mentions_not_found / len(mentions_in_datasets))
    json.dump({key: list(value) for key, value in new_mention_dict.items()}, open(f"mention_mapping_aida.json", "w"),
              indent=4)


def remove_all_out_of_kg_mentions(in_file: str, out_file:str):
    aida_transformed = json.load(open(in_file))
    dataset = []
    for example in aida_transformed:
        new_entities = []
        for entity in example["entities"]:
            if entity["out_of_kg"]:
                continue
            new_entities.append(entity)
        if not new_entities:
            continue
        example["entities"] = new_entities
        dataset.append(example)
    json.dump(dataset, open(out_file, "w"))



def remove_all_non_existing_mentions(in_file: str, out_file:str, available_qids):
    aida_transformed = json.load(open(in_file))
    dataset = []
    for example in aida_transformed:
        new_entities = []
        for entity in example["entities"]:
            if entity["qid"] not in available_qids and not entity["out_of_kg"]:
                continue
            new_entities.append(entity)
        if not new_entities:
            continue
        example["entities"] = new_entities
        dataset.append(example)
    json.dump(dataset, open(out_file, "w"))


def get_filtered_datasets(available_qids):
    remove_all_non_existing_mentions("aida_transformed_train.json", f"aida_transformed_train_filtered.json",
                                     available_qids)
    remove_all_non_existing_mentions("aida_transformed_testa.json", f"aida_transformed_testa_filtered.json",
                                     available_qids)
    remove_all_non_existing_mentions("aida_transformed_testb.json", f"aida_transformed_testb_filtered.json",
                                     available_qids)

    remove_all_out_of_kg_mentions(f"aida_transformed_train_filtered.json", f"aida_transformed_train_no_out_of_kg.json")
    remove_all_out_of_kg_mentions(f"aida_transformed_testa_filtered.json", f"aida_transformed_testa_no_out_of_kg.json")
    remove_all_out_of_kg_mentions(f"aida_transformed_testb_filtered.json", f"aida_transformed_testb_no_out_of_kg.json")


available_qids = set()
for x in jsonlines.open("filtered_all_entities_aida.jsonl"):
    available_qids.add(x["id"])


map_aida()

url_mapping = json.load(open("new_candidate_wikidata_mapping.json"))
page_id_mapping = json.load(open("new_candidate_wikidata_mapping2.json"))
extract_precomputed_mentions(available_qids)



mention_mapping_aida = json.load(open("mention_mapping_aida_alt.json"))
copied_mention_mapping_aida = {}
copied_mention_mapping_aida_upfill = {}
mention_dictionary_aida_es = json.load(open("mention_dictionary_aida_es.json"))

for mention, candidates in mention_dictionary_aida_es.items():
    if mention not in mention_mapping_aida:
        mention_mapping_aida[mention] = [{"qid": x, "prior": 0.0} for x in candidates]
    else:
        other_candidates = mention_mapping_aida[mention]
        num_none = sum(x["qid"] is None for x in other_candidates)
        other_candidates = [x for x in other_candidates if x["qid"] is not None]
        copied_mention_mapping_aida[mention] = copy.deepcopy(other_candidates)
        difference = list(set(candidates).difference({x["qid"] for x in other_candidates}))
        random.shuffle(difference)
        copied_mention_mapping_aida_upfill[mention] = copy.deepcopy(other_candidates) + [{"qid": x, "prior": 0.0} for x in difference]
        other_candidates += [{"qid": x, "prior": 0.0} for x in difference[:num_none]]
        mention_mapping_aida[mention] = other_candidates

json.dump(copied_mention_mapping_aida, open("mention_mapping_aida_no_none_alt.json", "w"))

get_filtered_datasets(available_qids)










