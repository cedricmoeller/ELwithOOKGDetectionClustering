import csv
import json
from itertools import chain
from pathlib import Path

from jsonlines import jsonlines
from tqdm import tqdm

from src.dataset_creation.wikievents.wikievents_creation import get_wikidata_ids

main_path = Path("data/datasets/conll-aida/le_titov")



def extract_candidates(dataset_path: Path):
    r = csv.reader(dataset_path.open(), delimiter="\t")
    mention_dict = {}
    ground_truth_candidates = set()
    no_candidate = 0
    counter = 0
    other_counter = 0

    current_doc_id = None
    current_candidates = []
    current_example = {"entities": current_candidates}
    examples = []
    for line in r:
        doc_id = line[0]
        if doc_id != current_doc_id and current_doc_id is not None:
            current_doc_id = doc_id
            examples.append(current_example)
            current_candidates = []
            current_example = {"entities": current_candidates}
        document_text = line[3] + " " + line[2] + " " + line[4]
        current_example["text"] = document_text
        current_example["docid"] = doc_id
        mention = line[2]
        gt_index = line.index("GT:")
        candidates = line[6:gt_index]
        ground_truth = line[gt_index + 1]
        ground_truth_candidate_idx = int(ground_truth[0:ground_truth.find(",")]) - 1
        new_candidates = []
        for x in candidates:
            if x == "EMPTYCAND":
                no_candidate += 1
                break
            comma_idx = x.find(",")
            page_id = x[:comma_idx]
            x = x[comma_idx + 1:]
            comma_idx = x.find(",")
            prior = x[:comma_idx]
            page_name = x[comma_idx + 1:]
            new_candidates.append((page_id, prior, page_name))
        if ground_truth_candidate_idx >= 0:
            ground_truth_candidate = new_candidates[ground_truth_candidate_idx]
            ground_truth_candidates.add(ground_truth_candidate)
        else:
            ground_truth_candidate = None
            other_counter += 1

        current_candidates.append({"mention": mention, "out_of_kg": ground_truth_candidate is None,
                                   "page_id": int(ground_truth_candidate[0]) if ground_truth_candidate is not None else None,
                                   "page_title": ground_truth_candidate[2] if ground_truth_candidate is not None else None,
                                   "docid": doc_id,
                                   "offset":len(line[3]) + 1,
                                   "length": len(mention)})
        counter += 1
        mention_dict[mention] = new_candidates
    examples.append(current_example)
    print(no_candidate)
    print(no_candidate/counter)
    print(other_counter)
    print(other_counter/counter)
    return mention_dict, ground_truth_candidates, examples


def map_examples(examples: list, page_id_mapping: dict, page_title_mapping: dict, available_qids:set) -> list:
    for example in examples:
        for entity in example["entities"]:
            if entity["page_id"] is not None:
                qid = page_id_mapping.get(entity["page_id"])
                if qid is None:
                    qid = page_title_mapping.get(entity["page_title"])
                    if qid not in available_qids:
                        qid = None
                entity["qid"] = qid
            else:
                entity["qid"] = None
            entity["out_of_kg"] = entity["qid"] is None

    return examples

def main():
    train, ground_truth_candidates_train, train_examples = extract_candidates(main_path.joinpath("aida_train.csv"))
    testa, ground_truth_candidates_testa, testa_examples = extract_candidates(main_path.joinpath("aida_testA.csv"))
    testb, ground_truth_candidates_testb, testb_examples = extract_candidates(main_path.joinpath("aida_testB.csv"))

    ground_truth_candidates = ground_truth_candidates_train
    ground_truth_candidates.update(ground_truth_candidates_testa)
    ground_truth_candidates.update(ground_truth_candidates_testb)
    mention_mapping = {}
    for key, value in train.items():
        if key not in mention_mapping:
            mention_mapping[key] = value

    for key, value in testa.items():
        if key not in mention_mapping:
            mention_mapping[key] = value

    for key, value in testb.items():
        if key not in mention_mapping:
            mention_mapping[key] = value

    for key, value in chain(train.items(), testa.items(), testb.items()):
        if value != mention_mapping[key]:
            print("NO")

    all_candidates = list(chain(*mention_mapping.values()))
    wikipedia_page_ids = {x[0] for x in all_candidates}
    wikipedia_page_titles = {x[2] for x in all_candidates}

    url_mapping, all_failed = get_wikidata_ids(wikipedia_page_titles, parallel=30, page_titles=True)


    page_id_mapping, all_failed = get_wikidata_ids(wikipedia_page_ids, parallel=30, page_titles=False)

    json.dump(page_id_mapping, open("alternative_aida_page_id_mapping_le_titov.json", "w"))
    json.dump(url_mapping, open("alternative_aida_url_mapping_le_titov.json", "w"))

    page_id_mapping = json.load(open("alternative_aida_page_id_mapping_le_titov.json"))
    url_mapping = json.load(open("alternative_aida_url_mapping_le_titov.json"))

    titov_qids = set()
    for value in page_id_mapping.values():
        titov_qids.add(value)
    for value in url_mapping.values():
        titov_qids.add(value)
    json.dump(list(titov_qids), open("titov_qids.json", "w"))
    available_qids = set()
    for x in tqdm(jsonlines.open("/data1/anonym/wikidata-old/parsed_dump_2019.jsonl")):
        available_qids.add(x["id"])

    train_examples = map_examples(train_examples, page_id_mapping, url_mapping, available_qids)
    testa_examples = map_examples(testa_examples, page_id_mapping, url_mapping, available_qids)
    testb_examples = map_examples(testb_examples, page_id_mapping, url_mapping, available_qids)

    ground_truth_candidates_not_found = set()
    for candidate in ground_truth_candidates:
        qid_1 = page_id_mapping.get(int(candidate[0]))
        qid_2 = url_mapping.get(candidate[2])
        if qid_1 is None and qid_2 is None:
            if qid_1 not in available_qids and qid_2 not in available_qids:
                ground_truth_candidates_not_found.add(candidate)
    print(len(ground_truth_candidates_not_found))

    counter = 0
    not_in_aida_kg = 0
    overall = 0
    new_set = {}

    for mention, candidates in mention_mapping.items():
        new_candidates = []
        for candidate in candidates:
            qid = page_id_mapping.get(int(candidate[0]), None)
            if qid is None:
                qid = url_mapping.get(candidate[2])
            if qid is None or qid not in available_qids:
                counter += 1
                if qid not in available_qids and qid is not None:
                    not_in_aida_kg += 1
            else:
                new_candidates.append(tuple(list(candidate) + [qid]))
            overall += 1
        if new_candidates:
            new_set[mention.lower()] = new_candidates

    print(not_in_aida_kg)
    print(counter)
    print(overall)
    print(counter / overall)

    aida_file_b = json.load(open("aida_datasets_in_use/aida_transformed_testb_filtered_ookg_art.json"))
    aida_file_a = json.load(open("aida_datasets_in_use/aida_transformed_testa_filtered_ookg_art.json"))
    aida_file_train = json.load(open("aida_datasets_in_use/aida_transformed_train_filtered_ookg_art.json"))
    all_qids = {entity["qid"] for example in aida_file_train for entity in example["entities"]}
    all_qids.update({entity["qid"] for example in aida_file_a for entity in example["entities"]})
    all_qids.update({entity["qid"] for example in aida_file_b for entity in example["entities"]})
    for mentions in new_set.values():
        for candidate in mentions:
            all_qids.add(candidate[-1])

    json.dump({mention: [{"qid": candidate[-1], "prior": float(candidate[1])} for candidate in candidates]
               for mention, candidates in new_set.items()}, open("/data1/anonym/aida_titov/le_titov_mention_mapping_2019.json", "w"), indent=4)

    json.dump(list(all_qids), open("/data1/anonym/aida_titov/le_titov_aida_all_qids_2019.json", "w"), indent=4)


main()