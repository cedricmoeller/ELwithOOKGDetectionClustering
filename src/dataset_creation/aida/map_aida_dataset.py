import json
from argparse import ArgumentParser
from pathlib import Path
from csv import reader

def parse_aida(aida_dataset_path: Path):
    documents = []
    current_document = []
    for line in aida_dataset_path.open():
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
                                "length": end_candidate - start_candidate,
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
                if len(element) > 6:
                    page_id = element[5]
                    page_ids.add(element[5])
                if len(element) > 4:
                    candidate_identifier = element[4]
                    page_title = element[3]
                    page_titles.add(element[3])
                    candidate_out_of_kg = False
                else:
                    candidate_identifier = None
                    page_id = None
                    page_title = None
                    candidate_out_of_kg = True
            else:
                if start_candidate >= 0:
                    candidates.append(
                        {
                            "offset": start_candidate,
                            "length": end_candidate - start_candidate,
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
                    "length": end_candidate - start_candidate,
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
    return all_documents_train, all_documents_testa, all_documents_testb


def map_to_ookg(all_documents: list, ookg_mapping: dict):
    new_documents = []
    for document in all_documents:
        new_entities = []
        if document["docid"] in ookg_mapping:
            mapping = ookg_mapping[document["docid"]]
            for entity in document["entities"]:
                if str(entity["offset"]) in mapping:
                    content = mapping[str(entity["offset"])]
                    entity["qid"] = content["qid"]
                    entity["out_of_kg"] = content["out_of_kg"]
                    new_entities.append(entity)
        if new_entities:
            document["entities"] = new_entities
            new_documents.append(document)
    return new_documents


def map_aida(aida_dataset_path: Path, ookg_mapping_train_path: Path,
             ookg_mapping_testa_path: Path,
             ookg_mapping_testb_path: Path):
    all_documents_train, all_documents_testa, all_documents_testb = parse_aida(aida_dataset_path)

    json.dump(map_to_ookg(all_documents_train, json.load(ookg_mapping_train_path.open())),
              open("datasets/aida_train_ookg_art_2019.json",
                                                                            "w"))
    json.dump(map_to_ookg(all_documents_testa, json.load(ookg_mapping_testa_path.open())),
              open("datasets/aida_testa_ookg_art_2019.json",
                                                                            "w"))
    json.dump(map_to_ookg(all_documents_testb, json.load(ookg_mapping_testb_path.open())),
              open("datasets/aida_testb_ookg_art_2019.json",
                                                                            "w"))


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("filename", type=str, default="AIDA-YAGO2-dataset.tsv")
    args = argparser.parse_args()

    map_aida(args.filename,
             Path("datasets/aida/aida_train_ookg_mapping.json"),
             Path("datasets/aida/aida_testa_ookg_mapping.json"),
             Path("datasets/aida/aida_testb_ookg_mapping.json"))
