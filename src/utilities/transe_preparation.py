import json
from pathlib import Path

import numpy
from tqdm import tqdm


def filter_embeddings(all_entities: Path, embedding_id_list_file: Path, embeddings_numpy_file: Path, output_file_embeddings: str, output_file_mapping: str):
    all_entities = set(json.load(all_entities.open()))

    embedding_id_list = json.load(embedding_id_list_file.open())
    indices = []
    qids = []
    for idx, item in tqdm(enumerate(embedding_id_list)):
        if item.startswith("<http://www.wikidata.org/entity/Q"):
            item = item[item.rfind("Q"):-1]
            if item in all_entities:
                indices.append(idx)
                qids.append(item)

    print("Load embeddings file")
    all_embeddings = numpy.load(str(embeddings_numpy_file))
    filtered_embeddings = all_embeddings[indices, :]
    numpy.save(output_file_embeddings, filtered_embeddings)
    json.dump(qids, open(output_file_mapping, "w"))



#filter_embeddings(Path("/data1/anonym/wikinews_extended/wikievents_2000-2022_all_entities.json"),
#                  Path("/data1/anonym/transe/wikidata_translation_v1_names.json"),
#                  Path("/data1/anonym/transe/wikidata_translation_v1_vectors.npy"),
#                  "/data1/anonym/transe/wikidata_translation_v1_vectors_filtered.npy",
#                  "/data1/anonym/transe/wikidata_translation_mapping.json")

# filter_embeddings(Path("/data1/anonym/wikinews_extended/wikievents_2000-2022_revised_v2_all_entities.json"),
#                  Path("/data1/anonym/transe/wikidata_translation_v1_names.json"),
#                  Path("/data1/anonym/transe/wikidata_translation_v1_vectors.npy"),
#                  "/data1/anonym/transe/wikidata_translation_v1_vectors_filtered_revised.npy",
#                  "/data1/anonym/transe/wikidata_translation_mapping_revised.json")


filter_embeddings(Path("/data1/anonym/aida_titov/aida_es_all_entities.json"),
                 Path("/data1/anonym/transe/wikidata_translation_v1_names.json"),
                 Path("/data1/anonym/transe/wikidata_translation_v1_vectors.npy"),
                 "/data1/anonym/transe/wikidata_translation_v1_vectors_filtered_aida_es.npy",
                 "/data1/anonym/transe/wikidata_translation_mapping_aida_es.json")

# filter_embeddings(Path("/data1/anonym/aida_titov/le_titov_aida_all_qids_2019.json"),
#                   Path("/data1/anonym/transe/wikidata_translation_v1_names.json"),
#                   Path("/data1/anonym/transe/wikidata_translation_v1_vectors.npy"),
#                   "/data1/anonym/transe/wikidata_translation_v1_vectors_filtered_aida.npy",
#                   "/data1/anonym/transe/wikidata_translation_mapping_aida.json")

