import json
from pathlib import Path

import pynif


ookg_url = "http://elookg.org/OutOfKG"
inkg_url = "http://elookg.org/InKG"

def transform_dataset(filepath: Path, document_prefix=""):
    nif_path = filepath.parents[0].joinpath(f"{filepath.stem}.nif")

    examples = json.load(filepath.open())
    collection = pynif.NIFCollection()
    for idx, example in enumerate(examples):
        context = collection.add_context(f"{document_prefix}{idx}", example["text"], 0, len(example["text"]))
        for entity in example["entities"]:
            context.add_phrase(entity["offset"], entity["offset"] + entity["length"], taIdentRef=f"http://www.wikidata.org/entity/{entity['qid']}",
                               taMsClassRef=ookg_url if entity["out_of_kg"] else inkg_url)

    collection.dump(nif_path)


# transform_dataset(Path("datasets/aida/aida_train_ookg_art_2019.json"), "http://aida_conll.org/")
# transform_dataset(Path("datasets/aida/aida_testb_ookg_art_2019.json"), "http://aida_conll.org/")
# transform_dataset(Path("datasets/aida/aida_testa_ookg_art_2019.json"), "http://aida_conll.org/")

transform_dataset(Path("datasets/wikievents/wikievents_2000-2022_dev.json"), "http://wikievents.org/")
transform_dataset(Path("datasets/wikievents/wikievents_2000-2022_test.json"), "http://wikievents.org/")
transform_dataset(Path("datasets/wikievents/wikievents_2000-2022_train.json"), "http://wikievents.org/")