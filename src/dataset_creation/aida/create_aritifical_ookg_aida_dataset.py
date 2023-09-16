import json
import random


def remove_out_of_kg_entities(dataset: list, artificial_ookg_entities: set, remove_original_out_of_kg_entities: bool = True):
    num_mentions_original = 0
    num_mentions_new = 0
    for example in dataset:
        new_mentions = []
        for mention in example["entities"]:
            num_mentions_original += 1
            if not mention["out_of_kg"] and mention["qid"] in artificial_ookg_entities:
                mention["out_of_kg"] = True

                # new_mentions.append(mention)
            elif not mention["out_of_kg"]:
                new_mentions.append(mention)
                num_mentions_new += 1
            elif not remove_original_out_of_kg_entities:
                new_mentions.append(mention)
                num_mentions_new += 1
        example["entities"] = new_mentions
    print(num_mentions_original)
    print(num_mentions_new)
    return dataset

def add_out_of_KG_entities(dataset: list, artificial_ookg_entities: set, remove_original_out_of_kg_entities: bool = True):
    for example in dataset:
        new_mentions = []
        for mention in example["entities"]:
            if not mention["out_of_kg"] and mention["qid"] in artificial_ookg_entities:
                mention["out_of_kg"] = True

                new_mentions.append(mention)
            elif not mention["out_of_kg"]:
                new_mentions.append(mention)
            elif not remove_original_out_of_kg_entities:
                new_mentions.append(mention)
        example["entities"] = new_mentions
    return dataset

def main():
    train = json.load(open("aida_datasets_in_use/aida_transformed_train_filtered.json"))
    testa = json.load(open("aida_datasets_in_use/aida_transformed_testa_filtered.json"))
    testb = json.load(open("aida_datasets_in_use/aida_transformed_testb_filtered.json"))

    all_entities = set()
    for example in testa + testb:
        for mention in example["entities"]:
            if not mention["out_of_kg"]:
                all_entities.add(mention["qid"])

    percentage: float = 0.1
    print(len(all_entities))
    artificial_ookg_entities = set(random.sample(all_entities, int(percentage * len(all_entities))))
    print(len(artificial_ookg_entities))

    train_ookg = remove_out_of_kg_entities(train, artificial_ookg_entities)
    testa_ookg = add_out_of_KG_entities(testa, artificial_ookg_entities)
    testb_ookg = add_out_of_KG_entities(testb, artificial_ookg_entities)

    json.dump(train_ookg, open("aida_transformed_train_filtered_ookg_art.json", "w"), indent=4)
    json.dump(testa_ookg, open("aida_transformed_testa_filtered_ookg_art.json", "w"), indent=4)
    json.dump(testb_ookg, open("aida_transformed_testb_filtered_ookg_art.json", "w"), indent=4)

main()