import json
from pathlib import Path
from typing import List

from tabulate import tabulate
from tqdm import tqdm


def format_results(lst: list):
    lst = [f"{item:.1f}" if isinstance(item, float) else f"{item:,}" for item in lst]
    text = f"{lst[0]}"
    if len(lst) > 1:
        bracket_content = ','.join([x for x in lst[1:]])
        text += f" ({bracket_content})"
    return text

def analyze_dataset(datasets: List[List]):
    num_mentions_all = []
    num_ookg_mentions_all = []
    num_kg_mentions_all = []
    num_unique_ookg_qids_all = []
    num_unique_qids_all = []
    avg_mentions_all = []
    num_examples_all = []
    maximum_active_entities_all = []
    for dataset in datasets:
        print(f"Num examples {len(dataset)}")
        num_ookg_mentions = 0
        num_kg_mentions = 0
        unique_kg_qids = set()
        unique_ookg_qids = set()
        will_be_mentioned_list = []
        for example in dataset:
            for entity in example["entities"]:
                if entity["out_of_kg"]:
                    unique_ookg_qids.add(entity["qid"])
                    num_ookg_mentions += 1
                else:
                    num_kg_mentions += 1
                    unique_kg_qids.add(entity["qid"])
                will_be_mentioned_list.append(entity["qid"])

        maximum_active_entities = 0
        was_mentioned = set()

        counter = 0
        for example in tqdm(dataset):
            for entity in example["entities"]:
                if entity["out_of_kg"]:
                    unique_ookg_qids.add(entity["qid"])
                    num_ookg_mentions += 1
                else:
                    num_kg_mentions += 1
                    unique_kg_qids.add(entity["qid"])
                was_mentioned.add(entity["qid"])
                counter += 1
                will_be_mentioned = set(will_be_mentioned_list[counter:])
                maximum_active_entities = max(maximum_active_entities, len(was_mentioned.intersection(will_be_mentioned)))

        num_mentions = num_ookg_mentions + num_kg_mentions
        num_examples_all.append(len(dataset))
        num_mentions_all.append(num_mentions)
        num_ookg_mentions_all.append(num_ookg_mentions)
        num_kg_mentions_all.append(num_kg_mentions)
        num_unique_ookg_qids_all.append(len(unique_ookg_qids))
        num_unique_qids_all.append(len(unique_kg_qids) + len(unique_ookg_qids))
        avg_mentions_all.append((num_ookg_mentions + num_kg_mentions) / len(dataset))
        maximum_active_entities_all.append(maximum_active_entities)

    return [format_results(num_examples_all), format_results(num_mentions_all), format_results(num_ookg_mentions_all), format_results(num_unique_qids_all),
            format_results(num_unique_ookg_qids_all), format_results(avg_mentions_all), format_results(maximum_active_entities_all)]

def analyze_datasets(train_path: List[Path], dev_path: List[Path], test_path: List[Path], show_overall=True):
    train_data = []
    for x in train_path:
        train_data.append(json.load(x.open()))
    dev_data = []
    for x in dev_path:
        dev_data.append(json.load(x.open()))
    test_data = []
    for x in test_path:
        test_data.append(json.load(x.open()))

    # new_train_data = []
    # for example in train_data:
    #     entities = []
    #     for entity in example["entities"]:
    #         if not entity["out_of_kg"]:
    #             entities.append(entity)
    #     example["entities"] = entities
    #     if entities:
    #         new_train_data.append(example)
    # train_data = new_train_data
    list_of_lists = []
    header = [""]

    stats = analyze_dataset(train_data)
    list_of_lists.append(stats)
    header.append("train")
    stats = analyze_dataset(dev_data)
    list_of_lists.append(stats)
    header.append("dev")
    stats = analyze_dataset(test_data)
    list_of_lists.append(stats)
    header.append("test")
    if show_overall:
        data = [train_data[i] + dev_data[i] + test_data[i] for i in range(len(train_data))]
        stats = analyze_dataset(data)
        list_of_lists.append(stats)
        header.append("all")

    list_of_lists = [list(x) for x in zip(*list_of_lists)]
    list_of_lists[0].insert(0, "# examples")
    list_of_lists[1].insert(0, "# mentions")
    list_of_lists[2].insert(0, "# out-of-KG mentions")
    list_of_lists[3].insert(0, "# unique entities")
    list_of_lists[4].insert(0, "# unique out-of-KG entities")
    list_of_lists[5].insert(0, "Average of # mentions per example")
    list_of_lists[6].insert(0, "Maximum # active entities")


    table = tabulate(list_of_lists, header, floatfmt=".1f", intfmt=",", tablefmt="latex_booktabs")
    print(table)
    print(len(test_data) / (len(test_data) + len(train_data) + len(dev_data)))

analyze_datasets([Path("/data1/anonym/wikinews_extended/wikievents_2000-2022_train.json")],
                [Path("/data1/anonym/wikinews_extended/wikievents_2000-2022_dev.json")],
                [Path("/data1/anonym/wikinews_extended/wikievents_2000-2022_test.json")])


analyze_datasets([Path("/data1/anonym/aida_titov/aida_train_ookg_art_2019.json"), Path("aida_datasets_in_use/aida_transformed_train_filtered.json")],
                [Path("/data1/anonym/aida_titov/aida_testa_ookg_art_2019.json"), Path("aida_datasets_in_use/aida_transformed_testa_filtered.json")],
                [Path("/data1/anonym/aida_titov/aida_testb_ookg_art_2019.json"), Path("aida_datasets_in_use/aida_transformed_testb_filtered.json")])

