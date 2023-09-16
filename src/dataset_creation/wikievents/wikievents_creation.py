import asyncio
import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta, datetime
from pathlib import Path
from time import sleep
from typing import Set, Tuple
from urllib.parse import unquote, quote

import aiohttp
import numpy as np
import requests
from SPARQLWrapper import SPARQLWrapper
from bs4 import BeautifulSoup, Tag, NavigableString
from jsonlines import jsonlines
from pynif import NIFCollection, NIFContext
from tqdm import tqdm

sparql = SPARQLWrapper("https://wdhqs.wmflabs.org/sparql")


def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(("@" + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]["#text"] = text
        else:
            d[t.tag] = text
    return d


def get_url_alt(date_: date):
    year = date_.year
    month = date_.strftime("%B")
    day = date_.day

    return f"https://en.wikipedia.org/w/api.php?action=parse&format=json&prop=wikitext|sections|links|text&page=Portal%3ACurrent%20events%2F{year}%20{month}%20{day}"


def query_api(url):
    r = requests.get(url=url)

    data = r.json()

    data = data["parse"]
    all_links = data["links"]
    # xml = ET.ElementTree(ET.fromstring(data["parsetree"]["*"])).getroot()
    #
    # print(json.dumps(etree_to_dict(xml), indent=4))

    text = data["text"]["*"]
    print(json.dumps(all_links, indent=4))


def get_url(date_: date):
    year = date_.year
    month = date_.strftime("%B")
    day = date_.day

    return f"https://en.wikipedia.org/wiki/Portal:Current_events/{year}_{month}_{day}"


def get_leaves(child):
    leaves = []
    if isinstance(child, Tag):
        if child.name == "li" and all(
            [
                isinstance(content, NavigableString) or content.name == "a"
                for content in child.contents
            ]
        ):
            return [child]
        if child.contents:
            for child_ in child.contents:
                leaves += get_leaves(child_)
    return leaves


def get_str(x):
    if isinstance(x, NavigableString):
        return x
    elif isinstance(x, Tag):
        return x.text


def parse_wikipedia_page(date):
    url = get_url(date)

    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find("div", {"class": "current-events-content description"})
    text_snippets = get_leaves(content)
    final_text_snippets = []

    all_titles = set()

    for text_snippet in text_snippets:
        text = ""
        entities = []
        for content in text_snippet.contents:
            if isinstance(content, Tag):
                page_title: str = content.attrs["href"]
                if not page_title.startswith("/wiki"):
                    continue

                page_title = page_title[6:]
                offset = len(text)

                if len(content.contents) > 1:
                    continue
                text += get_str(content.contents[0])
                length = len(content.contents[0])
                entities.append(
                    {
                        "page_title": page_title,
                        "offset": offset,
                        "length": length,
                        "mention": text[offset : offset + length],
                    }
                )
                all_titles.add(page_title)
            else:
                text += content

        final_text_snippets.append(
            {"text": text, "entities": entities, "date": str(date)}
        )

    return final_text_snippets, all_titles


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


async def get(url, session):
    try:
        async with session.get(url=url) as response:
            return await response.json()
    except Exception as e:
        print("Unable to get url {} due to {}.".format(url, e.__class__))


async def main(urls):
    async with aiohttp.ClientSession() as session:
        ret = await asyncio.gather(*[get(url, session) for url in urls])
    return ret


def parallel_requests(page_identifiers, batch, page_titles):
    split_page_identifiers = [page_identifiers[x:x+batch] for x in list(range(len(page_identifiers)))[0::batch]]
    urls = [link_builder(x, page_titles) for x in split_page_identifiers]
    ret = asyncio.run(main(urls))
    normalized = []
    redirects = []
    pages = {}
    min_not_found = 0
    failed = []
    for idx, result in enumerate(ret):
        if result is not None:
            if "normalized" in result["query"]:
                normalized += result["query"]["normalized"]
            if "redirects" in result["query"]:
                redirects += result["query"]["redirects"]
            for key, page in result["query"]["pages"].items():
                if int(key) < 0 and key not in pages:
                    pages[key] = page
                    min_not_found = min(int(key), min_not_found)
                elif int(key) < 0:
                    pages[str(min_not_found - 1)] = page
                    min_not_found -= 1
                else:
                    pages[key] = page
        else:
            failed += split_page_identifiers[idx]
    return {"query": {
        "normalized": normalized,
        "redirects": redirects,
        "pages": pages
    }}, failed


def link_builder(titles: list, page_titles=True):
    titles_concat = "|".join([quote(x) for x in titles])
    return f"https://en.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&redirects=1&ppprop=wikibase_item&indexpageids=1&{'titles' if page_titles else 'pageids'}={titles_concat}"


def find_redirects(not_found: list, newly_found: list):
    assert len(not_found) == len(newly_found)
    pass

def normalize_page_identifier(input: str) -> str:
    t=unquote(input)
    return t

def get_wikidata_ids(page_identifiers: Set[str], page_titles=True, batch=30, parallel=1):
    page_identifiers = list(page_identifiers)
    page_mappings = {}
    pbar = tqdm(total=len(page_identifiers))
    all_failed = []
    while page_identifiers:
        if parallel > 1:
            tmp = page_identifiers[0:parallel * batch]
            page_identifiers = page_identifiers[parallel * batch:]
            content, failed = parallel_requests(tmp, batch, page_titles)
            all_failed += failed
        else:
            tmp = page_identifiers[0:batch]
            page_identifiers = page_identifiers[batch:]
            tmp = [normalize_page_identifier(x) for x in tmp]
            content = requests.get(link_builder(tmp, page_titles))
            content = content.json()
        content = content["query"]
        normalizations = (
            {x["to"]: x["from"] for x in content["normalized"]}
            if "normalized" in content
            else {}
        )
        redirects = (
            {x["to"]: x["from"] for x in content["redirects"]}
            if "redirects" in content
            else {}
        )

        for x in content["pages"].values():
            remapped_title = x["title"] if page_titles else x["pageid"]
            if remapped_title in redirects:
                remapped_title = redirects[remapped_title]
            if remapped_title in normalizations:
                remapped_title = normalizations[remapped_title]
            # assert str(remapped_title) in tmp
            page_mappings[remapped_title] = x["pageprops"]["wikibase_item"] if "pageprops" in x else None
            if page_mappings[remapped_title] is None:
                all_failed.append(remapped_title)
        pbar.update(parallel * batch)
    return page_mappings, all_failed


parse_datetime = lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")


def parallel_requests_creation_dates(batch: list, query_creation_method):
    urls = [query_creation_method(x) for x in batch]
    ret = asyncio.run(main(urls))
    return ret


def get_wd_creation_dates(qids: list, batch_size=100):
    def create_query(x: str):
        return f"https://www.wikidata.org/w/api.php?action=query&format=json&prop=revisions&titles={x}&rvprop=ids|timestamp|flags|comment|user&rvlimit=1&rvdir=newer"
    creation_dates = {}
    total = len(qids)
    pbar = tqdm(total=total)
    wrongly_processed = []
    while qids:
        sleep(1)
        batch = qids[:batch_size]
        qids = qids[batch_size:]
        if len(batch) > 1:
            responses = parallel_requests_creation_dates(batch, create_query)
        else:
            qid = batch[0]
            responses = [requests.get(
                create_query(qid)
            ).json()]
        for response, qid in zip(responses, batch):
            pages = response["query"]["pages"]
            assert len(pages) == 1
            page = list(pages.values())[0]
            if "title" in page:
                if page["title"] != qid:
                    wrongly_processed.append((qid, page["title"]))
                if "revisions" in page:
                    creation_dates[qid] = page["revisions"][0]["timestamp"]
        pbar.update(len(batch))

    print(wrongly_processed)
    return creation_dates


def extract_creation_dates(final_events_examples: Path, creation_dates_file: Path):
    if not creation_dates_file.exists():
        all_examples = json.load(final_events_examples.open())
        all_entities = set()
        for example_groups in tqdm(all_examples):
            for example in example_groups["examples"]:
                for entity in example["entities"]:
                    all_entities.add(entity["qid"])

        creation_dates = get_wd_creation_dates(list(all_entities))
        json.dump(creation_dates, creation_dates_file.open("w"))


def dump_stats(filename, **fields):
    json.dump(fields, open(f"{filename}.json", "w"), indent=4)



def examine_dataset_ambiguity():
    # mention_dictionary = defaultdict(list)
    # labels = jsonlines.open("labels.json")
    # for item in tqdm(labels, total=81966486):
    #     for x in set([item["label"]] + item["aliases"]):
    #         mention_dictionary[x].append(int(item["qid"]))
    #
    # json.dump(mention_dictionary, open("mention_dictionary.json", "w"))

    mention_dictionary = json.load(open("mention_dictionary.json"))
    all_examples = json.load(open("dataset_creation/events_examples.json"))

    out_of_kg_entities = get_ookg_entity_identifiers(all_examples)

    general_ambiguity = 0
    out_of_kg_ambiguity = 0
    general_no_label = 0
    out_of_kg_no_label = 0
    general_not_found = 0
    out_of_kg_not_found = 0
    out_of_kg_entity_mention_found_counter = 0
    entity_mention_found_counter = 0
    mentions_counter = 0
    out_of_kg_mentions_counter = 0
    for example_block in all_examples:
        for example in example_block["examples"]:
            for entity in example["entities"]:
                mention = entity["mention"]
                qid = entity["qid"]
                corresponding_qids = mention_dictionary.get(mention, [])
                mentions_counter += 1
                if qid in out_of_kg_entities:
                    out_of_kg_mentions_counter += 1
                if mention in mention_dictionary:
                    entity_mention_found_counter += 1
                    general_ambiguity += len(corresponding_qids)
                    if qid in out_of_kg_entities:
                        out_of_kg_entity_mention_found_counter += 1
                        out_of_kg_ambiguity += len(corresponding_qids)
                else:
                    general_no_label += 1
                    if qid in out_of_kg_entities:
                        out_of_kg_no_label += 1

                if int(qid[1:]) not in corresponding_qids:
                    general_not_found += 1
                    if qid in out_of_kg_entities:
                        out_of_kg_not_found += 1

    dump_stats(
        "stats_ambiguity",
        general_ambiguity=general_ambiguity / entity_mention_found_counter,
        out_of_kg_ambiguity=out_of_kg_ambiguity / out_of_kg_entity_mention_found_counter,
        general_no_label=general_no_label / mentions_counter,
        out_of_kg_no_label=out_of_kg_no_label / out_of_kg_mentions_counter,
        general_not_found=general_not_found / mentions_counter,
        out_of_kg_not_found=out_of_kg_not_found / out_of_kg_mentions_counter,
    )


def filter_dataset(dump_related_dataset_info):
    creation_dates = json.load(open("../../../data/other_data/creation_dates"))
    creation_dates = {x: parse_datetime(y).date() for x, y in creation_dates.items()}
    all_examples = json.load(open("../../../data/other_data/events_examples.json"))

    min_date_block = min([date.fromisoformat(x["date"]) for x in all_examples])

    filtered_examples = []

    for example_block in all_examples:
        valid_examples = []
        for example in example_block["examples"]:
            example["entities"] = [x for x in example["entities"] if x["qid"] not in dump_related_dataset_info["entities_to_filter_out"]]
            if not example["entities"]:
                continue

            final_entities = []
            for entity in example["entities"]:
                qid = entity["qid"]
                entity_creation_date = creation_dates[qid]
                entity["out_of_kg"] = entity_creation_date > min_date_block
                if entity["out_of_kg"] or qid not in dump_related_dataset_info["entities_not_in_dump"]:
                    final_entities.append(entity)

            if not final_entities:
                continue

            example["entities"] = final_entities

            valid_examples.append(example)
        if valid_examples:
            example_block["examples"] = valid_examples
            filtered_examples.append(example_block)

    return filtered_examples


def examine_dataset_on_time_dependence(filtered_examples):
    creation_dates = json.load(open("../../../data/other_data/creation_dates"))
    creation_dates = {x: parse_datetime(y).date() for x, y in creation_dates.items()}

    entities = set()

    true_out_of_kg_entities = set()
    true_out_of_kg_entities_increment = 0

    artificial_out_of_kg_entities = set()
    num_artificial_out_of_kg_entity_mentions = 0

    min_date_block = min([date.fromisoformat(x["date"]) for x in filtered_examples])

    num_examples = 0
    dates_list_artificial_out_of_kg_entities = defaultdict(list)

    artificial_out_of_kg_entities_ratio_per_example = []

    num_examples_without_entity_mention = 0

    mentions_per_entity = defaultdict(list)
    mention_ambiguity = defaultdict(set)
    mention_ambiguity_out_of_kg = defaultdict(set)
    entity_mentions_in_example = []
    contains_out_of_kg_entity = []
    for example_block in filtered_examples:
        date_block = date.fromisoformat(example_block["date"])
        for example in example_block["examples"]:
            local_num_artificial_out_of_kg_entity_mentions = 0
            for entity in example["entities"]:
                qid = entity["qid"]
                entity_creation_date = creation_dates[qid]
                if entity_creation_date > date_block:
                    true_out_of_kg_entities_increment += 1
                    true_out_of_kg_entities.add(qid)
                entities.add(qid)
                mentions_per_entity[qid].append(entity["mention"])
                mention_ambiguity[entity["mention"]].add(qid)
                if entity["out_of_kg"]:
                    local_num_artificial_out_of_kg_entity_mentions += 1
                    artificial_out_of_kg_entities.add(qid)
                    dates_list_artificial_out_of_kg_entities[qid].append(date_block)
                    mention_ambiguity_out_of_kg[entity["mention"]].add(qid)
            contains_out_of_kg_entity.append(
                bool(local_num_artificial_out_of_kg_entity_mentions)
            )
            if len(example["entities"]) > 0:
                artificial_out_of_kg_entities_ratio_per_example.append(
                    local_num_artificial_out_of_kg_entity_mentions
                    / len(example["entities"])
                )
            else:
                num_examples_without_entity_mention += 1
            num_artificial_out_of_kg_entity_mentions += (
                local_num_artificial_out_of_kg_entity_mentions
            )
            num_examples += 1
            entity_mentions_in_example.append(len(example["entities"]))

    entity_mentions_per_example = sum(entity_mentions_in_example) / num_examples
    number_mentions_if_out_of_kg = [
        num_entity_mentions
        for num_entity_mentions, is_out_of_kg in zip(
            entity_mentions_in_example, contains_out_of_kg_entity
        )
        if is_out_of_kg
    ]
    avg_number_mentions_if_out_of_kg = sum(number_mentions_if_out_of_kg) / len(
        number_mentions_if_out_of_kg
    )
    median_number_mentions_if_out_of_kg = np.median(number_mentions_if_out_of_kg)

    sum_all_mean_distances = 0
    list_num_mentions = []
    for key, value in dates_list_artificial_out_of_kg_entities.items():
        distances = sum([(value[i + 1] - value[i]).days for i in range(len(value) - 1)])
        mean_distance = distances / len(value)
        if len(value) > 1:
            sum_all_mean_distances += mean_distance

        list_num_mentions.append(len(value))

    num_artificial_out_of_kg_entities_with_single_mention = sum(
        x == 1 for x in list_num_mentions
    )
    num_artificial_out_of_kg_entities_with_multiple_mention = (
        len(dates_list_artificial_out_of_kg_entities)
        - num_artificial_out_of_kg_entities_with_single_mention
    )
    mean_all_distances = (
        sum_all_mean_distances / num_artificial_out_of_kg_entities_with_multiple_mention
    )

    list_multiple_mentions = [x for x in list_num_mentions if x > 1]

    multiple_mentions_average_num_mentions = (
        sum(list_multiple_mentions)
        / num_artificial_out_of_kg_entities_with_multiple_mention
    )

    artificial_out_of_kg_entities_ratio_mean = sum(
        artificial_out_of_kg_entities_ratio_per_example
    ) / len(artificial_out_of_kg_entities_ratio_per_example)
    artificial_out_of_kg_entities_ratio_std = math.sqrt(
        sum(
            [
                (x - artificial_out_of_kg_entities_ratio_mean) ** 2
                for x in artificial_out_of_kg_entities_ratio_per_example
            ]
        )
        / len(artificial_out_of_kg_entities_ratio_per_example)
    )

    multiple_mentions_median_num_mentions = float(np.median(list_multiple_mentions))
    multiple_mentions_max_num_mentions = max(list_multiple_mentions)
    # multiple_mentions_num_mentions_sorted = sorted(list_multiple_mentions, reverse=True)

    artificial_out_of_kg_entities_many_mention_uniqueness = {
        key: len(set(value))
        for key, value in mentions_per_entity.items()
        if key in artificial_out_of_kg_entities and len(set(value)) > 1
    }

    mean_artificial_out_of_kg_entities_many_mention_uniqueness = sum(
        artificial_out_of_kg_entities_many_mention_uniqueness.values()
    ) / len(artificial_out_of_kg_entities_many_mention_uniqueness)
    median_artificial_out_of_kg_entities_many_mention_uniqueness = float(
        np.median(list(artificial_out_of_kg_entities_many_mention_uniqueness.values()))
    )

    mean_mention_ambiguity = sum(len(x) for x in mention_ambiguity.values())/len(mention_ambiguity)
    median_mention_ambiguity = float(np.median([len(x) for x in mention_ambiguity.values()]))

    mean_mention_ambiguity_out_of_kg = sum(len(x) for x in mention_ambiguity_out_of_kg.values()) / len(mention_ambiguity_out_of_kg)
    median_mention_ambiguity_out_of_kg = float(np.median([len(x) for x in mention_ambiguity_out_of_kg.values()]))

    dump_stats(
        "stats_o",
        multiple_mentions_max_num_mentions=multiple_mentions_max_num_mentions,
        mean_mention_ambiguity=mean_mention_ambiguity,
        median_mention_ambiguity=median_mention_ambiguity,
        mean_mention_ambiguity_out_of_kg=mean_mention_ambiguity_out_of_kg,
        median_mention_ambiguity_out_of_kg=median_mention_ambiguity_out_of_kg,
        multiple_mentions_median_num_mentions=multiple_mentions_median_num_mentions,
        multiple_mentions_average_num_mentions=multiple_mentions_average_num_mentions,
        mean_all_distances=mean_all_distances,
        multiple_mentions_single_mention_ratio=len(list_multiple_mentions)
        / len(list_num_mentions),
        num_artificial_out_of_kg_entity_mentions=num_artificial_out_of_kg_entity_mentions,
        num_artificial_out_of_kg_entities=len(artificial_out_of_kg_entities),
        entity_mentions_per_example=entity_mentions_per_example,
        entity_mentions_per_example_median=np.median(entity_mentions_in_example),
        avg_number_mentions_if_out_of_kg=avg_number_mentions_if_out_of_kg,
        median_number_mentions_if_out_of_kg=median_number_mentions_if_out_of_kg,
        num_examples=num_examples,
        num_entities=len(entities),
        artificial_out_of_kg_entities_ratio_mean=artificial_out_of_kg_entities_ratio_mean,
        artificial_out_of_kg_entities_ratio_std=artificial_out_of_kg_entities_ratio_std,
        num_examples_without_entity_mention=num_examples_without_entity_mention,
        mean_artificial_out_of_kg_entities_many_mention_uniqueness=mean_artificial_out_of_kg_entities_many_mention_uniqueness,
        median_artificial_out_of_kg_entities_many_mention_uniqueness=median_artificial_out_of_kg_entities_many_mention_uniqueness,
    )


def get_dump_related_dataset_info():
    all_examples = json.load(open("../../../data/other_data/events_examples.json"))

    entities = set()
    for example_block in all_examples:
        for example in example_block["examples"]:
            for entity in example["entities"]:
                entities.add(entity["qid"])

    entities_with_wrong_type_or_class = set()
    types = {}
    all_types = set()

    for line in jsonlines.open("../../../data/other_data/items_old.jsonl"):
        qid = f"Q{line['qid']}"
        if qid in entities:
            # Remove disambiguation and category pages, remove all entities that have no class and all that are a class
            if not line["subclass_of"] and not ("Q4167836" in line["instance_of"] or "Q4167410" in line["instance_of"]) and line["instance_of"]:
                types[qid] = line["instance_of"]
                all_types.update(line["instance_of"])
            else:
                entities_with_wrong_type_or_class.add(qid)
            entities.remove(qid)

    print(len(all_types))
    return {
        "used_types_no_classes": list(all_types),
        "entities_not_in_dump": list(entities),
        "entities_to_filter_out": list(entities_with_wrong_type_or_class),
    }
    # json.dump(list(all_types), open("used_types_no_classes.json", "w"), indent=4)
    # json.dump(list(entities), open("entities_not_in_dump.json", "w"), indent=4)
    # json.dump(list(entities_with_wrong_type_or_class), open("entities_to_filter.json", "w"), indent=4)


def get_all_events_between(
    start_date: date, end_date: date, tmp_events_examples, all_titles_file, page_mappings_file, all_failed_file, final_events_examples, delay=0.001, recreate=False
):
    if not tmp_events_examples.exists():
        all_examples = []
        all_titles = set()
        for date_ in tqdm(
            daterange(start_date, end_date), total=(end_date - start_date).days
        ):
            x, titles = parse_wikipedia_page(date_)
            all_titles.update(titles)

            all_examples.append({"date": str(date_), "examples": x})
            sleep(delay)

        json.dump(all_examples, tmp_events_examples.open("w"), indent=4)
        json.dump(list(all_titles), all_titles_file.open("w"), indent=4)
    if not page_mappings_file.exists():
        all_titles = set(json.load(all_titles_file.open()))
        page_mappings, all_failed = get_wikidata_ids(all_titles)
        json.dump(page_mappings, page_mappings_file.open("w"), indent=4)
        json.dump(all_failed, all_failed_file.open("w"), indent=4)
    if not final_events_examples.exists():
        all_examples = json.load(tmp_events_examples.open())
        page_mappings = json.load(page_mappings_file.open())
        for example_groups in tqdm(all_examples):
            for example in example_groups["examples"]:
                for entity in example["entities"]:
                    page_title = normalize_page_identifier(entity["page_title"])
                    if "#" in page_title:
                        page_title = page_title[0 : page_title.find("#")]
                    page_title = unquote(page_title)
                    if page_title not in page_mappings:
                        tmp, _ = get_wikidata_ids({entity["page_title"]})
                        tmp = list(tmp.values())
                        assert len(tmp) <= 1
                        entity["qid"] = tmp[0]
                    else:
                        entity["qid"] = page_mappings[page_title]
        json.dump(all_examples, final_events_examples.open("w"), indent=4)


def create_context(example, document_uri):
    context = NIFContext(uri=document_uri, mention=example["text"])
    for entity in example["entities"]:
        context.add_phrase(
            beginIndex=entity["offset"],
            endIndex=entity["offset"] + entity["length"],
            annotator="http://cedric.de",
            taIdentRef=f'http://www.wikidata.org/entity/{entity["qid"]}',
        )
    return context


def convert_to_nif():
    base_uri = "http://current_events_wiki.de"
    collection = NIFCollection(uri=base_uri)
    all_examples = json.load(open("dataset_creation/events_examples.json"))
    counter = 0
    for example_block in tqdm(all_examples):
        for example in example_block["examples"]:
            context = create_context(example, f"{base_uri}/doc{counter}")
            collection.contexts.append(context)
            counter += 1
    collection.dump("events_examples.ttl")



def send_collection(collection, url):
    response = requests.post(
        url,
        data=collection.dumps().encode(),
        headers={"content": "application/x-turtle"},
    )
    new_collection = NIFCollection.loads(response.text)
    return new_collection


def evaluate_response(response, collection) -> Tuple[int, int, int, int]:
    original_phrases = collection.contexts[0].phrases
    response_phrases = response.contexts[0].phrases

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for original_phrase, response_phrase in zip(original_phrases, response_phrases):
        if not response_phrase.taIdentRef.startswith(
            "http://www.wikidata.org/entity/Q"
        ):
            print(response_phrase.taIdentRef)
        if original_phrase.taIdentRef == response_phrase.taIdentRef:
            tp += 1
        else:
            fp += 1
    return tp, fp, tn, fn


def test_dataset_alt(url="https://opentapioca.org/api/nif"):
    all_examples = json.load(open("data/execution_relevant_data/events_examples_filtered.json"))

    overall_tp = overall_fp = overall_tn = overall_fn = 0

    pbar = tqdm(total=sum([len(x["examples"]) for x in all_examples]))
    counter = 0
    for example_block in all_examples:
        for example in example_block["examples"]:

            try:
                collection = NIFCollection()
                context = create_context(example, f"")
                collection.contexts.append(context)
                response = send_collection(collection, url)
                tp, fp, tn, fn = evaluate_response(response, collection)
                overall_tp += tp
                overall_fp += fp
                overall_tn += tn
                overall_fn += fn
            except Exception as e:
                print(f"Error at example {counter}")
            pbar.update(1)
            counter += 1

    precision = overall_tp / (overall_tp + overall_fp)
    recall = overall_tp / (overall_tp + overall_fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    dump_stats(
        "EL_test",
        precision=precision,
        recall=recall,
        f1=f1,
        overall_fp=overall_fp,
        overall_tp=overall_tp,
    )


def test_dataset(url="https://opentapioca.org/api/nif"):
    collection = NIFCollection.load("events_examples.ttl")

    for context in collection.contexts:
        response = send_collection(context)

        tp, fp, tp, np = evaluate_response(response, context)


def get_ookg_entity_identifiers(cutoff_date: date, creation_dates_file: Path, final_events_examples_file: Path):

    all_examples = json.load(final_events_examples_file.open())
    creation_dates = json.load(creation_dates_file.open())
    creation_dates = {x: parse_datetime(y).date() for x, y in creation_dates.items()}
    is_none = 0
    ookg_entities = set()
    for example_block in all_examples:
        for example in example_block["examples"]:
            for entity in example["entities"]:
                qid = entity["qid"]
                entity_creation_date = creation_dates.get(qid)
                if entity_creation_date is None or entity_creation_date > cutoff_date:
                    ookg_entities.add(qid)
                if entity_creation_date is None:
                    is_none += 1

    return ookg_entities


def split_time_sensitive_data(
    dataset: list, cutoff_date: date
):

    train_set = []
    rest = []
    for example in dataset:
        example_date = datetime.strptime(example["date"], '%Y-%m-%d').date()
        if example_date <= cutoff_date:
            train_set.append(example)
        else:
            rest.append(example)

    dev_indices = set(random.sample(list(range(len(rest))), len(rest) // 2))
    new_dev_set = []
    new_test_set = []

    for idx, element in enumerate(rest):
        if idx in dev_indices:
            new_dev_set.append(element)
        else:
            new_test_set.append(element)

    return train_set, new_dev_set, new_test_set


def clean_data(dataset_path: Path):
    dataset = json.load(dataset_path.open())

    valid_qids = set()
    upper_case_ratio = defaultdict(int)
    counter = defaultdict(int)
    dataset = [example for x in dataset for example in x["examples"]]
    for example in dataset:
        for entity in example["entities"]:
            if entity["mention"][0].isupper():
                valid_qids.add(entity["qid"])
                upper_case_ratio[entity["qid"]] += 1
            counter[entity["qid"]] += 1

    new_dataset = []
    removed = 0
    for example in dataset:
        final_entities = []
        for entity in example["entities"]:
            if entity["qid"] in valid_qids and upper_case_ratio[entity["qid"]]/counter[entity["qid"]] > 0.5:
                final_entities.append(entity)
            else:
                removed += 1

        if final_entities:
            example["entities"] = final_entities
            new_dataset.append(example)
    print(f"Removed: {removed}")
    return new_dataset

def mark_ookg(dataset: list, ookg_entities: set):
    for example in dataset:
        for entity in example["entities"]:
            entity["out_of_kg"] = entity["qid"] in ookg_entities
    return dataset

def remove_entities_not_in_kg(data: list, kg_path: Path):
    qids_in_kg = set()
    for item in tqdm(jsonlines.open(kg_path)):
        qids_in_kg.add(item["id"])

    new_data = []
    for example in data:
        new_entities = []
        for entity in example["entities"]:
            if entity["qid"] in qids_in_kg or entity["out_of_kg"]:
                new_entities.append(entity)
        if new_entities:
            example["entities"] = new_entities
            new_data.append(example)

    return new_data



def creation_pipeline(start_date: date, end_date: date, cutoff_date: date, main_name: str, kg_path: Path):
    tmp_events_examples = Path(f"{main_name}_event_examples_raw.json")
    all_titles_file = Path(f"{main_name}_all_titles.json")
    page_mappings_file = Path(f"{main_name}_page_mappings.json")
    all_failed_file = Path(f"{main_name}_all_failed.json")
    final_events_examples = Path(f"{main_name}_event_examples_extracted.json")
    creation_dates_file = Path(f"{main_name}_creation_dates.json")

    get_all_events_between(start_date, end_date, tmp_events_examples, all_titles_file, page_mappings_file, all_failed_file, final_events_examples)
    extract_creation_dates(final_events_examples, creation_dates_file)

    ookg_entities = get_ookg_entity_identifiers(cutoff_date, creation_dates_file, final_events_examples)

    cleaned_data = clean_data(final_events_examples)
    marked_data = mark_ookg(cleaned_data, ookg_entities)
    marked_data = remove_entities_not_in_kg(marked_data, kg_path)
    train_set, dev_set, test_set = split_time_sensitive_data(marked_data, cutoff_date)

    json.dump(train_set, open(f"{main_name}_train.json", "w"), indent=4)
    json.dump(dev_set, open(f"{main_name}_dev.json", "w"), indent=4)
    json.dump(test_set, open(f"{main_name}_test.json", "w"), indent=4)




creation_pipeline(date(1999, 12, 29), date(2022, 10, 1), date(2019,1,28), "/data1/anonym/wikinews_extended/wikievents_2000-2022",
                  Path("/data1/anonym/wikidata-old/parsed_dump_2019.jsonl"))

