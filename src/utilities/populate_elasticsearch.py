from argparse import ArgumentParser

import jsonlines
from elasticsearch import helpers, Elasticsearch
from tqdm import tqdm


def populate_elasticsearch(bulk_size, index_name, filename, total=34826476, entities_to_filter=None):

    es = Elasticsearch()
    es.indices.delete(index=index_name, ignore=404)
    es.indices.create(
        index=index_name,
        ignore=400,
        body={
            "mappings": {
                "properties": {
                    "uri": {"type": "keyword"},
                    "num_claims": {"type": "integer"},
                    "labels": {
                        "type": "nested",
                        "properties": {"value": {"type": "text"}},
                    },
                }
            }
        },
    )
    bulk = []
    input_file = jsonlines.open(filename)

    counter = 0
    pbar = tqdm(total=total)

    for content in input_file:
        if counter == bulk_size:
            counter = 0
            # for success, info in helpers.parallel_bulk(es, bulk):
            #    if not success:
            #        print('A document failed:', info)
            helpers.bulk(es, bulk)
            bulk = []
            pbar.update(bulk_size)

        labels = [{"value": content["labels"]}] + [
            {"value": x} for x in content["aliases"]
        ]
        document = {
            "_index": index_name,
            "_source": {"uri": content["id"],
                        "labels": labels,
                        "num_claims": len(content["claims"])
                        },
        }
        bulk.append(document)

        counter += 1
    else:
        if bulk:
            # for success, info in helpers.parallel_bulk(es, bulk):
            #    if not success:
            #        print('A document failed:', info)
            helpers.bulk(es, bulk)

    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, default="filtered.jsonl")
    parser.add_argument("--index_name", type=str, default="default_index")

    args = parser.parse_args()
    entities_to_filter = None
    populate_elasticsearch(
        bulk_size=10000, index_name=args.index_name, filename=args.filename, entities_to_filter=entities_to_filter
    )
