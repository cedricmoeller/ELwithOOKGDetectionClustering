# Readme for EL with OOKG Detection and Clustering

## Requirements
- Big Graph TransE embeddings (https://dl.fbaipublicfiles.com/torchbiggraph/wikidata_translation_v1.tsv.gz)
- Wikidata dump from 2019-01-28 https://archive.org/details/wikibase-wikidatawiki-20190128 (The link to the dump will be replaced in the future)

## Important files

### datasets/aida
Due to copyright reasons, the modified AIDA dataset is not available here. However, we provide the script to recreate it.
First create the AIDA-CoNLL dataset by following the steps on https://resources.mpi-inf.mpg.de/yago-naga/aida/downloads.html.
Use the created `AIDA-YAGO2-dataset.tsv` and run 
```
python src/dataset_creation/aida/map_aida_dataset.py {AIDA-YAGO2-dataset.tsv}
```
This creates:
- `aida_train_ookg_art_2019.json`: Training dataset based on AIDA-CoNLL with out-of-KG entities
- `aida_testa_ookg_art_2019.json`: Testa dataset based on AIDA-CoNLL with out-of-KG entities
- `aida_testb_ookg_art_2019.json`: Testb dataset based on AIDA-CoNLL with out-of-KG entities

Additionally, the following files are provided:
- `decisive_candidates_174_aida_letitov.json`: Types determined to be relevant for the type information inclusion (necessary if type information is included)
- `le_titov_mention_mapping_2019.json`: Candidate set 


### datasets/wikievents
- `wikievents_2000-2022_train.json`: Training dataset based on AIDA-CoNLL with out-of-KG entities
- `wikievents_2000-2022_dev.json`: Testa dataset based on AIDA-CoNLL with out-of-KG entities
- `wikievents_2000-2022_test.json`: Testb dataset based on AIDA-CoNLL with out-of-KG entities
- `decisive_candidates_71_wikievents_2019.json`: Types determined to be relevant for the type information inclusion (necessary if type information is included)

The generated candidate set is omitted due to its size.

## Preprocessing
### Generating a subset of the KG used for a dataset
Currently, the method only supports a KG which resides in memory. Due to the size of Wikidata, loading the whole Wikidata into memory is infeasible.
(In the future, we will support a SPARQL endpoint as access to a KG.)

Therefore, we provide a script to filter the KG to only include all entities in the candidate set and all ground truth entities. 
To accomplish this, the following needs to be done.

First, we parsed the original dump into a different format via 

``
python src/wikidata_scripts/parse_wikidata_dump.py {dump.json.gz} {parsed.jsonl} 
``

Second, given the list of all qids applicable, the previously generated jsonl file is filtered by executing:

``
python src/wikidata_scripts/create_one_two_hop_type_neighborhood_enriched_mem_kg.py {parsed.jsonl} {output_file.jsonl} --qid_to_filter {json_files_with_qids_as_lists} 
``

Each file given in `json_files_with_qids_as_lists` is contains just a single list of the QIDS in a JSON file.

The final output can be used during training and evaluation.

## Training

To train the method execute the following command:
```
python src/training\train.py ...
```

There are numerous parameters for the training pipeline. 
Get an overview via:
```
python src/training\train.py -h
```


## Evaluation
The evaluation consists of two stages:
### First Stage
Here, a dataset is given and each mention is disambiguated. Furthermore, if the model was trained to also output the out-of-KG score,
each mention is marked as either inKG or out-of-KG.

It is run by:

```
python src/evaluation/standalone_evaluate.py
```

The descriptions of the parameters can again be found via 
```
python src/evaluation/standalone_evaluate.py -h
```
### Second Stage
If the first stage was run with option `second_stage_clustering` set to `True`, then a `results.p` file was created which can be used here.
Optionally, also the development results file can be provided.


It is run by:

```
python src/evaluation/second_stage_clustering.py
```

The descriptions of the parameters can again be found via 
```
python src/evaluation/second_stage_clustering.py -h
```