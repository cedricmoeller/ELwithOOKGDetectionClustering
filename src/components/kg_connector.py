from collections import defaultdict
from pathlib import Path
from typing import Tuple, Union, List, Optional

import jsonlines
import torch
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

from src.utilities.utilities import create_type_index_list, transform_entity, encode_fully_connected
from src.utilities.various_dataclasses import KGCandidateEntity


class KnowledgeGraphConnector:
    def get_entity(self, qid: Union[str, dict], **nargs) -> KGCandidateEntity:
        raise NotImplementedError

    def connected_to(self, qid: str) -> Tuple[set, set]:
        raise NotImplementedError


class ElasticSearchKnowledgeGraphConnector(KnowledgeGraphConnector):
    def connected_to(self, qid: str) -> Tuple[set, set]:
        pass

    def get_entity(self, qid: str, **nargs) -> KGCandidateEntity:
        pass


class SPARQLKnowledgeGraphConnector(KnowledgeGraphConnector):
    def __init__(self, endpoint: str, type_list: list, two_hop_type_list: list, tokenizer,
                 max_length_description_encoder):
        self.sparql_wrapper = SPARQLWrapper(endpoint=endpoint)
        self.prefixes = ["PREFIX wd: <http://www.wikidata.org/entity/>",
                         "PREFIX wdt: <http://www.wikidata.org/prop/direct/>",
                         "PREFIX schema: <>"]
        self.sparql_wrapper.setReturnFormat(JSON)
        self.type_list = type_list
        self.type_list_indices = {type_qid: idx for idx, type_qid in enumerate(self.type_list)}
        self.two_hop_type_list = two_hop_type_list
        self.two_hop_type_list_indices = {type_qid: idx for idx, type_qid in enumerate(self.two_hop_type_list)}
        self.tokenizer = tokenizer
        self.max_length_description_encoder = max_length_description_encoder

    def get_entity_via_sparql(self, qid: str) -> Optional[KGCandidateEntity]:
        query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX schema: <http://schema.org/>
            SELECT ?p ?o (lang(?o) as ?lang) where {{
                VALUES ?p {{ rdfs:label skos:altLabel schema:description wdt:P31 wdt:P106 wdt:P641 wdt:P17 }}
                wd:{qid} ?p ?o
            }}
        """

        count_query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            SELECT (count(?o) as ?c) where {{
                wd:{qid} ?p ?o
            }}
        """
        self.sparql_wrapper.setQuery(query=query)

        result = self.sparql_wrapper.queryAndConvert()

        self.sparql_wrapper.setQuery(query=count_query)

        count_result = self.sparql_wrapper.queryAndConvert()

        raw_one_hop_types = defaultdict(int)
        aliases = []
        label = ""
        description = ""
        for r in result["results"]["bindings"]:
            pred = r["p"]["value"]
            obj = r["o"]["value"]
            lang = r["lang"]["value"]
            if pred in {"http://www.wikidata.org/prop/direct/P31", "http://www.wikidata.org/prop/direct/P106",
                        "http://www.wikidata.org/prop/direct/P641", "http://www.wikidata.org/prop/direct/P17"}:
                obj = obj[obj.rfind("/") + 1:]
                raw_one_hop_types[obj] += 1
            else:
                if lang == "en":
                    if pred == "http://schema.org/description":
                        description = obj
                    if pred == "http://www.w3.org/2000/01/rdf-schema#label":
                        label = obj
                    if pred == "http://www.w3.org/2004/02/skos/core#altLabel":
                        aliases.append(obj)

        if not label:
            return None

        one_hop_types = create_type_index_list(self.type_list_indices, [raw_one_hop_types])

        if one_hop_types:
            x, y = zip(*one_hop_types)
            y = [min(y_i, 1) for y_i in y]
            one_hop_types_tensor = torch.sparse_coo_tensor(torch.tensor(x).unsqueeze(0), y, size=(len(self.type_list),),
                                                           dtype=torch.float32)
        else:
            one_hop_types_tensor = torch.sparse_coo_tensor(size=(len(self.type_list),), dtype=torch.float32)



        num_claims = count_result["results"]["bindings"][0]["c"]["value"]

        batch_input_ids, batch_attention_mask = encode_fully_connected([transform_entity(label, description, [], [])],
                                                                       self.tokenizer,
                                                                       max_length_description_encoder=self.max_length_description_encoder)

        input_ids = batch_input_ids[0]
        attention_mask = batch_attention_mask[0]

        kg_candidate = KGCandidateEntity(qid=qid,
                                        label=label,
                                        aliases=aliases,
                                        description=description,
                                        claims=[],
                                        num_claims=num_claims,
                                        input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        one_hop_types=one_hop_types,
                                        one_hop_types_tensor=one_hop_types_tensor,
                                        two_hop_types=[],
                                        two_hop_types_tensor=None,
                        )

        return kg_candidate

    def get_entity(self, qid_content: Union[str, dict], **nargs) -> Union[KGCandidateEntity, str]:
        other_info = {}
        if isinstance(qid_content, dict):
            other_info = qid_content
            qid = qid_content["qid"]
        else:
            qid = qid_content

        entity = self.get_entity_via_sparql(qid)
        if entity is None:
            return qid

        entity.other_info.update(other_info)

        return entity


class InMemoryKnowledgeGraphConnector(KnowledgeGraphConnector):
    def __init__(self, entity_filepath: Path, type_list: list, two_hop_type_list: list, tokenizer,
                 alternative_description_name, max_length_description_encoder,
                 n_hops: int = 1, filter_out_types: bool = True, debug=False):
        print("Initialize InMemoryKnowledgeGraphConnector")

        self.debug = debug
        self.max_num_claims = 0
        self.n_hops = n_hops
        self.filter_out_types = filter_out_types
        self.type_list = type_list
        self.type_list_indices = {type_qid: idx for idx, type_qid in enumerate(self.type_list)}
        self.two_hop_type_list = two_hop_type_list
        self.two_hop_type_list_indices = {type_qid: idx for idx, type_qid in enumerate(self.two_hop_type_list)}

        dump = jsonlines.open(entity_filepath)
        self.entities, self.connections = self.create_kg_entity_structures(dump, tokenizer,
                                                                           alternative_description_name,
                                                                           max_length_description_encoder)

    def create_raw_entity(self, entity: dict, alternative_description_name:str):
        qid = entity["id"]
        num_claims = len(entity.get("claims", []))
        num_claims = max(entity.get("num_claims", 0), num_claims)

        other_keys = set(entity.keys()).difference({"id", "descriptions", "labels", "aliases", "claims"})

        if not entity.get("two_hop_types", {}):
            Warning(f"{entity['id']} has no types. {entity.get('claims')}")

        raw_one_hop_types = entity.get("one_hop_types", {})
        raw_two_hop_types = entity.get("two_hop_types", {})

        one_hop_types = create_type_index_list(self.type_list_indices, [raw_one_hop_types])
        two_hop_types = create_type_index_list(self.two_hop_type_list_indices,
                                               [raw_one_hop_types, raw_two_hop_types])
        one_hop_types_as_two_hop_vector = create_type_index_list(self.two_hop_type_list_indices,
                                                                 [raw_one_hop_types])

        if one_hop_types:
            x, y = zip(*one_hop_types)
            y = [min(y_i, 1) for y_i in y]
            one_hop_types_tensor = torch.sparse_coo_tensor(torch.tensor(x).unsqueeze(0), y, size=(len(self.type_list),),
                                                           dtype=torch.float32)
        else:
            one_hop_types_tensor = torch.sparse_coo_tensor(size=(len(self.type_list),), dtype=torch.float32)
        indices_list = []
        if two_hop_types:
            x, y = zip(*two_hop_types)
            x = [min(x_i, 1) for x_i in x]
            indices_list = [idx for idx, one_hot in enumerate(x) if one_hot]
        two_hop_types_tensor = torch.tensor(indices_list,
                                            dtype=torch.int64)

        label = entity.get("labels")
        if isinstance(label, dict):
            label = label["value"]

        description = entity.get("descriptions", "")
        other_info = {
            key: entity[key] for key in other_keys
        }

        alternative_description = other_info.get(alternative_description_name, None)
        if alternative_description is not None:
            description = alternative_description

        return {
                "qid": qid,
                "description": description,
                "label": label,
                "aliases": entity.get("aliases"),
                "one_hop_types": one_hop_types,
                "one_hop_types_tensor": one_hop_types_tensor,
                "two_hop_types": two_hop_types,
                "two_hop_types_tensor": two_hop_types_tensor,
                "claims": entity.get(
                "claims"),
                "num_claims": num_claims,
                "other_info": other_info
            }

    def instantiate_batch(self, batch: List[dict], tokenizer,
                          max_length_description_encoder: int):
        instantiated_batch = {}

        transformed_entities = []

        for entity in batch:
            transformed_entity = transform_entity(entity["label"], entity["description"])
            transformed_entities.append(transformed_entity)


        batch_input_ids, batch_attention_mask = encode_fully_connected(transformed_entities, tokenizer,
                               max_length_description_encoder=max_length_description_encoder)

        for raw_dict, input_ids, attention_mask in zip(batch, batch_input_ids, batch_attention_mask):
            degree = (raw_dict["num_claims"] / self.max_num_claims) if self.max_num_claims > 0 else 0
            instantiated_batch[raw_dict["qid"]] = KGCandidateEntity(**raw_dict, input_ids=input_ids,
                                                                   attention_mask=attention_mask, degree=degree)
        return instantiated_batch

    def create_kg_entities(self, raw_entities_dict: dict, tokenizer,
                           max_length_description_encoder: int, batch_size=1000):
        batch = []
        entities = {}
        pbar = tqdm(total=len(raw_entities_dict))
        for entity in raw_entities_dict.values():
            batch.append(entity)
            if len(batch) >= batch_size:
                entities.update(self.instantiate_batch(batch, tokenizer, max_length_description_encoder))
                pbar.update(len(batch))
                batch = []
        if batch:
            entities.update(self.instantiate_batch(batch, tokenizer, max_length_description_encoder))
        return entities

    def create_kg_entity_structures(self, dump: jsonlines.Reader, tokenizer, alternative_description_name,
                                    max_length_description_encoder):
        raw_entities = {}
        connections = defaultdict(lambda: (set(), set()))

        for entity in tqdm(dump):
            raw_entities[entity["id"]] = self.create_raw_entity(entity, alternative_description_name)
            num_claims = len(entity.get("claims", []))
            num_claims = max(entity.get("num_claims", 0), num_claims)
            self.max_num_claims = max(num_claims, self.max_num_claims)
            if "claims" in entity:
                for claim in entity["claims"]:
                    connections[entity["id"]][0].add((claim[0], claim[1]))
                    connections[claim[1]][1].add((claim[0], entity["id"]))
            if self.debug:
                break

        entities = self.create_kg_entities(raw_entities, tokenizer,
                                           max_length_description_encoder)

        return entities, connections

    def get_entity(self, qid_content: Union[str, dict], **nargs) -> Union[KGCandidateEntity, str]:
        other_info = {}
        if isinstance(qid_content, dict):
            other_info = qid_content
            qid = qid_content["qid"]
        else:
            qid = qid_content
        if self.debug:
            entity = iter(self.entities.values()).__next__().lazy_copy()
            entity.qid = qid
        else:
            entity = self.entities.get(qid, None)
            if entity is None:
                return qid

        entity.other_info.update(other_info)

        return entity


