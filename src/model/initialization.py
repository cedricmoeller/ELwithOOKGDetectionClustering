import copy
import json
import typing
from collections import OrderedDict
from typing import Any, Dict

import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizerFast, \
    AutoAdapterModel

from src.components.entity_manager import CandidateManager, BertCandidateManager
from src.components.kg_connector import KnowledgeGraphConnector, InMemoryKnowledgeGraphConnector, \
    SPARQLKnowledgeGraphConnector
from src.model.embedding_model import BertDocumentEmbedder, EntityEmbeddingBert
from src.model.mention_comparison import MentionComparator
from src.model.ranking_models import SupervisedRankingModel
from src.utilities.special_tokens import ENTITY_DESCRIPTION, ENTITY_START, ENTITY_END
from src.utilities.constants import AdditionalFeatures


def parse_state_dict(state_dict: typing.OrderedDict[str, Any], during_training = False):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key not in {"mention_sim_matrix", "inter_type_projection_matrix"}:
            if not during_training:
                key = key.replace("module.", "")
            new_state_dict[key] = value
    return new_state_dict

def update_args(args: typing.Dict[Any, Any], other_args = None, swap: bool = True):
    if other_args is None:
        other_args = {}
    args: typing.Dict[Any, Any] = {key: value for key, value in args.items() if value is not None}
    args: typing.Dict[Any, Any] = {**other_args, **args, "additional_entity_features": set()}
    # TODO: Remove later. Necessary due to a mistake
    if args.get("use_popularity", False):
        if swap:
            args["additional_entity_features"].add(AdditionalFeatures.EDISTANCE)
        else:
            args["additional_entity_features"].add(AdditionalFeatures.POPULARITY)
    if args.get("use_editdistance", False):
        if swap:
            args["additional_entity_features"].add(AdditionalFeatures.POPULARITY)
        else:
            args["additional_entity_features"].add(AdditionalFeatures.EDISTANCE)
    if args.get("use_prior", False):
        args["additional_entity_features"].add(AdditionalFeatures.PRIOR)
    return args

def initialize_static_objects(args: typing.Dict[Any, Any], skip_unnecessary: bool, sparql_kg=False):
    type_list = json.load(args["type_list"].open()) if args.get("type_list", None) is not None else []

    two_hop_type_list = json.load(args["two_hop_type_list"].open()) if args.get("two_hop_type_list", None) is not None else type_list
    model_checkpoint = "roberta-base"
    tokenizer_class = RobertaTokenizerFast

    # Init document tokenizer and model
    document_tokenizer = tokenizer_class.from_pretrained(model_checkpoint, add_prefix_space=True)
    document_tokenizer.add_special_tokens(
        {"additional_special_tokens": [ENTITY_START, ENTITY_END]}
    )

    # Init entity tokenizer and model
    entity_tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    entity_tokenizer.add_special_tokens(
        {"additional_special_tokens": [ENTITY_DESCRIPTION]}
    )
    if not skip_unnecessary:
        if sparql_kg:
            kg_connector: KnowledgeGraphConnector = SPARQLKnowledgeGraphConnector("https://query.wikidata.org/sparql",
                                                                                  type_list,
                                                                                  two_hop_type_list,
                                                                                  entity_tokenizer,
                                                                                  args["max_length_description_encoder"])
        else:
            kg_connector: KnowledgeGraphConnector = InMemoryKnowledgeGraphConnector(
                args["im_kg_path"], type_list=type_list, two_hop_type_list=two_hop_type_list, n_hops=args["hops"], tokenizer=entity_tokenizer,
                alternative_description_name=args["alternative_description"], max_length_description_encoder=args["max_length_description_encoder"]
            )
    else:
        kg_connector = None

    return document_tokenizer, entity_tokenizer, kg_connector, type_list, two_hop_type_list

def init_adapter_model(model_checkpoint: str, additional_tokens: int):
    model = AutoAdapterModel.from_pretrained(
        model_checkpoint,
    )

    model.resize_token_embeddings(additional_tokens)

    model.add_adapter(
        "el",
    )
    model.train_adapter("el")
    # for param in model.roberta.embeddings.parameters():
    #     param.requires_grad = True

    return model

def init_models(args, num_document_tokens: int, num_entity_tokens: int, stored_model=None,
                num_types: int = 0, num_global_types: int = 0):
    model_class = RobertaModel
    model_checkpoint = "roberta-base"
    adapter_models = args.get("adapter_models", False)
    if adapter_models:
        document_bert_model = init_adapter_model(model_checkpoint, num_document_tokens)

    else:
        document_bert_model = model_class.from_pretrained(model_checkpoint, hidden_dropout_prob=args.get("transformer_dropout"), attention_probs_dropout_prob=args.get("transformer_dropout"))

        document_bert_model.resize_token_embeddings(num_document_tokens)

        for param in document_bert_model.embeddings.parameters():
            param.requires_grad = False

    document_model = BertDocumentEmbedder(
        document_bert_model,
        out_dim=args["dim"],
        transformer_embedding_size=768,
        num_types=num_types
    )


    if adapter_models:
        entity_bert_model = init_adapter_model(model_checkpoint, num_entity_tokens)

    else:
        entity_bert_model = model_class.from_pretrained(model_checkpoint, hidden_dropout_prob=args.get("transformer_dropout"), attention_probs_dropout_prob=args.get("transformer_dropout"))
        entity_bert_model.resize_token_embeddings(num_entity_tokens)

        oldModuleList = entity_bert_model.encoder.layer
        newModuleList = nn.ModuleList()

        # Now iterate over all layers, only keepign only the relevant layers.
        for i in range(0, 2):
            newModuleList.append(oldModuleList[i])

        # create a copy of the model, modify it with the new list, and return
        entity_bert_model = copy.deepcopy(entity_bert_model)
        entity_bert_model.encoder.layer = newModuleList

        # modules = [entity_bert_model.embeddings, *entity_bert_model.encoder.layer[:8]]
        #
        for param in entity_bert_model.embeddings.parameters():
            param.requires_grad = False

    entity_model = EntityEmbeddingBert(
        entity_bert_model,
        dim=args["dim"],
        num_types= num_types,
        use_mean = args["use_mean_in_type_inclusion"]
    )

    ranking_model = SupervisedRankingModel(args["dim"], args.get("ranker_hidden_dim", 100), additional_features= len(args["additional_entity_features"]),
                                           include_ookg_score=args.get("include_ookg_score", True),
                                           use_types = args.get("use_types"),
                                           num_global_types=num_global_types,
                                           use_contextual_types=args.get("use_contextual_types", False),
                                           use_transe=args.get("use_transe", False),
                                           use_cosine=args.get("use_cosine", False),
                                           use_context_embeddings=args.get("use_context_embeddings", True))


    mention_comparator_model = MentionComparator(use_edit_distance=args.get("use_edit_distance_in_comparator", False),
                                                 use_transe_embeddings=args.get("use_transe_embeddings_in_comparator", False),
                                                 use_cosine_similarity=args.get("use_cosine_similarity_in_comparator", True),
                                                 transe_mixing=args.get("alternate_mention_embedding", False))
    if stored_model is not None:
        entity_model.load_state_dict(parse_state_dict(stored_model["entity_model"]), strict=False)
        document_model.load_state_dict(parse_state_dict(stored_model["main_model"]), strict=False)
        ranking_model.load_state_dict(parse_state_dict(stored_model["ranking_model"]), strict=False)
        if "mention_comparator_model" in stored_model:
            mention_comparator_model.load_state_dict(parse_state_dict(stored_model["mention_comparator_model"]), strict=False)

    return document_model, entity_model, ranking_model, mention_comparator_model


def init_for_evaluation(args: Dict[Any, Any], debug=False, swap=False, skip_unnecessary=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stored_model = None
    other_args = {}
    if args["model"] is not None:
        stored_model = torch.load(args["model"], map_location=device)
        other_args = stored_model["args"]
    args = update_args(args, other_args, swap)

    document_tokenizer, entity_tokenizer, kg_connector, type_list, two_hop_type_list = initialize_static_objects(args, skip_unnecessary)
    document_model, entity_model, ranking_model, mention_comparator_model = init_models(args, len(document_tokenizer), len(entity_tokenizer),
                                                              stored_model, num_types=len(type_list), num_global_types=len(two_hop_type_list))
    print("Model initialized")
    document_model = document_model.to(device)
    entity_model = entity_model.to(device)
    ranking_model = ranking_model.to(device)
    mention_comparator_model = mention_comparator_model.to(device)

    candidate_manager: CandidateManager = BertCandidateManager(device, entity_tokenizer, kg_connector, args, type_list=type_list,
                              two_hop_type_list=two_hop_type_list
                                  )
    print("Candidate manager initialized")
    candidate_manager.entity_model = entity_model

    return args, document_tokenizer, entity_tokenizer, kg_connector, type_list, document_model, entity_model, ranking_model, mention_comparator_model, candidate_manager, device