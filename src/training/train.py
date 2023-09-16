import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import configargparse
import numpy
import numpy as np
import torch
from torch import nn
import wandb
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss
from torch.nn.functional import normalize
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import pairwise_linear_similarity
from tqdm import tqdm
from transformers import RobertaTokenizerFast, get_linear_schedule_with_warmup

from src.components.entity_manager import BertCandidateManager, \
    prepare_document_containers
from src.components.kg_connector import (
    InMemoryKnowledgeGraphConnector,
    KnowledgeGraphConnector,
)
from src.utilities.dataloaders import (
    instantiate_dataloader, )
from src.evaluation.evaluate import Evaluator
from src.model.embedding_model import EntityEmbeddingBert, BertDocumentEmbedder
from src.model.initialization import init_models
from src.model.mention_comparison import MentionComparator
from src.model.ranking_models import SupervisedRankingModel
from src.utilities.special_tokens import ENTITY_START, ENTITY_END, ENTITY_DESCRIPTION
from src.utilities.constants import AdditionalFeatures, TEMP
from src.utilities.utilities import update_current_representations, calculate_similarities, \
    get_nested_candidates, create_candidate_from_mention, pairwise_loss
from src.utilities.various_dataclasses import DocumentContainerForProcessing, \
    CurrentStats, MentionContainerForProcessing, CandidateContainerForProcessing, EmbeddedMention


def set_seeds(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)

def is_true(x: str):
    return str(x).lower() in {"true", "1"}


def init(args, num_document_tokens, num_entity_tokens, stored_model=None, parallelized=True, num_types: int = 0,
         num_global_types: int  = 0):
    parallelize = torch.cuda.device_count() > 1 and parallelized
    (
        document_model,
        entity_model,
        ranking_model,
        mention_comparator_model,
    ) = init_models(args,num_document_tokens, num_entity_tokens, stored_model, num_types=num_types,
                    num_global_types=num_global_types)

    if parallelize:
        document_model = nn.DataParallel(document_model)


    # Init optimizer

    return (
        document_model,
        entity_model,
        ranking_model,
        mention_comparator_model,
        parallelize,
    )

class Trainer:
    def __init__(self, args: dict):
        args["additional_entity_features"] = set()
        if args.get("use_popularity", False):
            args["additional_entity_features"].add(AdditionalFeatures.POPULARITY)

        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mention_split: int = args.get("mention_split")
        self.mentions_to_add_if_full: int = args.get("mentions_to_add_if_full")

        self.cosine_loss_func: CosineEmbeddingLoss = CosineEmbeddingLoss()
        self.ce_loss =  CrossEntropyLoss()
        self.bce_loss = nn.BCELoss(reduction='sum')

        self.type_list = json.load(self.type_list.open()) if self.type_list else []

        if self.two_hop_type_list is None:
            self.two_hop_type_list = self.type_list
        else:
            self.two_hop_type_list = json.load(self.two_hop_type_list.open())

        model_checkpoint = "roberta-base"
        tokenizer_class = RobertaTokenizerFast

        print("Initialize tokenizers")
        # Init document tokenizer and model
        self.document_tokenizer = tokenizer_class.from_pretrained(model_checkpoint, add_prefix_space=True)
        self.document_tokenizer.add_special_tokens(
            {"additional_special_tokens": [ENTITY_START, ENTITY_END]}
        )

        # Init entity tokenizer and model
        self.entity_tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
        self.entity_tokenizer.add_special_tokens(
            {"additional_special_tokens": [ENTITY_DESCRIPTION]}
        )

        self.kg_connector: KnowledgeGraphConnector = InMemoryKnowledgeGraphConnector(
            self.im_kg_path, tokenizer=self.entity_tokenizer,
            alternative_description_name=self.args.get("alternative_description", None),
            max_length_description_encoder=self.args.get("max_length_description_encoder"),
            type_list=self.type_list, two_hop_type_list=self.two_hop_type_list, n_hops=self.hops
        )



        print("Initialize candidate manager")
        self.candidate_manager: BertCandidateManager = BertCandidateManager(
                self.device,
                self.entity_tokenizer,
                self.kg_connector,
                type_list=self.type_list,
                two_hop_type_list=self.two_hop_type_list,
                args=self.args,
                mask_token_id=self.document_tokenizer.mask_token_id
            )

        print("Initialize evaluator")
        self.evaluator = Evaluator(
            self.document_tokenizer,
            self.candidate_manager,
            self.device,
            self.kg_connector,
            self.args,
            num_types=len(self.type_list)
        )
        self.backup_name = f"model{self.args['file_suffix']}_backup"

        if self.load_model is None:
            backup_path = Path(self.backup_name)
            if backup_path.exists():
                self.load_model = backup_path

        assert self.separate_mention_training or len(self.additional_entity_features) == 0
        print("Finished main initialization.")

    def __getattr__(self, item: str):
        if item != "args":
            if item in self.args:
                return self.args[item]
        raise AttributeError

    def restore_from_loaded_model(self, model_to_load: Optional[Path]):
        start_epoch = 0
        start_step = 0
        stored_model = None
        if model_to_load is not None:
            print("Load existing model.")
            stored_model = torch.load(model_to_load, map_location=torch.device("cpu"))
            args = stored_model["args"]
            mismatching_args = {}
            for key, value in args.items():
                if self.args.get(key) != value:
                    mismatching_args[key] = (self.args.get(key), value)
            if mismatching_args:
                warning = "Arguments are mismatching at (given arguments, loaded model arguments):\n"
                for key, value in mismatching_args.items():
                    warning += f"{key}: {value}\n"
                print(warning)
            args.update(self.args)
            self.args = args
            start_epoch = stored_model["epoch"] + (0 if "step" in stored_model else 1)
            start_step = stored_model.get("step", 0)
        return start_epoch, start_step, stored_model

    def set_candidate_manager_for_new_run(self, entity_model, document_model,
                                          mention_document_index: dict = None):
        self.candidate_manager.mention_document_index = mention_document_index
        self.candidate_manager.entity_model = entity_model
        self.candidate_manager.document_embedder = document_model
        self.candidate_manager.reset_representations()


    def init_train_run(self, num_examples: int = 0, model_to_load: Optional[Path] = None,
                       mention_document_index: dict = None) -> Tuple[List[int], int, BertDocumentEmbedder, EntityEmbeddingBert, SupervisedRankingModel, MentionComparator, AdamW, LambdaLR]:
        if model_to_load is None and not self._10sec:
            model_to_load = self.load_model

        is_backup = False
        if model_to_load is not None:
            is_backup = model_to_load.name == self.backup_name
            if is_backup:
                print("Found a backup.")
        start_epoch, start_step, stored_model = self.restore_from_loaded_model(model_to_load)
        if not is_backup:
            start_epoch = 0
            start_step = 0

        (
            document_model,
            entity_model,
            ranking_model,
            mention_comparator_model,
            self.parallel,
        ) = init(self.args, len(self.document_tokenizer), len(self.entity_tokenizer), stored_model,
                 num_types=len(self.type_list), num_global_types=len(self.two_hop_type_list))

        document_model = document_model.to(self.device)
        entity_model = entity_model.to(self.device)
        ranking_model = ranking_model.to(self.device)
        mention_comparator_model = mention_comparator_model.to(self.device)

        self.set_candidate_manager_for_new_run(entity_model, document_model, mention_document_index)

        optimizer = AdamW(
            [
                {"params": document_model.embedding_model.parameters(), "lr": self.lr},
                {"params": document_model.projection.parameters(), "lr": self.head_lr},
                {"params": entity_model.bert.parameters(), "lr": self.lr},
                {"params": entity_model.mean_layer.parameters(), "lr": self.head_lr},
                {"params": document_model.entity_typer.parameters(), "lr": self.head_lr},
                {"params": ranking_model.parameters(), "lr": self.head_lr},
                {"params": mention_comparator_model.parameters(), "lr": self.head_lr},
            ],
            lr=self.lr,
        )
        if stored_model is not None and is_backup:
            optimizer.load_state_dict(stored_model['optimizer_state_dict'])

        if self._10sec:
            self.args["num_epochs"] = 1

        epochs = list(range(self.num_epochs))
        epochs = epochs[start_epoch:]

        num_training_steps = (len(epochs) * num_examples) / self.gradient_accumulation_size

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,  # 0.1 * num_training_steps,
            num_training_steps=num_training_steps,
        )

        if stored_model is not None and is_backup:
            scheduler.load_state_dict(stored_model['scheduler_state_dict'])

        self.candidate_manager.train = True

        return epochs, start_step, document_model, entity_model, ranking_model, mention_comparator_model, optimizer, scheduler

    def regular_training(self, training_dataloader: DataLoader, dev_data_loader: DataLoader, test_data_loader: DataLoader,
                         writer: SummaryWriter):
        (test_score, test_el_fmeasure, test_id_fmeasure,
         _,
         _,
         _
         ) = self.meta_train_operation(
            training_dataloader,
            dev_data_loader,
            test_data_loader,
            writer,
        )

    def train(self):
        set_seeds(self.seed)
        writer = SummaryWriter()

        best_dev_score = 0
        if self.zero_shot:
            # Deprecated
            train = json.load(
                self.training_dataset_path.open()) if self.training_dataset_path.suffix == ".json" else self.training_dataset_path
            training_dataloader = instantiate_dataloader(
                train, self.kg_connector, self.candidate_manager.transe_mappings, self.candidate_manager.transe_embeddings, self.document_tokenizer, max_length=self.max_length, mask_ratio=0.7,
                **{"batch_size": self.batch_size, "shuffle": True},
            )
            test_data_loader = None
            dev_data_loader = instantiate_dataloader(
                json.load(
                    self.dev_dataset_path.open()), self.kg_connector, self.candidate_manager.transe_mappings, self.candidate_manager.transe_embeddings,
                self.document_tokenizer, max_length=self.max_length,
                **{"batch_size": 100, "shuffle": False},
            )

        else:
            print("Load train dataset.")
            train = json.load(self.training_dataset_path.open()) if self.training_dataset_path.suffix == ".json" else self.training_dataset_path

            if self._10sec and isinstance(train, list):
                train = train[:int(0.02 * len(train))]

            if self.limited and isinstance(train, list):
                train = train[:int(0.1 * len(train))]

            training_dataloader = instantiate_dataloader(
                train, self.kg_connector, self.candidate_manager.transe_mappings, self.candidate_manager.transe_embeddings, self.window_size, self.document_tokenizer, max_length=self.max_length, mask_ratio=self.mask_ratio,
                **{"batch_size": self.batch_size, "shuffle": True},
            )
            print("Load test dataset.")
            test = json.load(
                self.test_dataset_path.open()) if self.test_dataset_path.suffix == ".json" else self.test_dataset_path

            if self._10sec and isinstance(test, list):
                test = test[:int(0.02 * len(test))]

            if self.limited and isinstance(test, list):
                test = test[:int(0.1 * len(test))]

            test_data_loader = instantiate_dataloader(
                test, self.kg_connector, self.candidate_manager.transe_mappings, self.candidate_manager.transe_embeddings, self.window_size, self.document_tokenizer, max_length=self.max_length,
                **{"batch_size": 20, "shuffle": False},
            )

            if self.dev_dataset_path is not None:
                print("Load dev dataset")
                dev = json.load(self.dev_dataset_path.open()) if self.dev_dataset_path.suffix == ".json" else self.dev_dataset_path

                if self._10sec and isinstance(dev, list):
                    dev = dev[:int(0.02 * len(dev))]
                if self.limited and isinstance(dev, list):
                    dev = dev[:int(0.1 * len(dev))]

                dev_data_loader = instantiate_dataloader(
                    dev, self.kg_connector, self.candidate_manager.transe_mappings, self.candidate_manager.transe_embeddings, self.window_size, self.document_tokenizer, max_length=self.max_length,
                    **{"batch_size": 20, "shuffle": False},
                )
            else:
                dev_data_loader = test_data_loader

        print("Datasets loaded.")
        self.regular_training(training_dataloader, dev_data_loader, test_data_loader, writer)

        writer.add_hparams(self.parsed_args, {"current_best_score": best_dev_score})
        writer.flush()
        writer.close()

    def meta_train_operation(
        self,
        training_dataloader,
        dev_dataloader,
        test_dataloader,
        writer,
        patience=10
    ):
        if not self.zero_shot:
            mention_document_index = self.prepare_mention_document_index(
                training_dataloader
            )
        else:
            mention_document_index = {}



        epochs, start_step, document_model, entity_model, ranking_model, mention_comparator_model, optimizer, scheduler = self.init_train_run(len(training_dataloader), mention_document_index=mention_document_index)

        current_stats = CurrentStats()
        # Initialize evaluator
        average_losses = []
        test_score = 0
        test_el_fmeasure = 0
        test_id_fmeasure = 0

        for epoch in epochs:
            wandb.log({"epoch": epoch})
            print(f"Epoch {epoch}")

            document_model.train()
            entity_model.train()
            ranking_model.train()
            self.candidate_manager.train = True

            average_loss = self.train_epoch(
                epoch, training_dataloader, dev_dataloader, document_model, entity_model, ranking_model, mention_comparator_model, optimizer, scheduler, writer, current_stats, step=start_step if epoch == epochs[0] else 0
            )
            average_loss = 0
            writer.add_scalar(
                f"Loss/train",
                average_loss,
                epoch,
            )
            average_losses.append(average_loss)

            self.store_models(
                Path(self.backup_name),
                document_model, entity_model, ranking_model, mention_comparator_model, optimizer,
                scheduler,
                epoch=epoch,
                loss=average_losses[-1],
                average_losses=average_losses,
            )

            if dev_dataloader is not None and self.evaluate_every_n_epoch and (
                (epoch + 1) % self.evaluate_every_n_epoch == 0 or self._10sec
            ):
                # Evaluate the model on dev data
                evaluation_results, _ = self.evaluator.evaluate_model(
                    dev_dataloader,
                    document_model,
                    entity_model,
                    ranking_model,
                    mention_comparator_model,
                    self.mention_split,
                    1,
                    number_candidates=self.number_candidates_eval
                )
                self.evaluator.candidate_manager.reset_representations()
                print(json.dumps(evaluation_results, indent=4))
                identification_harmonic_mean = evaluation_results[
                    "identification_harmonic_mean"
                ]
                wandb.log(evaluation_results)
                f_measure_in_kg_non_ookg = evaluation_results["f_measure_in_kg_non_ookg"]
                loss = evaluation_results["loss"]
                mention_loss = evaluation_results["mention_loss"]
                combined_loss = loss + mention_loss
                combined_measure = (
                    identification_harmonic_mean + f_measure_in_kg_non_ookg
                ) / 2
                for key, value in evaluation_results.items():
                    if key != "args":
                        if not isinstance(value, dict):
                            writer.add_scalar(
                                f"evaluation_results_{key}",
                                value,
                                epoch,
                            )
                if combined_loss <= current_stats.best_combined_loss:
                    self.store_models(
                        Path(
                            f"current_best_model{self.args['file_suffix']}"
                        ),
                        document_model, entity_model, ranking_model, mention_comparator_model,optimizer,
                        scheduler,
                        epoch=epoch,
                        loss=average_losses[-1],
                        average_losses=average_losses,
                        evaluation_results=evaluation_results,
                    )
                    current_stats.best_dev_score = float(combined_measure)
                    current_stats.best_dev_el_fmeasure = float(f_measure_in_kg_non_ookg)
                    current_stats.best_dev_id_fmeasure = float(identification_harmonic_mean)
                    current_stats.best_loss = loss
                    current_stats.best_mention_loss = mention_loss
                    current_stats.best_combined_loss = combined_loss
                    current_stats.no_improvement = 0
                else:
                    current_stats.no_improvement += 1
                if current_stats.no_improvement > patience:
                    break
            else:
                evaluation_results = {}

            if self.store_every_n_epoch:
                if (epoch + 1) % self.store_every_n_epoch == 0:
                    self.store_models(
                        Path(
                            f"model{self.args['file_suffix']}_epoch_{epoch}"
                        ),
                        document_model, entity_model, ranking_model, mention_comparator_model, optimizer,
                        scheduler,
                        epoch=epoch,
                        loss=average_losses[-1],
                        average_losses=average_losses,
                        evaluation_results=evaluation_results,
                    )

        if test_dataloader is not None:
            _, _, document_model, entity_model, ranking_model, mention_comparator_model, _, _ = self.init_train_run(model_to_load=Path(f"current_best_model{self.args['file_suffix']}"), mention_document_index=mention_document_index)
            evaluation_results, _ = self.evaluator.evaluate_model(
                test_dataloader,
                document_model,
                entity_model,
                ranking_model,
                mention_comparator_model,
                self.mention_split,
                1,
                number_candidates = self.number_candidates_eval
            )
            wandb.log(evaluation_results)
            print(json.dumps(evaluation_results, indent=4))
            identification_harmonic_mean = evaluation_results[
                "identification_harmonic_mean"
            ]
            f_measure_in_kg_non_ookg = evaluation_results["f_measure_in_kg_non_ookg"]
            combined_measure = (
                identification_harmonic_mean + f_measure_in_kg_non_ookg
            ) / 2
            wandb.log({"combined_measure": combined_measure})
            test_score = combined_measure
            test_el_fmeasure = f_measure_in_kg_non_ookg
            test_id_fmeasure = identification_harmonic_mean
            writer.add_scalar(
                f"evaluation_results_current_best_score_test",
                combined_measure,
            )
            for key, value in evaluation_results.items():
                if key != "args":
                    if not isinstance(value, dict):
                        writer.add_scalar(
                            f"evaluation_results_{key}_test", value
                        )

        return test_score, test_el_fmeasure, test_id_fmeasure, current_stats.best_dev_score, current_stats.best_dev_el_fmeasure, current_stats.best_dev_id_fmeasure

    @property
    def parsed_args(self):
        parsed_args = {}
        for key, value in self.args.items():
            if (
                isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, str)
                or isinstance(value, bool)
                or isinstance(value, torch.Tensor)
            ):
                parsed_args[key] = value
            else:
                parsed_args[key] = str(value)

        return parsed_args

    @staticmethod
    def prepare_mention_document_index(training_dataloader: DataLoader) -> Dict[str, List[dict]]:
        index = defaultdict(list)
        for example in training_dataloader.dataset.examples:
            for mention_idx, entity_mention in enumerate(example.mentions):
                if entity_mention.label_qid is not None:
                    index[entity_mention.label_qid].append((example, mention_idx))

        return index

    def mix_representations_and_add_other_mentions_as_negatives(self, entity_representations, current_entity_representations,
                                                                new_embedded_mentions: List[List[EmbeddedMention]],
                                                                current_embedded_mentions, mix_other_mentions: bool):
        if self.num_weak_negatives_mentions is None:
            num_weak_negatives_mentions = self.num_weak_negatives
        else:
            num_weak_negatives_mentions = self.num_weak_negatives_mentions
        if self.num_weak_negatives > 0 or num_weak_negatives_mentions > 0:
            all_entity_candidates = []
            all_mention_candidates = []
            entity_candidate_indices = defaultdict(list)
            mention_candidate_indices = defaultdict(list)

            for content_tuple in zip(current_entity_representations,current_embedded_mentions):
                for current_entity_representations_, current_embedded_mention in zip(*content_tuple):
                    for candidate in current_entity_representations_:
                        if candidate.complex_candidate.is_kg_candidate:
                            if candidate.complex_candidate.qid == current_embedded_mention.mention_container.label_qid:
                                entity_candidate_indices[candidate.complex_candidate.qid].append(len(all_entity_candidates))
                                all_entity_candidates.append(candidate)
                        else:
                            mention_candidate_indices[candidate.complex_candidate.qid].append(len(all_mention_candidates))
                            all_mention_candidates.append(candidate)
                    if mix_other_mentions:
                        mention_candidate_indices[current_embedded_mention.mention_container.label_qid].append(len(all_mention_candidates))
                        all_mention_candidates.append(create_candidate_from_mention(current_embedded_mention, self.device))

            taboo_lists = []
            for candidates, embedded_mentions in zip(entity_representations, new_embedded_mentions):
                taboo_list_example = [entity_candidate_indices[item.mention_container.label_qid] for item in embedded_mentions]

                for entity_representations_, current_mention in zip(candidates, embedded_mentions):
                    taboo_list_mentions_example = [mention_candidate_indices[current_mention.mention_container.label_qid]]
                    taboo_list = (taboo_list_example,
                                  taboo_list_mentions_example)
                    for candidate in entity_representations_:
                        if candidate.complex_candidate.is_kg_candidate:
                            taboo_list[0].append(entity_candidate_indices[candidate.complex_candidate.qid])
                        else:
                            taboo_list[1].append(mention_candidate_indices[candidate.complex_candidate.qid])

                    taboo_lists.append(taboo_list)

            for mention_candidates, mention_container, (taboo_list_entities, taboo_list_mentions) in zip(itertools.chain(*entity_representations), itertools.chain(*new_embedded_mentions), taboo_lists):
                entity_indices_to_sample = np.arange(len(all_entity_candidates), dtype='int32')
                mention_indices_to_sample = np.arange(len(all_mention_candidates), dtype='int32')
                entity_indices_to_sample = list(np.delete(entity_indices_to_sample, list(itertools.chain(*taboo_list_entities))))
                mention_indices_to_sample = list(np.delete(mention_indices_to_sample, list(itertools.chain(*taboo_list_mentions))))
                sampled_kg_negatives = random.sample(entity_indices_to_sample,
                                                            min(self.num_weak_negatives,
                                                                len(entity_indices_to_sample)))
                sampled_mention_negatives = random.sample(mention_indices_to_sample,
                                                            min(num_weak_negatives_mentions,
                                                                len(mention_indices_to_sample)))

                for idx in sampled_kg_negatives:
                    mention_candidates.append(all_entity_candidates[idx])

                for idx in sampled_mention_negatives:
                    mention_candidates.append(all_mention_candidates[idx])

    def optimize(self, nested_candidates, mention_idx, sub_sampled_embedded_mentions,
                 current_entity_representations, mention_split,
                 current_embedded_mentions, cls_embeddings, ranking_model, mention_comparator_model, already_embedded,
                 overall_num_mentions, overall_num_mentions_inkg, maximum_num_mentions, num_candidates_with_mentions):

        clusterized_candidates_embedded = self.candidate_manager.get_candidate_representations(nested_candidates,
                                                                                               sub_sampled_embedded_mentions,
                                                                                               mention_comparator_model)

        update_current_representations(clusterized_candidates_embedded, sub_sampled_embedded_mentions,
                                       current_entity_representations,
                                       current_embedded_mentions, mention_split)

        self.mix_representations_and_add_other_mentions_as_negatives(clusterized_candidates_embedded,
                                                                     current_entity_representations,
                                                                     sub_sampled_embedded_mentions, current_embedded_mentions,
                                                                     self.mix_other_mentions)

        if not clusterized_candidates_embedded:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        document_containers = prepare_document_containers(
            cls_embeddings,
            sub_sampled_embedded_mentions,
            clusterized_candidates_embedded,
        )

        calculate_similarities(document_containers, ranking_model)

        cs_loss = torch.zeros((), device=self.device)
        t_loss = torch.zeros((), device=self.device)
        b_loss = torch.zeros((), device=self.device)

        el_loss, entropy_loss, mention_loss = self.rank(
            ranking_model,
            mention_comparator_model,
            document_containers,
            already_embedded,
            overall_num_mentions=overall_num_mentions,
            overall_num_mentions_inkg=overall_num_mentions_inkg,
            mention_split=mention_split,
            num_candidates_with_mentions=num_candidates_with_mentions
        )

        for embedded_document in document_containers:
            for mention in embedded_document.mentions:
                if not mention.embedded_mention.mention_container.label_out_of_kg:
                    ground_truth_indices = torch.tensor([i for i, x in enumerate(mention.candidate_representations) if
                                                         x.complex_candidate.qid == mention.embedded_mention.mention_container.label_qid],
                                                        device=self.device)

                    if self.embedding_comparison_weight > 0.0 and self.use_context_embeddings:
                        scores = []
                        for c in mention.candidate_representations:
                            assert not self.separate_mention_training or c.complex_candidate.is_kg_candidate
                            scores.append(c.similarity)
                        if scores:
                            scores = torch.stack(scores)
                            if self.use_cosine:
                                scores /= TEMP
                            length = 1
                            if self.use_sample_weight:
                                length = len(self.candidate_manager.mention_document_index[
                                                 mention.embedded_mention.mention_container.label_qid])
                                length = length if length > 0 else 1
                            for ground_truth_index in ground_truth_indices:
                                cs_loss += self.ce_loss(scores,
                                                        ground_truth_index) / (
                                                       overall_num_mentions_inkg * len(ground_truth_indices) * length)
                            del scores
                    if self.candidate_comparison_weight > 0.0:
                        candidate_embeddings = []
                        candidate_dict = defaultdict(set)
                        for c_idx, c in enumerate(mention.candidate_representations):
                            if c.complex_candidate.is_kg_candidate:
                                candidate_embeddings.append(c.candidate_embedding)
                                candidate_dict[c.complex_candidate.qid].add(c_idx)
                        candidate_embeddings = torch.stack(candidate_embeddings)

                        if self.use_cosine:
                            candidate_embeddings = normalize(candidate_embeddings, dim=1)
                        pairwise_scores = pairwise_linear_similarity(candidate_embeddings)
                        # if self.use_cosine:
                        #     pairwise_scores = pairwise_scores / 0.05
                        pairwise_scores_1d = pairwise_scores[torch.tril_indices(*pairwise_scores.size(), -1).unbind()]
                        margin = 0.85

                        b_loss += torch.sum(torch.clamp(pairwise_scores_1d - margin, min=0.0)) / (
                                    candidate_embeddings.size(0) * overall_num_mentions)

                    if self.type_embedding_comparison_weight > 0.0 and self.use_types:
                        for ground_truth_index in ground_truth_indices:
                            if mention.candidate_representations[ground_truth_index].complex_candidate.is_kg_candidate:
                                type_labels = mention.candidate_representations[
                                    ground_truth_index].kg_one_hop
                                type_tensor_prediction = mention.embedded_mention.processed_mention.type_prediction
                                t_loss += self.bce_loss(type_tensor_prediction, type_labels) / (
                                            overall_num_mentions_inkg * len(self.type_list))
                    for candidate in mention.candidate_representations:
                        candidate.similarity = None

        cs_loss *= self.embedding_comparison_weight
        t_loss *= self.type_embedding_comparison_weight
        b_loss *= self.candidate_comparison_weight

        loss_mention = torch.sum(torch.stack((cs_loss, el_loss, t_loss, entropy_loss, b_loss, mention_loss)))
        loss_mention /= self.gradient_accumulation_size
        if loss_mention.requires_grad:
            loss_mention.backward(retain_graph=mention_idx < maximum_num_mentions)
        return float(loss_mention), float(el_loss), float(mention_loss), float(cs_loss), float(t_loss), float(b_loss), float(entropy_loss)

    def train_epoch(
        self, epoch, training_dataloader, dev_dataloader, document_model, entity_model, ranking_model, mention_comparator_model, optimizer, scheduler, writer, current_stats: CurrentStats, step=0
    ):
        self.candidate_manager.reset_representations()
        pbar = tqdm(training_dataloader)
        loss_sum = 0
        el_sum = 0
        mention_loss_sum = 0
        cs_loss_sum = 0
        t_loss_sum = 0
        b_loss_sum = 0
        entropy_loss_sum = 0
        counter_overall = 0
        epoch_losses = []
        for idx, examples in enumerate(pbar):
            if idx < step:
                continue

            if (idx + 1) % 5000== 0:
                self.store_models(
                    Path(self.backup_name),
                    document_model, entity_model, ranking_model, mention_comparator_model, optimizer,
                    scheduler,
                    epoch=epoch,
                    step=idx
                )
            if (idx + 1) % 50000 == 0:
                # Evaluate the model on dev data
                evaluation_results, _ = self.evaluator.evaluate_model(
                    dev_dataloader,
                    document_model,
                    entity_model,
                    ranking_model,
                    mention_comparator_model,
                    self.mention_split,
                    1,
                    number_candidates=self.number_candidates_eval
                )
                wandb.log(evaluation_results)
                document_model.train()
                entity_model.train()
                ranking_model.train()
                self.evaluator.candidate_manager.reset_representations()
                print(json.dumps(evaluation_results, indent=4))
                identification_harmonic_mean = evaluation_results[
                    "identification_harmonic_mean"
                ]
                f_measure_in_kg_non_ookg = evaluation_results["f_measure_in_kg_non_ookg"]
                loss = evaluation_results["loss"]
                mention_loss = evaluation_results["mention_loss"]
                combined_loss = (loss + mention_loss)
                combined_measure = (
                                           identification_harmonic_mean + f_measure_in_kg_non_ookg
                                   ) / 2
                for key, value in evaluation_results.items():
                    if not isinstance(value, dict):
                        writer.add_scalar(
                            f"evaluation_results_{key}",
                            value,
                            epoch,
                        )
                if combined_loss <= current_stats.best_combined_loss:
                    self.store_models(
                        Path(
                            f"current_best_model{self.args['file_suffix']}"
                        ),
                        document_model, entity_model, ranking_model, mention_comparator_model, optimizer,
                        scheduler,
                        epoch=epoch,
                        loss=sum(epoch_losses) / len(epoch_losses),
                        average_losses=[sum(epoch_losses) / len(epoch_losses)],
                        evaluation_results=evaluation_results,
                    )
                    current_stats.best_dev_score = float(combined_measure)
                    current_stats.best_dev_el_fmeasure = float(f_measure_in_kg_non_ookg)
                    current_stats.best_dev_id_fmeasure = float(identification_harmonic_mean)
                    current_stats.best_loss = loss
                    current_stats.best_mention_loss = mention_loss
                    current_stats.best_combined_loss = combined_loss
                    current_stats.no_improvement = 0
                else:
                    if epoch >= 10:
                        current_stats.no_improvement += 1



            # If we link globally and use artificially created out_of_kg entities,
            # some of the entity mentions are chosen and set to out_of_kg
            if self.ookg_probability > 0.0:
                for example in examples:
                    for mention in example.mentions:
                        mention.be_artificial = False
                        if random.random() < self.ookg_probability:
                            mention.be_artificial = True

            cls_embeddings, processed_mentions = self.candidate_manager.embed_documents(
                self.device,
                examples,
                mention_comparator_model
            )

            overall_num_mentions = sum(len(x.mentions) for x in examples)
            maximum_num_mentions = max(len(x.mentions) for x in examples)

            mention_split = self.mention_split
            if mention_split < 0:
                mention_split = maximum_num_mentions


            overall_num_mentions_inkg = sum(len([y for y in x.mentions if not y.label_out_of_kg]) for x in examples)

            loss_sum_mention = 0

            already_embedded: List[List] = [[] for i in range(len(examples))]
            current_entity_representations: List[List] = [[] for i in range(len(examples))]
            current_embedded_mentions: List[List] = [[] for i in range(len(examples))]

            el_losses = 0.0
            mention_losses = 0.0
            cs_losses = 0.0
            t_losses = 0.0
            b_losses = 0.0
            entropy_losses = 0.0

            self.candidate_manager.global_test = []
            all_nested_candidates, num_candidates_with_mentions = get_nested_candidates(maximum_num_mentions, examples, processed_mentions,
                          mention_split, self.mentions_to_add_if_full, self.candidate_manager)

            if self.alternate_mention_embedding:
                num_candidates_with_mentions = overall_num_mentions_inkg
            # Subsample mentions but accumulate losses
            for nested_candidates, sub_sampled_embedded_mentions, mention_idx in all_nested_candidates:
                loss_mention, el_loss, mention_loss, cs_loss, t_loss, b_loss, entropy_loss = self.optimize(nested_candidates, mention_idx, sub_sampled_embedded_mentions, current_entity_representations, mention_split,
                                                                                                                        current_embedded_mentions, cls_embeddings, ranking_model, mention_comparator_model, already_embedded,
                                                                                                                        overall_num_mentions, overall_num_mentions_inkg, maximum_num_mentions, num_candidates_with_mentions)

                loss_sum_mention += loss_mention
                el_losses += el_loss
                mention_losses += mention_loss
                cs_losses += cs_loss
                t_losses += t_loss
                b_losses += b_loss
                entropy_losses += entropy_loss
                wandb.log({"loss": loss_mention, "el_loss": el_loss, "mention_loss": mention_loss, "cs_loss": cs_loss,
                           "t_loss": t_loss, "b_loss": b_loss, "entropy_loss": entropy_loss})

            counter_overall += 1
            loss_sum += float(loss_sum_mention)
            el_sum += float(el_losses)
            mention_loss_sum += float(mention_losses)
            cs_loss_sum += float(cs_losses)
            t_loss_sum += float(t_losses)
            b_loss_sum += float(b_losses)
            entropy_loss_sum += float(entropy_losses)
            pbar.set_description(
                f"Avg Losses {(loss_sum / counter_overall).__round__(4)},"
                f"[{float(el_sum / counter_overall).__round__(4)}, {float(cs_loss_sum / counter_overall).__round__(4)}, {float(t_loss_sum / counter_overall).__round__(4)}, "
                f"{float(entropy_loss_sum / counter_overall).__round__(4)}, {float(b_loss_sum / counter_overall).__round__(4)}, {float(mention_loss_sum / counter_overall).__round__(4)}]"
            )

            epoch_losses.append(float(loss_sum))
            if ((idx + 1) % self.gradient_accumulation_size == 0):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            del processed_mentions
            del already_embedded
            del cls_embeddings
            del current_entity_representations

        else:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        return sum(epoch_losses) / len(epoch_losses)

    def separate_and_calculate_mention_loss(self, mention: MentionContainerForProcessing,
                                            mention_comparator_model: MentionComparator):
        ground_truth_indices_mentions = []
        kg_candidates = []
        mention_candidates = []
        separate_mention_scores = []
        for idx, candidate in enumerate(mention.candidate_representations):
            if candidate.complex_candidate.is_kg_candidate:
                kg_candidates.append(candidate)
            else:
                if candidate.complex_candidate.qid == mention.embedded_mention.mention_container.label_qid:
                    ground_truth_indices_mentions.append(len(mention_candidates))
                mention_candidates.append(candidate)
                separate_mention_scores.append(candidate.similarity)
        if self.alternate_mention_embedding:

            x_1 = []
            x_2 = []
            y = []
            for candidate in mention_candidates:
                x_2.append(candidate.post_mention_embedding)
                y.append(torch.tensor(0, device=self.device) if candidate.complex_candidate.qid != mention.embedded_mention.mention_container.label_qid else torch.tensor(1, device=self.device))
            x_2 = torch.stack(x_2)
            y = torch.stack(y)
            mention_loss = pairwise_loss(mention.embedded_mention.post_mention_embedding, x_2, y)
        else:
            mention_loss_1 = torch.zeros((), device=self.device)

            if separate_mention_scores and self.use_cosine_similarity_in_comparator:
                separate_mention_scores = torch.stack(separate_mention_scores)
                separate_mention_scores /= TEMP
                for ground_truth_index in ground_truth_indices_mentions:
                    mention_loss_1 += self.ce_loss(separate_mention_scores, torch.tensor(ground_truth_index, device=self.device))
                mention_loss_1 /= len(ground_truth_indices_mentions) if len(ground_truth_indices_mentions) > 0  else 1
                mention_loss_1 *= self.embedding_comparison_weight

            mention_loss_2 = torch.zeros((), device=self.device)
            if mention_candidates:
                scores = mention_comparator_model(mention, mention_candidates)
                for ground_truth_index in ground_truth_indices_mentions:
                    mention_loss_2 += self.ce_loss(scores, torch.tensor(ground_truth_index, device=self.device))
                mention_loss_2 /= len(ground_truth_indices_mentions) if len(ground_truth_indices_mentions) > 0  else 1
            length = 1
            if self.use_sample_weight:
                length = len(self.candidate_manager.mention_document_index[
                                 mention.embedded_mention.mention_container.label_qid])
                length = length if length > 0 else 1
            mention_loss_1 /= length
            mention_loss_2 /= length
            mention_loss = mention_loss_1 + mention_loss_2
        mention.candidate_representations = kg_candidates
        return mention, mention_loss

    def calculate_el_entropy_loss(self, individual_weights, valid_probs, ce_labels, entropy_losses, overall_num_mentions, overall_num_mentions_inkg):
        loss = torch.zeros((), device=self.device)
        for probs_, ce_label, w in zip(valid_probs, ce_labels, individual_weights):
            loss += self.ce_loss(probs_.T, ce_label.unsqueeze(0)) * w
        loss /= (((overall_num_mentions - overall_num_mentions_inkg) if self.include_ookg_score else 0) +
                 overall_num_mentions_inkg)

        entropy_loss = torch.zeros((), device=self.device)
        for loss_ in entropy_losses:
            entropy_loss += loss_
        entropy_loss /= (overall_num_mentions - overall_num_mentions_inkg) if (
                                                                                      overall_num_mentions - overall_num_mentions_inkg) > 0 else 1

        return loss, entropy_loss

    def sequential(self,
                   ranking_model, mention_comparator_model,
                   embedded_documents: List[DocumentContainerForProcessing],
                   already_embedded: List[List[tuple]], overall_num_mentions: int, overall_num_mentions_inkg: int,
                   mention_split: int, num_candidates_with_mentions: int
                   ):
        individual_weights = []
        valid_probs = []
        ce_labels = []
        entropy_losses = []
        mention_loss_sum = torch.zeros((), device=self.device)

        for idx, (
            embedded_document,
            already_embedded_example
        ) in enumerate(zip(
            embedded_documents,
            already_embedded
        )):

            for mention in embedded_document.mentions:
                mention_loss = torch.zeros((), device=self.device)
                if self.separate_mention_training:
                    mention, mention_loss = self.separate_and_calculate_mention_loss(mention, mention_comparator_model)


                non_normalized_scores, concatenated_candidate_embeddings = ranking_model(
                    mention,
                    already_embedded_example
                )
                individual_weights_, valid_probs_, ce_labels_, entropy_losses_ = self.calculate_losses(mention.candidate_representations, concatenated_candidate_embeddings,
                                      mention, non_normalized_scores,
                                      ranking_model, mention_split, already_embedded_example)
                if mention_loss > 0:
                    mention_loss_sum += (mention_loss / (num_candidates_with_mentions if num_candidates_with_mentions > 0 else 1))

                individual_weights += individual_weights_
                valid_probs += valid_probs_
                ce_labels += ce_labels_
                entropy_losses += entropy_losses_
        loss, entropy_loss = self.calculate_el_entropy_loss(individual_weights, valid_probs, ce_labels, entropy_losses, overall_num_mentions, overall_num_mentions_inkg)
        return loss, entropy_loss, mention_loss_sum

    def calculate_losses(self, candidates: List[CandidateContainerForProcessing], concatenated_candidate_embeddings
                         , mention: MentionContainerForProcessing, non_normalized_scores, ranking_model, mention_split: int = 0,
                          already_embedded=None):

        ground_truth_indices = {i for i, x in enumerate(candidates) if
                                x == mention.embedded_mention.mention_container.label_qid}

        individual_weights = []
        valid_probs = []
        ce_labels = []
        entropy_losses = []

        if self.use_cosine and len(self.additional_entity_features) == 0 and not self.use_transe:
            non_normalized_scores /= TEMP
        weight = 1 / (len(ground_truth_indices) + (mention.embedded_mention.mention_container.label_out_of_kg or len(ground_truth_indices) == 0))
        if self.use_sample_weight:
            length = len(self.candidate_manager.mention_document_index[mention.embedded_mention.mention_container.label_qid])
            length = length if length > 0 else 1
            weight *= 1 / length
        if len(ground_truth_indices) > 0:
            for correct_index in ground_truth_indices:
                logits = non_normalized_scores
                ce_labels.append(
                    torch.tensor(
                        correct_index, device=self.device
                    )
                )
                if already_embedded is not None and candidates[correct_index].complex_candidate.is_kg_candidate:
                    if len(already_embedded) >= mention_split:
                        already_embedded.pop(0)
                    #TODO: Removed mention embedding from already_embedded. Add again if necessary
                    # Introduce noise
                    index_to_add = correct_index
                    if self.introduce_sequential_noise and random.random() < 1 / mention_split:
                        (max_elements, max_indices) = torch.max(logits, dim=0)
                        for item in max_indices:
                            if item not in ground_truth_indices:
                                index_to_add = item
                                break
                    already_embedded.append((candidates[index_to_add], None, None ))#logits[index_to_add]))

                valid_probs.append(logits)
                individual_weights.append(weight)

        elif self.include_ookg_score:
            ce_labels.append(
                torch.tensor(len(non_normalized_scores) - 1, device=self.device)
            )
            valid_probs.append(non_normalized_scores)
            individual_weights.append(weight)
        elif self.use_entropy_loss:
            normalized_scores = torch.softmax(non_normalized_scores, dim=0).squeeze(1)
            neg_entropy = torch.dot(normalized_scores, torch.log(normalized_scores))
            entropy_losses.append(neg_entropy)
        if mention.embedded_mention.mention_container.label_out_of_kg and ground_truth_indices and self.include_ookg_score:
            other_non_normalized_scores = non_normalized_scores

            idx_all = torch.ones(non_normalized_scores.size(0), dtype=torch.bool, device=self.device)
            idx_all[torch.tensor(ground_truth_indices, device=self.device, dtype=torch.int64)] = False
            idx_all[-1] = False

            tmp_scores = other_non_normalized_scores[idx_all]
            tmp_concatenated_candidate_embeddings = concatenated_candidate_embeddings[idx_all[:-1]]
            ookg_score = ranking_model.compute_ookg_representation(tmp_scores,
                                                                   tmp_concatenated_candidate_embeddings)
            tmp_scores = torch.cat((tmp_scores, ookg_score))
            logits = tmp_scores
            ce_labels.append(
                torch.tensor(len(logits) - 1, device=self.device)
            )
            valid_probs.append(logits)
            individual_weights.append(weight)
        return individual_weights, valid_probs, ce_labels, entropy_losses

    def rank(
        self,
        ranking_model,
        mention_comparator_model,
        embedded_documents: List[DocumentContainerForProcessing],
        already_embedded: List[List[tuple]],
        overall_num_mentions: int = 1,
        overall_num_mentions_inkg: int = 1,
        mention_split: int = 1,
        num_candidates_with_mentions: int = 1
    ):
        return self.sequential(ranking_model, mention_comparator_model, embedded_documents, already_embedded, overall_num_mentions,
                               overall_num_mentions_inkg, mention_split, num_candidates_with_mentions)

    def store_models(
        self,
        path: Path,
        document_model, entity_model, ranking_model, mention_comparator_model, optimizer,
        scheduler: LambdaLR,
        **other_info,
    ):
        torch.save(
            {
                "main_model": document_model.state_dict(),
                "entity_model": entity_model.state_dict(),
                "mention_comparator_model": mention_comparator_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": self.args,
                "ranking_model": ranking_model.state_dict(),
                **other_info,
            },
            path,
        )


def main():
    # torch.autograd.set_detect_anomaly(True)

    argparser = configargparse.ArgParser(
        default_config_files=["configs/default_config.yml"]
    )

    # Files
    argparser.add_argument("--config", is_config_file=True, help="config file path")
    argparser.add_argument("--training_dataset_path", type=Path, required=True, help="Training data file. Expected to be a json file as can be found in the datasets folder.")
    argparser.add_argument("--test_dataset_path", type=Path, default=None, help="Test data file. Expected to be a json file as can be found in the datasets folder.")
    argparser.add_argument("--dev_dataset_path", type=Path, default=None, help="Dev data file. Expected to be a json file as can be found in the datasets folder.")
    argparser.add_argument("--im_kg_path", type=Path, required=True, help="In memory KG file. Expected to be a jsonl file parsed as detailed in the README.md")
    argparser.add_argument("--type_list", type=Path, required=False, help="A JSON file with including all types that should be used for the type feature as a list.")
    argparser.add_argument("--two_hop_type_list", type=Path, required=False, help="A JSON file with including all types that should be used for the two-hop type feature as a list. Currently not in use.")
    argparser.add_argument("--transe_embeddings_file", type=Path, required=False, help="A numpy file containing all TransE embeddings necessary for the in memory KG.")
    argparser.add_argument("--transe_mappings_file", type=Path, required=False, help="A JSON file mapping each QID to a TransE embedding index as provided via argument transe_embeddings_file.")
    argparser.add_argument("--mention_dictionary_file", type=Path, required=True, help="A JSON file containing a mapping from each mention to a set of candidates.")
    argparser.add_argument("--load_model", type=Path, default=None, help="Path to an existing model dump which will be used for training.")

    # General model parameters
    argparser.add_argument("--batch_size", type=int, required=True, default=4)
    argparser.add_argument("--dim", type=int, required=True, default=300)
    argparser.add_argument("--num_epochs", type=int, required=True, default=30)
    argparser.add_argument("--number_candidates", type=int, required=True, default=30)
    argparser.add_argument("--store_every_n_epoch", type=int, required=True, default=50)
    argparser.add_argument("--evaluate_every_n_epoch", type=int, required=True, default=1)
    argparser.add_argument("--lr", type=float, required=True, default=3e-5, help="Learning rate used for all transformer models")
    argparser.add_argument("--head_lr", type=float, default=3e-3, help="Learning rate used for all heads and other models besides the transformers.")
    argparser.add_argument("--ranker_hidden_dim", type=int, help="Dimension of hidden layer in ranker. Currently not in use.")
    argparser.add_argument("--max_length", type=int, default=512, help="Maximum length (number of tokens) of documents. If documents are longer they are split.")

    # Architecture or feature changes
    argparser.add_argument("--use_popularity", type=is_true, default=False)
    argparser.add_argument("--use_types", type=is_true, default=False)
    argparser.add_argument("--hops", type=int, default=0)
    argparser.add_argument("--use_contextual_types", type=is_true, default=False, help="Types as a feature for the sequential model. Currently not in use.")
    argparser.add_argument("--include_ookg_score", type=is_true, default=True, help="Determines whether to calculate the ookg score or not.")
    argparser.add_argument("--adapter_models", type=is_true, default=True, help="Activates the use of adapters.")
    argparser.add_argument("--use_cosine", type=is_true, default=True, help="Determines whether to use cosine or dot product for comparing embeddings")
    argparser.add_argument("--use_context_embeddings", type=is_true, default=True, help="Determines whether to use the Encoder embeddings as feature for the ranker.")
    argparser.add_argument("--use_transe", type=is_true, default=False)
    argparser.add_argument("--use_edit_distance_in_comparator", type=is_true, default=False, help="Determines whether to use edit distance as feature in the mention comparison.")
    argparser.add_argument("--use_transe_embeddings_in_comparator", type=is_true, default=True, help="Determines whether to use transe as feature in the mention comparison.")
    argparser.add_argument("--use_cosine_similarity_in_comparator", type=is_true, default=True, help="Determines whether to use mention encoder  as feature in the mention comparison.")
    argparser.add_argument("--use_mean_in_type_inclusion", type=is_true, default=False, help="Deprecated.")
    argparser.add_argument("--alternate_mention_embedding", type=is_true, default=False, help="Alternative embedding method, combining TransE and Encoder embeddings and comparing them via pairwise distance loss.")
    argparser.add_argument("--alternative_description", type=str, default=None, help="Replace the Wikidata description with another.")
    argparser.add_argument("--max_length_description_encoder", type=int, default=32, help="Description embedding length.")

    # Clustering
    argparser.add_argument("--num_mentions_in_cluster", type=int, default=1, help="Number of mentions to which a mention in current batch is compared.")
    argparser.add_argument("--num_cluster_candidates", type=int, default=0, help="Deprecated.")
    argparser.add_argument("--window_size", type=int, default=6, help="Window size of TransE embedding inclusion.")

    # Others
    argparser.add_argument("--file_suffix", type=str, default="default", help="Name of the model.")
    argparser.add_argument("--_10sec", type=is_true, default=False)
    argparser.add_argument("--limited", type=is_true, default=False)

    # Training
    argparser.add_argument("--gradient_accumulation_size", type=int, default=1)
    argparser.add_argument("--num_weak_negatives", type=int, default=10, help="Number of weak negatives used during training.")
    argparser.add_argument("--num_weak_negatives_mentions", type=int, default=25, help="Number of weak negatives used during training for mentions.")
    argparser.add_argument(
        "--add_correct_candidate_during_training", type=is_true, default=True
    )
    argparser.add_argument("--ookg_probability", type=float, default=0.1, help="Probability that the ground-truth candidate is removed during training for a mention.")
    argparser.add_argument("--separate_mention_training", type=is_true, default=True, help="Determines whether to train the mention comparison and entity linking together.")
    argparser.add_argument("--mention_split", type=int, default=5, help="Number of mentions per example, for which the gradients are calculated at the same time.")
    argparser.add_argument("--number_candidates_eval", type=int, default=30)
    argparser.add_argument("--mentions_to_add_if_full", type=int, default=5, help="Number of mentions added as neighbors during training")
    argparser.add_argument("--zero_shot", type=is_true, default=False)
    argparser.add_argument("--mask_ratio", type=float, default=0.0, help="Option to mask the mention of examples")
    argparser.add_argument("--embedding_comparison_weight", type=float, default=1.0, help="Loss weight of pure mention-entity encoder comparison")
    argparser.add_argument("--type_embedding_comparison_weight", type=float, default=0.0, help="Loss weight of type prediction loss")
    argparser.add_argument("--use_sample_weight", type=is_true, default=False, help="Whether to weight the impact of each mentions loss depending on their frequency in the dataset")
    argparser.add_argument("--transformer_dropout", type=float, default=0.1)
    argparser.add_argument("--smoothed_type_labels", type=float, default=1.0)
    argparser.add_argument("--use_entropy_loss", type=is_true, default=False, help="Deprecated")
    argparser.add_argument("--candidate_comparison_weight", type=float, default=0.0, help="Deprecated")
    argparser.add_argument("--mix_other_mentions", type=is_true, default=True, help="Whether to mix the other mentions in document as negative candidates")
    argparser.add_argument("--introduce_sequential_noise", type=is_true, default=False, help="Deprecated")
    argparser.add_argument("--seed", type=int, default=0)

    arguments = vars(argparser.parse_args())

    wandb.init(project="ookg_train", config=arguments,
               name=f"run_{arguments['file_suffix']}")

    set_seeds(arguments["seed"])
    for key, value in arguments.items():
        print(f"{key}: {value}")
    trainer = Trainer(arguments)
    trainer.train()


if __name__ == "__main__":
    main()
