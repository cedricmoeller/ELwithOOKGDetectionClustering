from typing import List, Tuple

import torch
from torch import cat, einsum
from torch.nn import Module, Linear, Dropout, Parameter, AdaptiveAvgPool1d, AdaptiveMaxPool1d
from torch.nn.functional import relu, softmax

from src.utilities.constants import feature_dropout
from src.utilities.various_dataclasses import MentionContainerForProcessing, CandidateContainerForProcessing


class SupervisedRankingModel(Module):
    def __init__(self, input_dim: int, hidden_dim: int, additional_features: int = 0, include_ookg_score: bool = False,
                 use_types: bool = False, num_global_types: int = 1, use_contextual_types: bool = False,
                 use_transe: bool = False, use_cosine=False, use_context_embeddings=True):

        super().__init__()
        self.single_layer = True
        self.dropout = Dropout(feature_dropout)
        if self.single_layer:
            self.linear_1 = Linear(
                int(use_context_embeddings) + int(use_contextual_types or use_transe) + additional_features + int(
                    use_types), 1)
        else:
            self.linear_1 = Linear(
                int(use_context_embeddings) + int(use_contextual_types or use_transe) + additional_features + int(
                    use_types), hidden_dim)
        self.linear_2 = Linear(hidden_dim, 1)
        self.linear_3 = Linear(
            (int(use_context_embeddings) + int(use_contextual_types or use_transe) + additional_features) * 3, 1)
        self.linear_4 = Linear((int(use_context_embeddings) + int(
            use_contextual_types or use_transe) + additional_features + int(use_types)), 1)
        self.avg_pool = AdaptiveAvgPool1d(1)
        self.max_pool = AdaptiveMaxPool1d(1)
        self.rel_score_matrix = Parameter(torch.diag(torch.randn(input_dim)))
        self.contextual_embedding_matrix = Parameter(torch.diag(torch.randn(input_dim)))

        self.use_cosine = use_cosine

        self.use_contextual_types = use_contextual_types
        self.use_context_embeddings = use_context_embeddings
        self.use_transe = use_transe
        self.use_types = use_types
        self.include_ookg_score = include_ookg_score

        self.additional_features = additional_features

        self.one = torch.ones((1, 1))
        self.register_buffer("one_const", self.one) # Access to device of model



    def get_contextual_scores(
        self, candidate_embeddings: torch.Tensor, already_embedded: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        :param candidate_embeddings: Candidate embeddings of current mention
        :param already_embedded: Embeddings of past actions
        :return: Returns contextual scores including information of past actions
        """
        if already_embedded:
            already_embedded = torch.stack(already_embedded)
            values, _ = torch.max(
                einsum(
                    "ij,jk,lk->il",
                    candidate_embeddings,
                    self.rel_score_matrix,
                    already_embedded,
                ),
                dim=0,
            )
            attentions = softmax(values, dim=0)
            contextual_scores = einsum(
                "i,ij,jk,lk->l",
                attentions,
                already_embedded,
                self.contextual_embedding_matrix,
                candidate_embeddings,
            )
            contextual_scores = contextual_scores.unsqueeze(1)
        else:
            contextual_scores = torch.zeros(
                (candidate_embeddings.size()[0], 1), device=self.one_const.device
            )
        return contextual_scores

    def compute_ookg_representation(self, scores: torch.Tensor, concatenated_candidate_embeddings: torch.Tensor):
        if self.training:
            attention_scores = torch.softmax(scores, dim=0)
            best = torch.einsum("ij,ik->jk",attention_scores,concatenated_candidate_embeddings)
        else:
            arg = torch.argmax(scores)
            best = concatenated_candidate_embeddings[arg].unsqueeze(0)
        return self.linear_4(best)


    def calculate_type_based_context_scores(self,candidate_representations: List[CandidateContainerForProcessing],
                                            already_embedded: List[Tuple[CandidateContainerForProcessing, torch.Tensor]]):
        if already_embedded:
            scores = []
            for candidate in candidate_representations:
                sum_scores = torch.tensor(0.0, device=self.one_const.device)
                for x in already_embedded:
                    sum_scores += torch.sum(torch.logical_and(x[0].two_hop, candidate.two_hop)) / (torch.sum(x[0].two_hop)) if torch.any(x[0].two_hop) else torch.tensor(1.0, device=self.one_const.device)
                sum_scores /= len(already_embedded)
                scores.append(sum_scores)
            contextual_scores = torch.stack(scores)

            contextual_scores = contextual_scores.unsqueeze(1)
        else:
            contextual_scores = torch.zeros(
                (len(candidate_representations), 1), device=self.one_const.device
            )
        return contextual_scores

    def calculate_transe_based_context_scores(self, mention_embedding: torch.Tensor,
                                            candidate_representations: List[CandidateContainerForProcessing],
                                            already_embedded: List[Tuple[CandidateContainerForProcessing, torch.Tensor]]):
        already_embedded_transe = [x[0].kg_embedding for x in already_embedded if x[0].kg_embedding is not None]
        if already_embedded_transe:
            candidate_transe = torch.stack([candidate.kg_embedding if candidate.kg_embedding is not None else torch.zeros(200, device=self.one_const.device) for candidate in candidate_representations])
            already_embedded_transe = torch.stack(already_embedded_transe)
            dot_products = torch.matmul(candidate_transe, already_embedded_transe.T)
            dot_products = dot_products
            contextual_scores = torch.mean(dot_products, dim=1)

            contextual_scores = contextual_scores.unsqueeze(1)
        else:
            contextual_scores = torch.zeros(
                (len(candidate_representations), 1), device=self.one_const.device
            )
        return contextual_scores

    def calculate_pairwise_scores(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # one has to be mention
        if a.size(0) == 0:
            return torch.zeros((0,1))

        if self.use_cosine:
            return torch.cosine_similarity(a, b)
        else:
            return torch.einsum("ij,ij->i", a, b)


    def calculate_scores(self, mention_container: MentionContainerForProcessing,
                         candidate_scores: torch.Tensor,
                         already_embedded: List[Tuple[CandidateContainerForProcessing, torch.Tensor]]):
        i_m = mention_container
        if not i_m.candidate_representations:
            return torch.ones((1,1), device=self.one_const.device), \
                   torch.ones((1,0), device=self.one_const.device)

        concatenated_candidate_embeddings = torch.zeros((len(i_m.candidate_representations), 0), device=self.one_const.device)
        if self.use_context_embeddings:
            concatenated_candidate_embeddings = torch.cat((concatenated_candidate_embeddings,
                                               candidate_scores), dim=1)

        if self.use_types:
            type_scores = torch.tensor([candidate.type_distance for candidate in mention_container.candidate_representations],
                                       device=self.one_const.device).unsqueeze(1)
            concatenated_candidate_embeddings = torch.cat(
                (concatenated_candidate_embeddings, type_scores), dim=1
            )

        assert not (self.use_contextual_types and self.use_transe)
        if self.use_contextual_types:
            contextual_scores = self.calculate_type_based_context_scores(mention_container.candidate_representations,
                                                                           already_embedded)
            concatenated_candidate_embeddings = torch.cat(
                (concatenated_candidate_embeddings, contextual_scores), dim=1
            )

        elif self.use_transe:
            contextual_scores = self.calculate_transe_based_context_scores(mention_container.embedded_mention.processed_mention.mention_embedding,
                                                                           mention_container.candidate_representations,
                                                                           already_embedded)
            concatenated_candidate_embeddings = torch.cat(
                (concatenated_candidate_embeddings, contextual_scores), dim=1
            )

        additional_mention_features = torch.stack([c.additional_features for c in mention_container.candidate_representations])
        concatenated_candidate_embeddings = torch.cat(
            (concatenated_candidate_embeddings, additional_mention_features), dim=1
        )

        h = concatenated_candidate_embeddings
        if concatenated_candidate_embeddings.size(1) > 1 or not self.use_context_embeddings:
            h = self.linear_1(concatenated_candidate_embeddings)

        if not self.single_layer:
            h  = self.dropout(relu(h))
            h  = self.linear_2(h)

        if self.include_ookg_score:
            h_out_of_kg = self.compute_ookg_representation(h, concatenated_candidate_embeddings)
            non_normalized_vector = cat([h, h_out_of_kg], dim=0)
        else:
            non_normalized_vector = h

        return non_normalized_vector, concatenated_candidate_embeddings

    def forward(
        self, mention_container: MentionContainerForProcessing,
            already_embedded: List[Tuple[CandidateContainerForProcessing, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param mention_container: Current mention container
        :param already_embedded: all past linking decisions
        :return: Returns actions, the log probs representing the candidate actions and order actions, the order of mentions
        """

        candidate_scores = torch.stack(
                [c.similarity for c in mention_container.candidate_representations]).unsqueeze(
                1) if mention_container.candidate_representations else torch.empty((0, 1), device=self.one.device)
        scores, concatenated_candidate_embeddings = self.calculate_scores(mention_container,
                                          candidate_scores, already_embedded)


        return scores, concatenated_candidate_embeddings