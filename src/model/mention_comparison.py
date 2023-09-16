from typing import List

import torch
from torch.nn import Linear, Module, Parameter
from torch.nn.utils.rnn import pad_sequence

from src.utilities.various_dataclasses import CandidateContainerWrapper, MentionContainerForProcessing, EmbeddedMention


class MentionComparator(Module):
    def __init__(self, use_edit_distance: bool, use_transe_embeddings: bool, use_cosine_similarity: bool, transe_mixing=False,
                 dim=300):
        super().__init__()
        self.use_edit_distance = use_edit_distance
        self.use_transe_embeddings = use_transe_embeddings
        self.use_cosine_similarity = use_cosine_similarity
        self.linear = Linear(use_cosine_similarity + use_edit_distance + use_transe_embeddings, 1)
        if transe_mixing:
            self.combine_layer = Linear(dim + 200, dim)
            self.importance = Linear(200, 1)
            self.mapping = Linear(dim, 100)
        self.dummy_param = Parameter(torch.empty(0))

    def combine_mentions_with_mention_embeddings(self, mention_embedding: torch.Tensor,
                                                 mention_embeddings_in_window: List[torch.Tensor],
                                                 transe_embedding_in_window: List[torch.Tensor]):
        return self.combine_mentions_with_mention_embeddings_special(mention_embedding, mention_embeddings_in_window,
                                                                transe_embedding_in_window)

    def combine_mentions_with_mention_embeddings_special(self, mention_embedding: torch.Tensor,
                                                 mention_embeddings_in_window: List[torch.Tensor],
                                                 transe_embedding_in_window: List[torch.Tensor]):
        if transe_embedding_in_window:
            other_mention_embeddings = torch.stack(mention_embeddings_in_window).to(self.dummy_param.device)
            other_mention_embeddings = self.mapping(other_mention_embeddings)
            mapped_mention_embedding = self.mapping(mention_embedding)
            dot_products = torch.matmul(mapped_mention_embedding, other_mention_embeddings.T)
            transe_embeddings = torch.stack(transe_embedding_in_window).to(self.dummy_param.device)
            importance_scores = torch.softmax(dot_products, dim=0)
            combined_transe = torch.sum(importance_scores.repeat((200, 1)).T * transe_embeddings, dim=0)
            return self.combine_layer(torch.cat((mention_embedding, combined_transe)))
        return mention_embedding


    def calculate_pairwise_scores(self, a: List[EmbeddedMention], b: List[EmbeddedMention], embedding_distances: torch.Tensor,
                                  edit_distances: torch.Tensor) -> torch.Tensor:
        # one has to be mention
        assert len(a) == len(b)
        if not a:
            return torch.zeros((0,1))

        concatenated_features = torch.zeros((len(a), 0), device=self.dummy_param.device)
        if self.use_cosine_similarity:
            concatenated_features = torch.cat((concatenated_features,
                                               embedding_distances.unsqueeze(1)), dim=1)

        if self.use_edit_distance:
            concatenated_features = torch.cat((concatenated_features,
                                               edit_distances.unsqueeze(1)), dim=1)

        if self.use_transe_embeddings:
            transe_scores = self.calculate_transe_scores_pairwise(a, b)
            concatenated_features = torch.cat((concatenated_features,
                                               transe_scores), dim=1)

        if self.use_edit_distance or self.use_transe_embeddings:
            return self.linear(concatenated_features).squeeze(1)
        else:
            return concatenated_features.squeeze(1)

    def calculate_transe_scores_pairwise(self, a: List[EmbeddedMention], b: List[EmbeddedMention]) -> torch.Tensor:
        scores = []
        for a_i, b_i in zip(a, b):
            if not a_i.mention_container.transe_embeddings or not b_i.mention_container.transe_embeddings:
                scores.append(torch.tensor([0.0], device=self.dummy_param.device))
                continue
            transe_a = torch.stack(
                a_i.mention_container.transe_embeddings).to(self.dummy_param.device)
            transe_b = torch.stack(
                b_i.mention_container.transe_embeddings).to(self.dummy_param.device)

            scores_for_candidate = torch.matmul(transe_a, transe_b.T)
            mean = torch.mean(scores_for_candidate)
            scores.append(torch.stack((mean, )))
        return torch.stack(scores)

    @staticmethod
    def chunk_it(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


    def calculate_transe_scores(self, mention_container: MentionContainerForProcessing,
                                candidate_containers: List[CandidateContainerWrapper]) -> torch.Tensor:
        if mention_container.embedded_mention.mention_container.transe_embeddings:
            m_transe_embeddings = torch.stack(mention_container.embedded_mention.mention_container.transe_embeddings)
        else:
            return torch.zeros((len(candidate_containers), 1), device=self.dummy_param.device)
        m_transe_embeddings = m_transe_embeddings.to(self.dummy_param.device)
        candidate_transe_embeddings = [torch.stack(c.complex_candidate.candidate.mention_container.transe_embeddings) if c.complex_candidate.candidate.mention_container.transe_embeddings else torch.empty((0, 200)) for c in candidate_containers]
        candidate_transe_embeddings = pad_sequence(candidate_transe_embeddings, batch_first=True, padding_value=torch.nan)
        candidate_transe_embeddings = candidate_transe_embeddings.to(self.dummy_param.device)

        scores_for_candidate = torch.matmul(candidate_transe_embeddings, m_transe_embeddings.T)
        scores_for_candidate = scores_for_candidate.view(scores_for_candidate.size(0), -1)
        nan_indices = torch.isnan(scores_for_candidate)
        all_nan = torch.all(nan_indices, dim=1)
        mask = (torch.logical_not(torch.isnan(scores_for_candidate))).float()
        scores_for_candidate[nan_indices] = 0.0
        means = scores_for_candidate.sum(1) / (mask.sum(1))
        scores_for_candidate[nan_indices] = -float("inf")
        means[all_nan] = 0
        scores = torch.stack((means,), dim=1)
        return scores


    def forward(self, mention_container: MentionContainerForProcessing,
                candidate_containers: List[CandidateContainerWrapper]):


        concatenated_features = torch.zeros((len(candidate_containers), 0), device=self.dummy_param.device)
        if self.use_cosine_similarity:
            embedding_distances = torch.stack([item.similarity for item in candidate_containers]).to(self.dummy_param.device)
            concatenated_features = torch.cat((concatenated_features,
                                               embedding_distances.unsqueeze(1)), dim=1)

        if self.use_edit_distance:
            concatenated_features = torch.cat((concatenated_features,
                                              torch.stack([torch.tensor(item.edit_distance, device=self.dummy_param.device) for item in candidate_containers], dim=0).unsqueeze(1)), dim=1)
        if self.use_transe_embeddings:
            concatenated_features = torch.cat((concatenated_features, self.calculate_transe_scores(mention_container, candidate_containers)), dim=1)

        return self.linear(concatenated_features).squeeze(1)
