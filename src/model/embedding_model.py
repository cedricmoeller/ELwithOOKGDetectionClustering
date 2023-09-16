
import torch
from torch import nn
from torch.nn import Module, Linear
from transformers import BertModel

from src.model.entity_typing import EntityTyper
from src.utilities.constants import entity_embedding_dropout


# Necessary as else checkpointing won't work
class DummyModule(Module):
    def __init__(self, embedding_model: BertModel):
        super().__init__()
        self.embedding_model = embedding_model

    def forward(self, input_ids, attention_masks, dummy_tensor=None):
        return self.embedding_model(input_ids, attention_masks)


class EntityEmbeddingBert(Module):
    def __init__(self, bert_model: BertModel, dim: int = 64, num_types: int = 0, use_mean: bool = False, special_model=False):
        super().__init__()
        self.bert = bert_model
        self.mean_layer = Linear(384 if special_model else 768 , dim)
        self.dim = dim
        self.dropout = nn.Dropout(entity_embedding_dropout)


    def forward(self, input_ids, attention_masks, *x, **kwargs):
        if "head_mask" in kwargs and kwargs["head_mask"] is not None:
            kwargs["head_mask"] = torch.transpose(kwargs["head_mask"],0,1)

        embedded = self.bert(input_ids=input_ids, attention_mask=attention_masks, *x, **kwargs).last_hidden_state[:, 0]
        embedded = self.mean_layer(embedded)
        return embedded,



class BertDocumentEmbedder(Module):
    def __init__(self, embedding_model: BertModel, out_dim=64, num_types: int = 0, transformer_embedding_size=768):
        super().__init__()
        self.embedding_model = DummyModule(embedding_model)
        self.out_dim = out_dim
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.projection = Linear(transformer_embedding_size, out_dim)
        self.entity_typer = EntityTyper(out_dim, num_types)

    def forward(self, input_ids, attention_masks, start_embedding_positions, end_embedding_positions):

        embedded = self.embedding_model(input_ids, attention_masks).last_hidden_state
        entity_mention_embeddings = torch.stack(
            [torch.mean(embedded[idx, start_position: end_position + 1, :], dim=0) for
             idx, (start_example, end_example) in enumerate(zip(start_embedding_positions, end_embedding_positions)) for
             start_position, end_position in zip(start_example, end_example) if start_position != -1]
        )
        cls_embeddings = self.projection(embedded[:, 0, :])
        entity_mention_embeddings = self.projection(entity_mention_embeddings)
        entity_type_predictions = self.entity_typer(entity_mention_embeddings)

        return cls_embeddings, entity_mention_embeddings, entity_type_predictions