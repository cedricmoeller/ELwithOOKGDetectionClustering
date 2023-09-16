from pathlib import Path
from typing import Any, Union

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from src.utilities.dataset_classes import (
    OutOfKGEntityLinkingDataset,
    LazyOutOfKGEntityLinkingDataset,
)
from src.utilities.utils import transform_item, preprocess_and_split_texts, get_encodings_and_entity_positions


def instantiate_online_dataset(dataset_path: Path, document_tokenizer: RobertaTokenizerFast, max_length: int,
                               mask_ratio: float):
    assert max_length <= 512
    max_length -= 2
    return LazyOutOfKGEntityLinkingDataset(dataset_path, document_tokenizer, max_length, mask_ratio)

def instantiate_dataset(dataset, kg_connector, transe_mappings, transe_embeddings, window_size: int, document_tokenizer: RobertaTokenizerFast=None, max_length = 50, mask_ratio: float = 0.0,
                        ):
    data = []
    if document_tokenizer:
        # Special tokens
        assert max_length <= 512
        max_length -= 2
        data = preprocess_and_split_texts(dataset, document_tokenizer, max_length, kg_connector, transe_mappings, transe_embeddings, window_size)
    else:
        for x in tqdm(dataset):
            transformed = transform_item(x)
            (input_ids_to_use, attention_masks_to_use, start_embedding_positions_to_use,
             end_embedding_positions_to_use) = get_encodings_and_entity_positions(
                document_tokenizer, [transformed[0]])

            assert start_embedding_positions_to_use[0].size(0) == end_embedding_positions_to_use[0].size(0)
            data.append((((input_ids_to_use, attention_masks_to_use, start_embedding_positions_to_use[0],
                                end_embedding_positions_to_use[0]), *transformed[1:]), x))

    return OutOfKGEntityLinkingDataset(data, mask_ratio=mask_ratio, mask_token_id=document_tokenizer.mask_token_id)


def instantiate_dataloader(dataset: Union[Path, list], kg_connector, transe_mappings, transe_embeddings, window_size: int, document_tokenizer: RobertaTokenizerFast=None, max_length = 50,
                           mask_ratio: float = 0.0, **dataloader_args):
    def _collate_fn(l: list) -> Any:
        return l

    if isinstance(dataset, Path):
        instantiated_dataset = instantiate_online_dataset(Path(dataset), document_tokenizer, max_length=max_length, mask_ratio=mask_ratio)
        if "shuffle" in dataloader_args:
            del dataloader_args["shuffle"]
    else:
        instantiated_dataset = instantiate_dataset(dataset, kg_connector, transe_mappings, transe_embeddings, window_size, document_tokenizer, max_length=max_length, mask_ratio=mask_ratio,
                                                   )

    return DataLoader(instantiated_dataset, collate_fn=_collate_fn, **dataloader_args,
                num_workers=1 if isinstance(dataset, Path) else 0, prefetch_factor= 8 if isinstance(dataset, Path) else 2)