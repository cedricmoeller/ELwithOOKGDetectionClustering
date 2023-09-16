import copy
import math
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from src.utilities.special_tokens import ENTITY_START, ENTITY_END
from src.utilities.various_dataclasses import Mention, Example


def transform_item(item, lower_text: bool = False):
    entities = item["entities"]
    text = item["text"].lower() if lower_text else item["text"]
    sorted_entities = sorted(entities, key=lambda x: x["offset"], reverse=False)
    entities = []
    split_text = []
    offset = 0
    for idx, entity in enumerate(sorted_entities):
        entity_end = entity["offset"] + entity["length"]
        mention = text[entity["offset"]: entity_end]
        split_text.append(text[offset:entity_end])
        split_text.append(f"{ENTITY_START}{mention}{ENTITY_END}")
        offset = entity_end
        entity["mention"] = mention
        entities.append(entity)
    split_text.append(text[offset:])

    return split_text, [item["mention"] for item in entities], [(item.get("qid", None), item.get("out_of_kg", None)) for item in
                                                                entities], entities


def preprocess_and_split_texts(texts: list, document_tokenizer: RobertaTokenizerFast, max_length: int, kg_connector,
                               transe_mappings, transe_embeddings, window_size: int,
                               batch_size = 1000
                              ):
    data = []
    batch = []
    pbar = tqdm(total=len(texts))
    mention_counter = 0
    for x in texts:
        batch.append(transform_item(x))
        if len(batch) > batch_size:
            new_data, mention_counter= preprocess_and_split_text(batch, document_tokenizer, max_length, len(data),
                                              kg_connector, transe_mappings, transe_embeddings, window_size,
                                              mention_counter)
            data += new_data
            pbar.update(len(batch))
            batch = []
    if batch:
        new_data, mention_counter= preprocess_and_split_text(batch, document_tokenizer, max_length, len(data), kg_connector,
                                          transe_mappings, transe_embeddings, window_size,
                                          mention_counter)
        data += new_data
        pbar.update(len(batch))
    return data
def preprocess_and_split_text(batch: list, document_tokenizer: RobertaTokenizerFast, max_length: int,
                              example_offset: int, kg_connector, transe_mappings, transe_embeddings,
                              window_size, mention_counter: int):


    (input_ids_all,
     attention_mask_all,
     start_embedding_positions_all, end_embedding_positions_all, offset_mappings_all) = get_encodings_and_entity_positions(document_tokenizer,
                                                                                              [x[0] for x in batch],
                                                                                              truncation=False,
                                                                                              add_special_tokens=False,
                                                                                              return_offsets_mapping=True)

    new_texts = []
    counter = 0
    for input_ids, attention_mask, start_embedding_positions, end_embedding_positions, offset_mappings, original_text in zip(input_ids_all, attention_mask_all, start_embedding_positions_all, end_embedding_positions_all, offset_mappings_all, batch):
        mention_start_positions = {}
        for idx, (start_position, end_position) in enumerate(zip(start_embedding_positions,end_embedding_positions)):
            mention_start_positions[int(start_position)] = (idx, int(start_position), int(end_position))

        input_ids = input_ids.squeeze().tolist()
        attention_mask = attention_mask.squeeze().tolist()
        try:
            first_padding_idx = input_ids.index(1)
            input_ids = input_ids[:first_padding_idx]
            attention_mask = attention_mask[:first_padding_idx]
        except ValueError:
            pass
        offset_mappings = offset_mappings.squeeze().tolist()

        split_token_sequences = defaultdict(list)
        token_sequence_idx = 0
        for idx in range(len(input_ids)):
            if idx in mention_start_positions:
                end_position = mention_start_positions[idx][2]
                if end_position - idx + 1 + len(split_token_sequences[token_sequence_idx]) > max_length:
                    token_sequence_idx += 1
            split_token_sequences[token_sequence_idx].append(idx)
            if len(split_token_sequences[token_sequence_idx]) == max_length:
                token_sequence_idx += 1

        offset = 0
        for y in split_token_sequences.values():
            mentions = []
            labels = []
            separate_entities = []

            entities = []
            input_ids_to_use = []
            attention_masks_to_use = []
            start_embedding_positions_to_use = []
            end_embedding_positions_to_use = []
            for token_idx in y:
                input_ids_to_use.append(input_ids[token_idx])
                attention_masks_to_use.append(attention_mask[token_idx])
                if token_idx in mention_start_positions:
                    idx, start_position, end_position = mention_start_positions[token_idx]
                    mentions.append(original_text[1][idx])
                    labels.append(original_text[2][idx])
                    tmp = original_text[3][idx]
                    tmp["offset"] = math.nan
                    separate_entities.append(tmp)
                    entity = copy.deepcopy(original_text[3][idx])
                    entity["offset"] =  math.nan
                    start_embedding_positions_to_use.append(start_position - offset + 1)
                    end_embedding_positions_to_use.append(end_position - offset + 1)
                    entities.append(entity)
            offset += len(y)
            if entities:
                input_ids_to_use.append(2)
                attention_masks_to_use.append(1)
                input_ids_to_use.insert(0, 0)
                attention_masks_to_use.append(1)
                input_ids_to_use = torch.tensor(input_ids_to_use).unsqueeze(0)
                attention_masks_to_use = torch.tensor(attention_masks_to_use).unsqueeze(0)
                start_embedding_positions_to_use = torch.tensor(start_embedding_positions_to_use).unsqueeze(1)
                end_embedding_positions_to_use = torch.tensor(end_embedding_positions_to_use).unsqueeze(1)

                assert start_embedding_positions_to_use.size(0) == end_embedding_positions_to_use.size(0)

                final_mentions_sub = []
                for idx, (mention, label, other_info__) in enumerate(zip(mentions, labels, separate_entities)):
                    other_entities_before = [x for idx_, x in enumerate(labels) if
                                             x[0] != label[0] and not x[1] and idx_ < idx]
                    other_entities_after = [x for idx_, x in enumerate(labels) if
                                             x[0] != label[0] and not x[1] and idx_ > idx]

                    context_entities = get_context_entities(other_entities_before,other_entities_after, window_size)
                    transe_embeddings_of_mention = []
                    for other_entity in [kg_connector.get_entity(x[0]) for x
                                         in
                                         context_entities if not isinstance(kg_connector.get_entity(x[0]), str)]:
                        if other_entity.qid in transe_mappings:
                            transe_embeddings_of_mention.append(transe_embeddings[transe_mappings[other_entity.qid]])
                    if label[0] in transe_mappings:
                        transe_embedding = transe_embeddings[transe_mappings[label[0]]]
                    else:
                        transe_embedding = None
                    final_mentions_sub.append(Mention(mention, label[0], label[1], other_info__, transe_embeddings_of_mention,
                                                      transe_embedding=transe_embedding,
                                                      mention_counter=mention_counter))
                    mention_counter += 1
                new_texts.append(Example(input_ids_to_use, attention_masks_to_use, start_embedding_positions_to_use, end_embedding_positions_to_use, example_offset + counter, final_mentions_sub))
                counter += 1

    return new_texts, mention_counter


def get_context_entities(l_before: list, l_after: list, window_size: int):
    context_entities = []
    while len(context_entities) < window_size and (l_before or l_after):
        if l_before:
            context_entities.append(l_before.pop())
        if l_after:
            context_entities.append(l_after.pop(0))
    return context_entities


def get_encodings_and_entity_positions(
    tokenizer, texts, max_length=512, truncation=True, add_special_tokens=True, return_offsets_mapping=False
):
    start_special_token = tokenizer.get_vocab()[ENTITY_START]
    end_special_token = tokenizer.get_vocab()[ENTITY_END]

    encoded = tokenizer(
        list(texts),
        padding=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=return_offsets_mapping,
        is_split_into_words=True
    )
    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]

    start_embedding_positions = [
        (input_ids[i, :] == start_special_token).nonzero() for i in range(input_ids.size(0))
    ]
    end_embedding_positions = [
        (input_ids[i, :] == end_special_token).nonzero() for i in range(input_ids.size(0))
    ]

    new_start_embedding_positions = []
    new_end_embedding_positions = []
    new_input_ids = []
    new_attention_masks = []

    for idx, (start_embedding_positions_, end_embedding_positions_) in enumerate(zip(start_embedding_positions, end_embedding_positions)):
        indices = torch.ones(input_ids[idx,:].size(), dtype=torch.int64)
        offset = 0
        new_start_embedding_positions_ = []
        new_end_embedding_positions_ = []

        for idx_, (start_embedding_position, end_embedding_position) in enumerate(reversed(list(zip(start_embedding_positions_, end_embedding_positions_)))):
            indices[start_embedding_position] = 0
            indices[end_embedding_position] = 0
            new_start_embedding_positions_.append(start_embedding_position - 2 * (len(start_embedding_positions_) - 1 - idx_))
            new_end_embedding_positions_.append(end_embedding_position - 2 * (len(start_embedding_positions_) - idx_ - 1) - 2)
        new_start_embedding_positions_ = list(reversed(new_start_embedding_positions_))
        new_end_embedding_positions_ = list(reversed(new_end_embedding_positions_))
        new_input_ids_ = input_ids[idx, torch.nonzero(indices)].T
        new_attention_masks_ = attention_masks[idx, torch.nonzero(indices)].T
        new_input_ids.append(new_input_ids_)
        for a, b, c, d in zip(new_start_embedding_positions_, new_end_embedding_positions_, start_embedding_positions_,  end_embedding_positions_):
            assert torch.all(new_input_ids_[0, a:b + 1] == input_ids[idx,c + 1: d])
        new_attention_masks.append(new_attention_masks_)
        new_start_embedding_positions.append(torch.tensor(new_start_embedding_positions_))
        new_end_embedding_positions.append(torch.tensor(new_end_embedding_positions_))
    start_embedding_positions = new_start_embedding_positions
    end_embedding_positions = new_end_embedding_positions
    input_ids = new_input_ids
    attention_masks = new_attention_masks


    if return_offsets_mapping:
        return input_ids, attention_masks, start_embedding_positions, end_embedding_positions, encoded["offset_mapping"]
    else:
        return input_ids, attention_masks, start_embedding_positions, end_embedding_positions