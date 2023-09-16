from copy import deepcopy
from pathlib import Path
from random import random
from typing import Tuple, List, Dict, Iterator

import jsonlines
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co, IterableDataset

from src.utilities.utils import preprocess_and_split_text
from src.utilities.utilities import mask_example

SPAN = Tuple[int, int]


class BaseEntity:
    def __init__(self, identifier: int, span: SPAN, out_of_kg: bool = False):
        self.identifier = identifier
        self.out_of_kg = out_of_kg
        self.examples = []


class Entity(BaseEntity):
    def __init__(self, identifier: int, span: SPAN, out_of_kg: bool = False):
        super(Entity, self).__init__(identifier, span, out_of_kg)
        self.span = span

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return other.identifier == self.identifier
        return False


class EfficientEntity(BaseEntity):
    def __init__(self, identifier: int, span: SPAN, out_of_kg: bool = False):
        super(EfficientEntity, self).__init__(identifier, span, out_of_kg)

    def __hash__(self):
        return hash(self.identifier)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return other.identifier == self.identifier
        return False


class BaseExample:
    def __init__(self, sentence: str, entities: List[BaseEntity]):
        self._num_out_of_kg_entities = 0

    @staticmethod
    def surface_forms() -> List[str]:
        raise NotImplementedError

    def dump(self) -> Dict:
        raise NotImplementedError

    def reset_out_of_kg_entities(self):
        self._num_out_of_kg_entities = 0

    @property
    def ratio_out_of_kg_entities(self):
        raise NotImplementedError

    def set_entity_to_emerge(self, entity: Entity):
        raise NotImplementedError

    @property
    def num_out_of_kg_entities(self):
        raise NotImplementedError

    def ids(self) -> List[int]:
        raise NotImplementedError

    @staticmethod
    def spans() -> List[SPAN]:
        raise NotImplementedError


class EfficientExample(BaseExample):
    def __init__(self, sentence: str, entities: List[EfficientEntity]):
        super(EfficientExample, self).__init__(sentence, entities)
        self.entities: List[EfficientEntity] = entities

    def dump(self) -> Dict:
        return {
            "entities": [
                {"identifier": entity.identifier, "out_of_kg": entity.out_of_kg}
                for entity in self.entities
            ]
        }

    @staticmethod
    def surface_forms() -> List[str]:
        return []

    @property
    def ratio_out_of_kg_entities(self):
        return self.num_out_of_kg_entities / len(self.entities)

    def set_entity_to_emerge(self, entity: Entity):
        if not entity.out_of_kg:
            self._num_out_of_kg_entities += 1

        entity.out_of_kg = True

    @property
    def num_out_of_kg_entities(self):
        return self._num_out_of_kg_entities

    def ids(self) -> List[int]:
        return [entity.identifier for entity in self.entities]

    @staticmethod
    def spans() -> List[SPAN]:
        return []


# class Example(BaseExample):
#     def __init__(self, sentence: str, entities: List[Entity], section_counter=None, section_span=None):
#         super(Example, self).__init__(sentence, entities)
#         self.sentence = sentence
#         self.entities: List[Entity] = entities
#         self.section_counter = section_counter
#         self.section_span = section_span
#
#     def surface_forms(self) -> List[str]:
#         return [
#             self.sentence[entity.span[0] : entity.span[1]] for entity in self.entities
#         ]
#
#     def dump(self) -> Dict:
#         return {
#             "entities": [
#                 {
#                     "identifier": entity.identifier,
#                     "out_of_kg": entity.out_of_kg,
#                     "start": entity.span[0],
#                     "end": entity.span[1],
#                 }
#                 for entity in self.entities
#             ],
#             "sentence": self.sentence,
#         }
#
#     @property
#     def ratio_out_of_kg_entities(self):
#         return self.num_out_of_kg_entities / len(self.entities)
#
#     def set_entity_to_emerge(self, entity: Entity):
#         if not entity.out_of_kg:
#             self._num_out_of_kg_entities += 1
#
#         entity.out_of_kg = True
#
#     @property
#     def num_out_of_kg_entities(self):
#         return self._num_out_of_kg_entities
#
#     def ids(self) -> List[int]:
#         return [entity.identifier for entity in self.entities]
#
#     def spans(self) -> List[SPAN]:
#         return [entity.span for entity in self.entities]


ENTITY_LIST = List[Tuple[int, bool, Tuple[int, int]]]


class OutOfKGEntityLinkingDataset(Dataset):
    def __init__(self, examples: list, mask_ratio: float, mask_token_id):
        self.examples = examples
        self.mask_ratio = mask_ratio
        self.mask_token_id = mask_token_id

    def __getitem__(self, index) -> T_co:
        example = self.examples[index]
        return mask_example(example, self.mask_ratio, self.mask_token_id)

    def __len__(self):
        return len(self.examples)


class LazyOutOfKGEntityLinkingDataset(IterableDataset):
    class ExampleIterator:
        def __init__(self, file_iterator: jsonlines.Reader, document_tokenizer, max_length, mask_ratio):
            self.file_iterator = file_iterator.__iter__()
            self._current_batch = []
            self.document_tokenizer = document_tokenizer
            self.max_length = max_length
            self.mask_ratio = mask_ratio
            self.examples_to_load = 100

        def __iter__(self):
            return self

        def __next__(self):
            if not self._current_batch:
                next_example = self.file_iterator.__next__()
                new_examples = preprocess_and_split_text([next_example], self.document_tokenizer, self.max_length, 0)
                new_examples = [mask_example(example, self.mask_ratio, self.document_tokenizer.mask_token_id) for example in new_examples]
                self._current_batch += new_examples

            return self._current_batch.pop(0)

    def __iter__(self) -> Iterator[T_co]:
        self.current_iterator = self.ExampleIterator(jsonlines.open(self.jsonlines_filepath), self.document_tokenizer, self.max_length, self.mask_ratio)
        return self.current_iterator

    def __len__(self):
        return 14255119


    # TODO: Already load the next batch while the first is running
    def __init__(self, jsonlines_filepath: Path, document_tokenizer, max_length=512, mask_ratio: float =0.0):
        if not jsonlines_filepath.exists():
            raise FileNotFoundError
        self.jsonlines_filepath = jsonlines_filepath
        self.document_tokenizer = document_tokenizer
        self.max_length = max_length
        self.mask_ratio = mask_ratio
        self.current_iterator = None

