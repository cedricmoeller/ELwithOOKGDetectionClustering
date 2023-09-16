import bisect
import dataclasses
import math
from copy import deepcopy
from typing import List, Union, Optional

import torch


def from_tensor(elem: Optional[torch.Tensor]):
    if elem is not None:
        return elem.tolist()
    return None


def to_tensor(elem: Optional[list]):
    if elem is not None:
        return torch.tensor(elem)
    return None


@dataclasses.dataclass
class Mention:
    mention: str
    label_qid: str
    _label_out_of_kg: bool
    other_info: dict
    transe_embeddings: Optional[List[torch.Tensor]]
    transe_embedding: Optional[torch.Tensor]
    be_artificial: bool = False
    mention_counter: int = 0

    @classmethod
    def from_dict(cls, in_dict: dict):
        return cls(
            mention=in_dict["mention"],
            label_qid=in_dict["label_qid"],
            _label_out_of_kg=in_dict.get("_label_out_of_kg", in_dict.get("_label_emerging")),
            other_info=in_dict["other_info"],
            be_artificial=in_dict["be_artificial"],
            transe_embeddings=[to_tensor(x) for x in in_dict["transe_embeddings"]] if in_dict["transe_embeddings"] else None,
            transe_embedding=to_tensor(in_dict["transe_embeddings"])
        )

    def to_dict(self):
        return {
            "mention": self.mention,
            "label_qid": self.label_qid,
            "_label_out_of_kg": self._label_out_of_kg,
            "other_info": self.other_info,
            "be_artificial": self.be_artificial,
            "transe_embeddings": [from_tensor(x) for x in self.transe_embeddings] if self.transe_embeddings else None,
            "transe_embedding": from_tensor(self.transe_embedding)
        }

    @property
    def label_out_of_kg(self):
        return self._label_out_of_kg or self.be_artificial


@dataclasses.dataclass
class ProcessedMention:
    mention_embedding: torch.Tensor
    type_prediction: torch.Tensor

    @classmethod
    def from_dict(cls, in_dict: dict):
        return cls(
            mention_embedding = to_tensor(in_dict["mention_embedding"]),
            type_prediction = to_tensor(in_dict["type_prediction"]),
        )

    def to_dict(self):
        return {
            "mention_embedding": from_tensor(self.mention_embedding),
            "type_prediction": from_tensor(self.type_prediction),
        }


@dataclasses.dataclass
class EmbeddedMention:
    mention_container: Mention
    processed_mention: ProcessedMention
    post_mention_embedding: torch.Tensor

    @classmethod
    def from_dict(cls, in_dict: dict):
        return cls(
            mention_container=Mention.from_dict(in_dict["mention_container"]),
            processed_mention=ProcessedMention.from_dict(in_dict["processed_mention"]),
            post_mention_embedding=to_tensor(in_dict["post_mention_embedding"])
        )

    def to_dict(self):
        return {
            "mention_container": self.mention_container.to_dict(),
            "processed_mention": self.processed_mention.to_dict(),
            "post_mention_embedding": from_tensor(self.post_mention_embedding)
        }

@dataclasses.dataclass
class Example:
    input_ids: torch.Tensor
    attention_masks: torch.Tensor
    start_embedding_positions: torch.Tensor
    end_embedding_positions: torch.Tensor
    identifier: int
    mentions: List[Mention]

    @classmethod
    def from_dict(cls, in_dict: dict):
        return cls(
            input_ids=to_tensor(in_dict["input_ids"]),
            attention_masks=to_tensor(in_dict["attention_masks"]),
            start_embedding_positions=to_tensor(in_dict["start_embedding_positions"]),
            end_embedding_positions=to_tensor(in_dict["end_embedding_positions"]),
            identifier=in_dict["identifier"],
            mentions=[Mention.from_dict(x) for x in in_dict["mentions"]]
        )

    def to_dict(self):
        return {
            "input_ids": from_tensor(self.input_ids),
            "attention_masks": from_tensor(self.attention_masks),
            "start_embedding_positions": from_tensor(self.start_embedding_positions),
            "end_embedding_positions": from_tensor(self.end_embedding_positions),
            "identifier": self.identifier,
            "mentions": [x.to_dict() for x in self.mentions]
        }

@dataclasses.dataclass
class CurrentStats:
    best_dev_score: float = 0.0
    best_dev_el_fmeasure: float = 0.0
    best_dev_id_fmeasure: float = 0.0
    best_loss: float = math.inf
    best_mention_loss: float = math.inf
    best_combined_loss: float = math.inf
    no_improvement: int = 0


@dataclasses.dataclass
class MentionInClusterAbstract:
    cluster_identifier: Union[int, str]
    example: Optional[Example]
    mention_identifier: int
    mention_container: Mention
    global_type: Optional[torch.Tensor]

    @classmethod
    def from_dict(cls, in_dict: dict):
        return cls(
            cluster_identifier=in_dict["cluster_identifier"],
            example=in_dict["example"] if in_dict["example"] is None else Example.from_dict(in_dict["example"]),
            mention_identifier=in_dict["mention_identifier"],
            global_type=to_tensor(in_dict["global_type"]),
            mention_container=Mention.from_dict(in_dict["mention_container"])
        )

    def to_dict(self):
        return {
            "cluster_identifier": self.cluster_identifier,
            "example": self.example if self.example is None else self.example.to_dict(),
            "mention_identifier": self.mention_identifier,
            "global_type": from_tensor(self.global_type),
            "mention_container": self.mention_container.to_dict()
        }

    def __eq__(self, other):
        if isinstance(other, MentionInClusterAbstract):
            return other.mention_identifier == self.mention_identifier
        elif isinstance(other, int):
            return other == self.mention_identifier
        return False

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return str(self.mention_identifier)

    def __str__(self):
        return str(self.mention_identifier)

    def __hash__(self):
        return hash(self.mention_identifier)

@dataclasses.dataclass
class MentionInClusterToBeEmbedded(MentionInClusterAbstract):
    pass

@dataclasses.dataclass
class MentionInCluster(MentionInClusterAbstract):
    embedded_mention_container: EmbeddedMention
    non_normalized_score: float

    @classmethod
    def from_dict(cls, in_dict: dict):
        return cls(
            cluster_identifier=in_dict["cluster_identifier"],
            example=in_dict["example"] if in_dict["example"] is None else Example.from_dict(in_dict["example"]),
            mention_identifier=in_dict["mention_identifier"],
            global_type=to_tensor(in_dict["global_type"]),
            embedded_mention_container=EmbeddedMention.from_dict(in_dict["embedded_mention_container"]),
            non_normalized_score=in_dict["non_normalized_score"],
            mention_container=Mention.from_dict(in_dict["mention_container"])
        )

    def to_dict(self):
        return {
            "cluster_identifier": self.cluster_identifier,
            "example": self.example if self.example is None else self.example.to_dict(),
            "mention_identifier": self.mention_identifier,
            "global_type": from_tensor(self.global_type),
            "non_normalized_score": self.non_normalized_score,
            "embedded_mention_container": self.embedded_mention_container.to_dict(),
            "mention_container": self.mention_container.to_dict()
        }


@dataclasses.dataclass
class KGCandidateEntity:
    qid: str
    description: str
    label: str
    aliases: List[str]
    claims: list
    num_claims: int
    one_hop_types: list
    one_hop_types_tensor: torch.Tensor
    two_hop_types: list
    two_hop_types_tensor: Optional[torch.Tensor] = None
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor =  None
    degree: float = None
    other_info: dict = dataclasses.field(default_factory=dict)

    def to_dict(self, skip_unimportant: bool = False):
        return {
            "qid": self.qid,
            "description": self.description if not skip_unimportant else None,
            "label":self.label if not skip_unimportant else None,
            "aliases": self.aliases if not skip_unimportant else None,
            "claims":self.claims if not skip_unimportant else None,
            "num_claims":self.num_claims if not skip_unimportant else None,
            "one_hop_types":self.one_hop_types if not skip_unimportant else None,
            "one_hop_types_tensor": from_tensor(self.one_hop_types_tensor.to_dense()) if not skip_unimportant else None,
            "two_hop_types": self.two_hop_types if not skip_unimportant else None,
            "two_hop_types_tensor": from_tensor(self.two_hop_types_tensor) if not skip_unimportant else None,
            "input_ids": from_tensor(self.input_ids) if not skip_unimportant else None,
            "attention_mask": from_tensor(self.attention_mask) if not skip_unimportant else None,
            "degree": self.degree if not skip_unimportant else None,
            "other_info": self.other_info if not skip_unimportant else None,
        }

    @classmethod
    def from_dict(cls, in_dict: dict, skip_unimportant: bool = False):
        return KGCandidateEntity(
            qid=in_dict["qid"],
            description=in_dict["description"],
            label=in_dict["label"],
            aliases=in_dict["aliases"],
            claims=in_dict["claims"],
            num_claims=in_dict["num_claims"],
            one_hop_types=in_dict["one_hop_types"],
            one_hop_types_tensor= to_tensor(in_dict["one_hop_types_tensor"]),
            two_hop_types=in_dict["two_hop_types"],
            two_hop_types_tensor= to_tensor(in_dict["two_hop_types_tensor"]),
            input_ids= to_tensor(in_dict["input_ids"]),
            attention_mask=to_tensor(in_dict["attention_mask"]),
            degree= in_dict["degree"],
            other_info=in_dict["other_info"],
        )

    def __hash__(self):
        return hash(self.qid)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.qid
        if not isinstance(other, KGCandidateEntity):
            return False
        return other.qid == self.qid

    def lazy_copy(self):
        return KGCandidateEntity(self.qid,
                          self.description,
                          self.label,
                          self.aliases,
                          self.claims,
                        self.num_claims,
                          self.one_hop_types,
                          self.one_hop_types_tensor,
                          self.two_hop_types,
                          self.two_hop_types_tensor,
                          self.input_ids,
                          self.attention_mask,
                          self.degree,
                          self.other_info)


@dataclasses.dataclass
class ComplexCandidateEntity:
    candidate: Union[KGCandidateEntity, MentionInCluster, MentionInClusterToBeEmbedded] = None
    identified_in_cluster_comparison: bool = False

    @classmethod
    def from_dict(cls, in_dict):
        candidate_content = in_dict["candidate"]
        if in_dict["type"] == "KGCandidateEntity":
            candidate = KGCandidateEntity.from_dict(candidate_content)
        elif in_dict["type"] == "MentionInCluster":
            candidate = MentionInCluster.from_dict(candidate_content)
        else:
            candidate = MentionInClusterToBeEmbedded.from_dict(candidate_content)

        return ComplexCandidateEntity(
            candidate=candidate,
            identified_in_cluster_comparison=in_dict["identified_in_cluster_comparison"],
        )

    def to_dict(self, skip_unimportant: bool = False):
        if isinstance(self.candidate, KGCandidateEntity):
            type_id = "KGCandidateEntity"
            candidate = self.candidate.to_dict(skip_unimportant=skip_unimportant)
        elif isinstance(self.candidate, MentionInCluster):
            type_id = "MentionInCluster"
            candidate = self.candidate.to_dict()
        else:
            type_id = "MentionInClusterToBeEmbedded"
            candidate = self.candidate.to_dict()
        return {
            "type": type_id,
            "candidate": candidate,
            "identified_in_cluster_comparison": self.identified_in_cluster_comparison,
        }

    @property
    def is_kg_candidate(self):
        return isinstance(self.candidate, KGCandidateEntity)

    @property
    def is_mention_cluster_to_be_embedded(self):
        return isinstance(self.candidate, MentionInClusterToBeEmbedded)

    @property
    def qid(self):
        if isinstance(self.candidate, MentionInClusterAbstract):
            return self.candidate.cluster_identifier
        else:
            return self.candidate.qid

    def __hash__(self):
        return hash(self.qid)

    def __eq__(self, other):
        if isinstance(other, str):
            return other == self.qid
        if isinstance(other, int):
            return other == self.qid
        if not isinstance(other, ComplexCandidateEntity):
            return False
        return other.qid == self.qid


@dataclasses.dataclass
class CandidateContainerForProcessing:
    complex_candidate: ComplexCandidateEntity
    candidate_embedding: torch.Tensor
    kg_embedding: Optional[torch.Tensor]
    two_hop: Optional[torch.Tensor]
    additional_features: torch.Tensor
    kg_one_hop: Optional[torch.Tensor]
    post_mention_embedding: Optional[torch.Tensor] = None
    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, v)
        return result

    def __eq__(self, other):
        if isinstance(other, CandidateContainerForProcessing):
            return other.complex_candidate == self.complex_candidate
        return other == self.complex_candidate


class CandidateContainerWrapper(CandidateContainerForProcessing):
    def __init__(self, candidate_container_for_processing: CandidateContainerForProcessing,
                 similarity: torch.Tensor = None, edit_distance=None, type_distance=None):
        self.complex_candidate = candidate_container_for_processing.complex_candidate
        self.candidate_embedding = candidate_container_for_processing.candidate_embedding
        self.kg_embedding =candidate_container_for_processing.kg_embedding
        self.two_hop =candidate_container_for_processing.two_hop
        self.post_mention_embedding = candidate_container_for_processing.post_mention_embedding
        self.additional_features = candidate_container_for_processing.additional_features
        self.kg_one_hop = candidate_container_for_processing.kg_one_hop
        self.similarity = similarity
        self.edit_distance: float = edit_distance
        self.type_distance: float = type_distance

    @classmethod
    def from_dict(cls, in_dict):
        return CandidateContainerWrapper(
            CandidateContainerForProcessing(
            complex_candidate=ComplexCandidateEntity.from_dict(in_dict["complex_candidate"]),
            candidate_embedding=to_tensor(in_dict["candidate_embedding"]),
            kg_embedding=to_tensor(in_dict["kg_embedding"]),
            two_hop=to_tensor(in_dict["two_hop"]),
            additional_features=to_tensor(in_dict["additional_features"]),
            kg_one_hop=to_tensor(in_dict["kg_one_hop"])),
            similarity=to_tensor(in_dict["similarity"]),
            edit_distance=in_dict["edit_distance"],
            type_distance=to_tensor(in_dict["type_distance"])
        )

    def to_dict(self, skip_unimportant: bool = True):
        return {
            "complex_candidate": self.complex_candidate.to_dict(skip_unimportant=skip_unimportant),
            "candidate_embedding": from_tensor(self.candidate_embedding) if not skip_unimportant else None,
            "kg_embedding": from_tensor(self.kg_embedding) if not skip_unimportant else None,
            "two_hop": from_tensor(self.two_hop) if not skip_unimportant else None,
            "additional_features": from_tensor(self.additional_features) if not skip_unimportant else None,
            "kg_one_hop": from_tensor(self.kg_one_hop) if not skip_unimportant else None,
            "similarity": from_tensor(self.similarity),
            "edit_distance": self.edit_distance,
            "type_distance": from_tensor(self.type_distance)
        }




@dataclasses.dataclass
class MentionContainerForProcessing:
    embedded_mention: EmbeddedMention
    candidate_representations: List[CandidateContainerWrapper]

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in {"mention_embedding"}:
                setattr(result, k, deepcopy(v, memo))
            else:
                setattr(result, k, v)
        return result


@dataclasses.dataclass
class DocumentContainerForProcessing:
    cls_embedding: torch.Tensor
    mentions: List[MentionContainerForProcessing]

    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "cls_embedding":
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


@dataclasses.dataclass
class ClusterAssignment:
    assignment: int
    ground_truth: str

class ClusterRepresentation:
    def __init__(
        self,
        identifier: Union[str, int],
        embedded_mention_container: Union[MentionContainerForProcessing, List[MentionContainerForProcessing]],
        text: Union[Example, List[Example]],
        global_type: Union[List[torch.Tensor], torch.Tensor],
        non_normalized_score: Union[List[float], float],
        mention_identifier: Union[int, List[int]],
        num_mentions_in_cluster: int = 5
    ):
        self.identifier = identifier

        self.mentions: List[MentionInCluster] = []

        if isinstance(mention_identifier, list):
            assert len(embedded_mention_container) == len(text) == len(global_type) == len(non_normalized_score) == len(mention_identifier)
            for embedded_mention_container_, description, global_type_, non_normalized_score_, mention_identifier_ in zip(embedded_mention_container, text, global_type, non_normalized_score, mention_identifier):
                self.update(embedded_mention_container_, description, global_type_, non_normalized_score_,
                            mention_identifier_)

        else:
            self.update(embedded_mention_container, text, global_type, non_normalized_score,
                                                  mention_identifier)


        self.max_number_representations = num_mentions_in_cluster

    def get_bounded_list(self) -> List[MentionInCluster]:
        if self.max_number_representations < 0:
            return self.mentions
        return self.mentions[-self.max_number_representations:]

    @property
    def surface_forms(self):
        return [x.embedded_mention_container.mention_container.mention for x in self.get_bounded_list()]

    def get_mentions_in_cluster(self):
        return self.get_bounded_list()

    def remove_mention(self, mention_identifier: int):
        try:
            self.mentions.remove(mention_identifier)
        except ValueError:
            pass

    def update(self, embedded_mention_container_: MentionContainerForProcessing,
               text: Example,
               global_type: torch.Tensor,
               non_normalized_score: float,
               mention_identifier: int):

        idx_to_insert = bisect.bisect([mention.non_normalized_score for mention in self.mentions], non_normalized_score)
        self.mentions.insert(idx_to_insert, MentionInCluster(self.identifier, text, mention_identifier, embedded_mention_container_.embedded_mention.mention_container, global_type, embedded_mention_container_.embedded_mention , non_normalized_score,
                                                             ))

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "surface_forms": [mention.embedded_mention_container.mention_container.mention for mention in self.mentions],
        }

    def __repr__(self):
        return str(self.identifier)

    def __str__(self):
        return str(self.identifier)


@dataclasses.dataclass
class Result:
    link: Union[int, str, ComplexCandidateEntity]
    mention: EmbeddedMention
    is_ookg: bool
    non_normalized_scores: List[float]
    non_normalized_score: float
    action_prob: float
    context_transe_embeddings: List[torch.Tensor]
    post_mention_embedding: Optional[torch.Tensor]
    candidates: List[CandidateContainerWrapper]

    def to_dict(self):
        return {
            "link": self.link if isinstance(self.link, (int, str)) else self.link.to_dict(),
            "mention": self.mention.to_dict(),
            "is_ookg": self.is_ookg,
            "non_normalized_scores": self.non_normalized_scores,
            "non_normalized_score": self.non_normalized_score,
            "action_prob": self.action_prob,
            "candidates": [candidate.to_dict(True) for candidate in self.candidates],
            "post_mention_embedding": from_tensor(self.post_mention_embedding),
            "context_transe_embeddings": [from_tensor(x) for x in self.context_transe_embeddings]
        }

    @classmethod
    def from_dict(cls, in_dict: dict):
        return Result(
            link=in_dict["link"] if isinstance(in_dict["link"], (int, str)) else CandidateContainerWrapper.from_dict(in_dict["link"]),
            mention= EmbeddedMention.from_dict(in_dict["mention"]),
            is_ookg=in_dict["is_ookg"],
            non_normalized_scores=in_dict["non_normalized_scores"],
            non_normalized_score=in_dict["non_normalized_score"],
            action_prob=in_dict["action_prob"],
            post_mention_embedding=to_tensor(in_dict["post_mention_embedding"]),
            candidates= [CandidateContainerWrapper.from_dict(x) for x in in_dict["candidates"]],
            context_transe_embeddings=[to_tensor(x) for x in in_dict["context_transe_embeddings"]]
        )



