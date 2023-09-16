from enum import Enum


class EmbeddingMode(Enum):
    PROB = "PROB"
    POINT = "POINT"


TEMP = 0.05

class ActivatedLosses(Enum):
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    ENTITY_RELATION = "ENTITY_RELATION"


class GNNModules(Enum):
    GCN="GCN"
    GAT="GAT"
    GraphConv="GraphConv"


class AdditionalFeatures(Enum):
    NONE = "NONE"
    TYPE="TYPE"
    POPULARITY="POPULARITY"
    EDISTANCE = "EDISTANCE"
    PRIOR = "PRIOR"
    BOTH = "BOTH"


class DataSplitMode(Enum):
    KCROSS = "KCROSS"
    KCROSSTEST = "KCROSSTEST"
    TRAINVALTEST = "TRAINVALTEST"
    TRAINVALTESTR = "TRAINVALTESTR"


class FeaturesToSort(Enum):
    COS = "COS"
    POP="POP"
    EDIT = "EDIT"
    COSPOP = "COSPOP"
    COSPOPEDIT = "COSPOPEDIT"
    POPEDIT = "POPEDIT"
    COSEDIT = "COSEDIT"


entity_embedding_dropout = 0.1
feature_dropout = 0.05
