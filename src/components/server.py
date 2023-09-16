from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List

from src.evaluation.evaluate import Evaluator
from src.model.initialization import init_for_evaluation


class Server:
    def __init__(self, args: dict):
        self.args, document_tokenizer, self.entity_tokenizer, kg_connector, type_list, document_model, entity_model, self.ranking_model, self.mention_comparator_model, candidate_manager, device = init_for_evaluation(
            args)

        document_model.eval()
        entity_model.eval()
        self.ranking_model.eval()
        candidate_manager.eval()
        candidate_manager.entity_model = entity_model
        candidate_manager.document_embedder = document_model

        self.evaluator = Evaluator(
            document_tokenizer,
            candidate_manager,
            device,
            kg_connector,
            args,
            num_types=len(type_list)
        )


    def link(self, texts: List[dict]):
        return self.evaluator.pipeline(texts, self.ranking_model, self.mention_comparator_model)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--model", type=Path, help="Path to model file to evaluate")
    argparser.add_argument("--im_kg_path", type=Path, help="In memory KG file. Expected to be a jsonl file parsed as detailed in the README.md")
    argparser.add_argument("--mention_dictionary_file", type=Path, required=False, help="A JSON file containing a mapping from each mention to a set of candidates.")
    argparser.add_argument("--type_list", type=Path, required=False, help="A JSON file with including all types that should be used for the type feature as a list.")
    argparser.add_argument("--two_hop_type_list", type=Path, required=False,
                           help="A JSON file with including all types that should be used for the type feature as a list.")
    argparser.add_argument("--transe_embeddings_file", type=Path, required=False,
                           help="A numpy file containing all TransE embeddings necessary for the in memory KG.")
    argparser.add_argument("--transe_mappings_file", type=Path, required=False,
                           help="A JSON file mapping each QID to a TransE embedding index as provided via argument transe_embeddings_file.")

    server = Server(vars(argparser.parse_args()))
    print(server.link([{"text":"Dude is crazy.", "entities": [{"offset":0, "length": 4}]}]))
