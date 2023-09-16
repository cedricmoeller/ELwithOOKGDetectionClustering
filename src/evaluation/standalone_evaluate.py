import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Any

from src.utilities.dataloaders import instantiate_dataloader
from src.evaluation.evaluate import Evaluator
from src.model.initialization import init_for_evaluation
from src.utilities.constants import FeaturesToSort
import wandb
from src.utilities.utilities import is_true


def process_model(args: Dict[Any, Any]):
    debug = args.get("debug", False)
    if debug:
        args["_10sec"] = True
    args, document_tokenizer, entity_tokenizer, kg_connector, type_list, document_model, entity_model, ranking_model, mention_comparator_model, candidate_manager, device = init_for_evaluation(
        args, debug=debug, swap=False)

    batch_size = args.get("batch_size")

    dataset = json.load(args["dataset"].open())
    if isinstance(dataset, dict):
        dataset = dataset["test"]
    elif isinstance(dataset[0], list):
        dataset = dataset[0][0]
    test_data_loader = instantiate_dataloader(dataset, kg_connector, candidate_manager.transe_mappings,
                                              candidate_manager.transe_embeddings,
                                              args.get("window_size", 6),
                                              document_tokenizer=document_tokenizer,
                                              max_length=args.get("max_length", 50),
                                              **{**{}, "batch_size": batch_size, "shuffle": args.get("shuffle", False)})
    evaluator = Evaluator(
        document_tokenizer,
        candidate_manager,
        device,
        kg_connector,
        args,
        num_types=len(type_list)
    )
    result, detailed_results = evaluator.evaluate_model(test_data_loader, document_model, entity_model, ranking_model,
                                                        mention_comparator_model,
                                                        args.get("mention_split", 4),
                                                        args.get("mentions_to_add_if_full", 1),
                                                        second_stage_clustering=args.get("second_stage_clustering"),
                                                        detailed_info=True,
                                                        spoof_perfect_linking=args.get("spoof_perfect_linking", False),
                                                        add_correct_candidate=args.get("add_correct_candidate", False))
    wandb.summary.update(result)
    json.dump({**result, **detailed_results}, open(f"detailed_results{args['suffix']}.json", "w"), indent=4)
    # for key, value in result.items():
    #     writer.add_scalar(
    #         f"evaluation_results_{key}", value
    #     )
    # writer.flush()
    # writer.close()
    if "detailed_info" in result:
        del result["detailed_info"]
    print(json.dumps(result, indent=4))


def main(args: Dict[Any, Any]):
    wandb.init(project="ookg", group="first_stage", name=args["name"])
    wandb.config.update(args)
    og_suffix = args["suffix"]
    if args["models"] is not None:
        print("models parameter is filled. Iterate over them and ignore model.")
        for idx, model in enumerate(args["models"]):
            args["model"] = model
            args["suffix"] = f"{og_suffix}{idx}"
            process_model(args)
    else:
        process_model(args)


if __name__ == '__main__':
    argparser = ArgumentParser()

    argparser.add_argument("--model", type=Path, help="Path to model file to evaluate")
    argparser.add_argument("--models", type=Path, nargs='+', help="Path to multiple model files to evaluate")
    argparser.add_argument("--dataset", type=Path, help="Path to dataset to evaluate on. Expected to be a json file as can be found in the datasets folder.")
    argparser.add_argument("--im_kg_path", type=Path, help="In memory KG file. Expected to be a jsonl file parsed as detailed in the README.md")
    argparser.add_argument("--mention_dictionary_file", type=Path, required=True, help="A JSON file containing a mapping from each mention to a set of candidates.")
    argparser.add_argument("--type_list", type=Path, required=False, help="A JSON file with including all types that should be used for the type feature as a list.")
    argparser.add_argument("--two_hop_type_list", type=Path, required=False, help="A JSON file with including all types that should be used for the two-hop type feature as a list. Currently not in use.")
    argparser.add_argument("--transe_embeddings_file", type=Path, required=False, help="A numpy file containing all TransE embeddings necessary for the in memory KG.")
    argparser.add_argument("--transe_mappings_file", type=Path, required=False, help="A JSON file mapping each QID to a TransE embedding index as provided via argument transe_embeddings_file.")
    argparser.add_argument("--suffix", type=str, default="")
    argparser.add_argument("--beams", type=int, default=1, help="Number of beams during evaluation.")
    argparser.add_argument("--mention_split", type=int, default=5,
                           help="Number of mentions per example, for which the gradients are calculated at the same time.")

    argparser.add_argument("--mentions_to_add_if_full", type=int, default=5,
                           help="Number of mentions added as neighbors during training")

    argparser.add_argument("--number_candidates", type=int)
    argparser.add_argument("--num_cluster_candidates", type=int, default=0, help="Deprecated.")
    argparser.add_argument("--num_mentions_in_cluster", type=int, default=0, help="Deprecated")
    argparser.add_argument("--batch_size", type=int, default=1)


    argparser.add_argument("--max_length", type=int, default=512, help="Maximum length (number of tokens) of documents. If documents are longer they are split.")
    argparser.add_argument("--window_size", type=int, default=6, help="Window size of TransE embedding inclusion.")
    argparser.add_argument("--use_cosine_for_filtering", type=is_true, default=False, help="Deprecated")
    argparser.add_argument("--debug", type=is_true, default=False)
    argparser.add_argument("--add_correct_candidate", type=is_true, default=False)
    argparser.add_argument("--second_stage_clustering", type=is_true, default=False, help="If set to true, outputs a file containing all the results which can be further processed during clustering in second_stage_clustering.py ")

    argparser.add_argument("--include_ookg_score", type=is_true, default=True, help="Determines whether to calculate the ookg score or not.")
    argparser.add_argument("--spoof_perfect_linking", type=is_true, default=False, help="Deprecated")
    argparser.add_argument("--shuffle", type=is_true)
    argparser.add_argument("--clustering_general_threshold", type=float, default=0.9, help="Deprecated")
    argparser.add_argument("--clustering_mention_threshold", type=float, help="Deprecated")
    argparser.add_argument("--spoof_linking_and_detection", type=is_true, help="Leads to perfect out-of-KG detection. Used to evaluate the pure clustering performacne of out-of-KG entities")
    argparser.add_argument("--name", type=str, default="evaluation")

    main(vars(argparser.parse_args()))