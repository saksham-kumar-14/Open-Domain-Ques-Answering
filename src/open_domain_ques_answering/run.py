import json
import argparse

from open_domain_ques_answering.data_loader import load_assigned_dataset
from open_domain_ques_answering.baseline_lexical.baseline_lexical import ClassicalQA

def main():
    parser = argparse.ArgumentParser(description="Run Task 1 Baseline across Project 10 Datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="rag-mini-wikipedia",
        choices=["rag-mini-wikipedia", "nq_open", "trivia_qa"],
        help="Dataset assignment from Section 10 of Datasets document."
    )
    parser.add_argument("--sample_size", type=int, default=300, help="Evaluation sample size for NQ/TriviaQA")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed for reproducibility")

    args = parser.parse_args()

    # Load the assigned dataset and build corpus
    corpus, qa_pairs = load_assigned_dataset(args.dataset, sample_size=args.sample_size, seed=args.seed)

    # Run Task 1 Baseline
    print(f"=== Running Task 1: Classical Lexical Retrieve-and-Extract on [{args.dataset}] ===") #
    t1_model = ClassicalQA(corpus)

    t1_metrics = t1_model.evaluate(qa_pairs, k_list=[1, 3, 5, 10]) #

    # Output results
    print(f"\nTask 1 Final Metrics ({args.dataset}):")
    print(json.dumps(t1_metrics, indent=2))

if __name__ == "__main__":
    main()
