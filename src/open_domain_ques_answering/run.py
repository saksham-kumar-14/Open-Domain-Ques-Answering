import json
import argparse

from open_domain_ques_answering.data_loader import load_assigned_dataset
from open_domain_ques_answering.baseline_lexical.baseline_lexical import ClassicalQA
from open_domain_ques_answering.rag_reproduction.rag_reproduction import RAGSystem

def main():
    parser = argparse.ArgumentParser(description="Run Task 1 and 2 across Project 10 Datasets")
    parser.add_argument("--dataset", type=str, default="rag-mini-wikipedia", choices=["rag-mini-wikipedia", "nq_open", "trivia_qa"])
    parser.add_argument("--sample_size", type=int, default=300, help="Evaluation sample size for NQ/TriviaQA")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus, qa_pairs = load_assigned_dataset(args.dataset, sample_size=args.sample_size, seed=args.seed)

    print(f"\n=== Running Task 1: Classical Lexical Retrieve-and-Extract on [{args.dataset}] ===")
    t1_model = ClassicalQA(corpus)
    t1_metrics = t1_model.evaluate(qa_pairs, k_list=[1, 3, 5, 10])
    print(f"Task 1 Final Metrics ({args.dataset}):\n{json.dumps(t1_metrics, indent=2)}")

    print(f"\n=== Running Task 2: Dense Retrieval + RAG on [{args.dataset}] ===")
    t2_model = RAGSystem(corpus)
    t2_metrics = t2_model.evaluate(qa_pairs, k_list=[1, 3, 5, 10])
    print(f"Task 2 Final Metrics ({args.dataset}):\n{json.dumps(t2_metrics, indent=2)}")

if __name__ == "__main__":
    main()
