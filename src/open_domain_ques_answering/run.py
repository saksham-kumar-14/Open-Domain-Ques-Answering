import json
import argparse

from open_domain_ques_answering.data_loader import load_assigned_dataset
from open_domain_ques_answering.baseline_lexical.baseline_lexical import ClassicalQA
from open_domain_ques_answering.rag_reproduction.rag_reproduction import RAGSystem
from open_domain_ques_answering.hybrid.hybrid import HybridQA

def main():
    parser = argparse.ArgumentParser(description="Run Task 1, Task 2, and Task 3 across Project 10 Datasets")
    parser.add_argument("--dataset", type=str, default="rag-mini-wikipedia", choices=["rag-mini-wikipedia", "nq_open", "trivia_qa"])
    parser.add_argument("--sample_size", type=int, default=300, help="Evaluation sample size")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load dataset
    corpus, qa_pairs = load_assigned_dataset(args.dataset, sample_size=args.sample_size, seed=args.seed)

    # Task 1: Classical Lexical Retrieve-and-Extract[cite: 1]
    print(f"\n=== Task 1: Classical Lexical Retrieve-and-Extract [{args.dataset}] ===")
    t1_model = ClassicalQA(corpus)
    t1_metrics = t1_model.evaluate(qa_pairs, k_list=[1, 3, 5, 10])
    print(f"Task 1 Metrics:\n{json.dumps(t1_metrics, indent=2)}")

    # Task 2: Dense Retrieval + RAG[cite: 1]
    print(f"\n=== Task 2: Dense Retrieval + RAG [{args.dataset}] ===")
    t2_model = RAGSystem(corpus)
    t2_metrics = t2_model.evaluate(qa_pairs, k_list=[1, 3, 5, 10])
    print(f"Task 2 Metrics:\n{json.dumps(t2_metrics, indent=2)}")

    # Task 3: Hybrid Fusion & Groundedness Verification[cite: 1]
    print(f"\n=== Task 3: Hybrid Retrieval (RRF) & Groundedness Verification [{args.dataset}] ===")
    t3_model = HybridQA(t1_model, t2_model, rrf_k=60)
    t3_metrics, groundedness_stats = t3_model.evaluate(qa_pairs, k_list=[1, 3, 5, 10])

    print(f"Task 3 Hybrid Metrics:\n{json.dumps(t3_metrics, indent=2)}")
    print(f"Task 3 Groundedness Analysis:\n{json.dumps(groundedness_stats, indent=2)}")

if __name__ == "__main__":
    main()
