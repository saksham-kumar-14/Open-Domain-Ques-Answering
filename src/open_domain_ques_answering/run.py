import json
import argparse
from datasets import load_dataset

# Absolute import matching your src/ layout
from open_domain_ques_answering.baseline_lexical.baseline_lexical import ClassicalQA

def main():
    parser = argparse.ArgumentParser(description="Run Open-Domain QA Tasks")
    parser.add_argument("--dataset", type=str, default="rag-mini-wikipedia", help="Assigned dataset")
    args = parser.parse_args()

    print(f"--- Loading Dataset: {args.dataset} ---")
    if args.dataset == "rag-mini-wikipedia":
        # Load the passage corpus[cite: 20]
        ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
        corpus = [{"id": str(i), "text": item["passage"]} for i, item in enumerate(ds["passages"])]

        # Load the question-answer pairs[cite: 20]
        qa_ds = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
        qa_pairs = [
            {"question": item["question"], "answers": [item["answer"]]} for item in qa_ds["test"]
        ]
    else:
        raise NotImplementedError("Only rag-mini-wikipedia is set up by default.")

    print(f"Loaded {len(corpus)} passages and {len(qa_pairs)} QA pairs.")

    # Execute Task 1 Baseline
    print("\n=== Running Task 1: BM25 + Extraction ===")
    t1_model = ClassicalQA(corpus)

    # Evaluate returns EM, F1, and Recall@k metrics
    t1_metrics = t1_model.evaluate(qa_pairs)

    print("\nTask 1 Metrics:")
    print(json.dumps(t1_metrics, indent=2))

if __name__ == "__main__":
    main()
