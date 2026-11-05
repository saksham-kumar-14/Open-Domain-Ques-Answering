import json
import argparse
from typing import Any, Dict, List

import numpy as np

from open_domain_ques_answering.data_loader import load_assigned_dataset
from open_domain_ques_answering.baseline_lexical.baseline_lexical import ClassicalQA
from open_domain_ques_answering.rag_reproduction.rag_reproduction import RAGSystem
from open_domain_ques_answering.hybrid.hybrid import HybridQA
from open_domain_ques_answering.utils import compute_recall_at_k, save_results

K_LIST = [1, 3, 5, 10]


def split_questions(num_questions: int, dev_size: int, split_seed: int):
    """Returns fixed dev and test question indices, so tuning never sees the test questions."""
    if not 0 < dev_size < num_questions:
        raise ValueError(f"--dev_size must be between 1 and {num_questions - 1}")
    order = np.random.RandomState(split_seed).permutation(num_questions)
    return sorted(order[:dev_size].tolist()), sorted(order[dev_size:].tolist())


def evaluate_recall(hybrid: HybridQA, qa_pairs: List[Dict[str, Any]], indices: List[int]) -> Dict[str, float]:
    """Recall@k of the fused ranking, using the same retrieval code as Task 3."""
    hits = {k: 0 for k in K_LIST}
    for i in indices:
        retrieved_idx = hybrid.retrieve_rrf(qa_pairs[i]["question"], top_k=max(K_LIST))
        passages = [hybrid.corpus[j]["text"] for j in retrieved_idx]
        for k in K_LIST:
            hits[k] += compute_recall_at_k(passages[:k], qa_pairs[i]["answers"])
    return {f"recall@{k}": hits[k] / len(indices) for k in K_LIST}


def main():
    parser = argparse.ArgumentParser(description="Choose the BM25 weight for weighted RRF on held-out dev questions")
    parser.add_argument("--dataset", type=str, default="nq_open", choices=["rag-mini-wikipedia", "nq_open", "trivia_qa"])
    parser.add_argument("--sample_size", type=int, default=300, help="Must match the sample size used with `task`")
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used with `task`")
    parser.add_argument("--dev_size", type=int, default=100, help="Questions used for tuning; the rest are the test split")
    parser.add_argument("--split_seed", type=int, default=0, help="Seed for the dev/test split")
    parser.add_argument("--weights", type=str, default="0,0.1,0.2,0.3,0.5,0.7,1.0",
                        help="Comma-separated BM25 weights to try, with the dense weight fixed at 1.0")
    parser.add_argument("--metric", type=str, default="recall@5", choices=[f"recall@{k}" for k in K_LIST],
                        help="Dev metric used to choose the weight")
    args = parser.parse_args()

    # Only the BM25/dense ratio changes the ranking, so the dense weight stays at 1.0.
    # 1.0 is standard RRF and 0.0 is pure dense retrieval; both are always evaluated as reference points.
    weights = sorted({float(w) for w in args.weights.split(",")} | {0.0, 1.0})

    corpus, qa_pairs = load_assigned_dataset(args.dataset, sample_size=args.sample_size, seed=args.seed)
    dev_idx, test_idx = split_questions(len(qa_pairs), args.dev_size, args.split_seed)
    print(f"Split: {len(dev_idx)} dev questions for tuning, {len(test_idx)} test questions held out (split_seed={args.split_seed})")

    classical = ClassicalQA(corpus)
    dense = RAGSystem(corpus, load_generator=False)
    hybrids = {w: HybridQA(classical, dense, rrf_k=60, bm25_weight=w, dense_weight=1.0) for w in weights}

    print(f"\n=== Tuning on dev split, selecting by {args.metric} ===")
    dev_results = {}
    for w in weights:
        dev_results[w] = evaluate_recall(hybrids[w], qa_pairs, dev_idx)
        row = "  ".join(f"{m}={v:.3f}" for m, v in dev_results[w].items())
        print(f"bm25_weight={w:<5} {row}")

    # Exact ties go to the weight closest to standard RRF (1.0), the setting the assignment specifies
    best_score = max(r[args.metric] for r in dev_results.values())
    best_weight = min((w for w in weights if dev_results[w][args.metric] == best_score), key=lambda w: abs(w - 1.0))
    print(f"\nBest bm25_weight on dev: {best_weight} ({args.metric}={best_score:.3f})")

    # The chosen weight is fixed before the test split is scored, so these numbers are a fair estimate
    print(f"\n=== Test split ({len(test_idx)} questions), evaluated once ===")
    test_results = {}
    for label, w in [("dense only", 0.0), ("standard RRF", 1.0), ("weighted RRF", best_weight)]:
        if w not in test_results:
            test_results[w] = evaluate_recall(hybrids[w], qa_pairs, test_idx)
        row = "  ".join(f"{m}={v:.3f}" for m, v in test_results[w].items())
        print(f"{label:<13} (bm25_weight={w}) {row}")

    metrics = {
        "best_bm25_weight": best_weight,
        "selection_metric": args.metric,
        "dev_size": len(dev_idx),
        "test_size": len(test_idx),
        "split_seed": args.split_seed,
        "dev": {str(w): r for w, r in dev_results.items()},
        "test": {str(w): r for w, r in test_results.items()},
        "test_question_indices": test_idx,
    }
    command = (f"uv run tune-rrf --dataset {args.dataset} --sample_size {args.sample_size} --seed {args.seed} "
               f"--dev_size {args.dev_size} --split_seed {args.split_seed} --weights {args.weights} --metric {args.metric}")
    save_results("rrf_tuning", args.dataset, len(qa_pairs), args.seed, command, metrics)
    print(json.dumps({"best_bm25_weight": best_weight, "dense_weight": 1.0}))


if __name__ == "__main__":
    main()
