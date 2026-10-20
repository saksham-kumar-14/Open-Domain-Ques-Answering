import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from open_domain_ques_answering.baseline_lexical.baseline_lexical import ClassicalQA
from open_domain_ques_answering.rag_reproduction.rag_reproduction import RAGSystem
from open_domain_ques_answering.utils import compute_exact, compute_f1, compute_recall_at_k, normalize_answer

class HybridQA:
    def __init__(self, classical_model: ClassicalQA, rag_model: RAGSystem, rrf_k: int = 60):
        self.bm25_model = classical_model.bm25
        self.corpus = classical_model.corpus
        self.rag_model = rag_model
        self.rrf_k = rrf_k

    def retrieve_rrf(self, query: str, top_k: int = 10) -> List[int]:
        # 1. Get BM25 lexical rankings
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_model.get_scores(tokenized_query)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]

        # 2. Get Dense vector rankings
        dense_ranked_indices = self.rag_model.retrieve(query, top_k=len(self.corpus))

        # 3. Compute RRF Scores (constant = 60)
        rrf_scores = defaultdict(float)

        for rank, doc_idx in enumerate(bm25_ranked_indices):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + (rank + 1))

        for rank, doc_idx in enumerate(dense_ranked_indices):
            rrf_scores[doc_idx] += 1.0 / (self.rrf_k + (rank + 1))

        # 4. Sort candidate documents by total RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_idx for doc_idx, _ in sorted_docs[:top_k]]

    def compute_lexical_overlap(self, answer: str, passages: List[str]) -> float:
        """Calculates token overlap ratio between the generated answer and evidence passages[cite: 1]."""
        norm_ans = normalize_answer(answer)
        ans_tokens = set(norm_ans.split())
        if not ans_tokens:
            return 0.0

        combined_passages = " ".join([normalize_answer(p) for p in passages])
        passage_tokens = set(combined_passages.split())

        overlap = ans_tokens.intersection(passage_tokens)
        return len(overlap) / len(ans_tokens)

    def evaluate(self, qa_pairs: List[Dict[str, Any]], k_list=[1, 3, 5, 10], overlap_threshold: float = 0.5):
        results = {"em": 0.0, "f1": 0.0}
        for k in k_list:
            results[f"recall@{k}"] = 0.0

        correct_high_overlap = 0
        total_correct = 0

        incorrect_low_overlap = 0
        total_incorrect = 0

        grounded_check_correct = 0

        for item in qa_pairs:
            q, gts = item['question'], item['answers']

            # Hybrid RRF Retrieval
            retrieved_idx = self.retrieve_rrf(q, top_k=max(k_list))
            retrieved_passages = [self.corpus[i]['text'] for i in retrieved_idx]

            # Compute Hybrid Recall@k
            for k in k_list:
                if compute_recall_at_k(retrieved_passages[:k], gts):
                    results[f"recall@{k}"] += 1.0

            # Generate Answer using top-5 hybrid retrieved passages
            pred = self.rag_model.generate_answer(q, retrieved_passages[:5])

            em_score = compute_exact(pred, gts)
            f1_score = compute_f1(pred, gts)

            results["em"] += em_score
            results["f1"] += f1_score

            # Groundedness Check
            overlap_ratio = self.compute_lexical_overlap(pred, retrieved_passages[:5])
            is_grounded = (overlap_ratio >= overlap_threshold)
            is_correct = (em_score == 1.0)

            if is_correct:
                total_correct += 1
                if is_grounded:
                    correct_high_overlap += 1
            else:
                total_incorrect += 1
                if not is_grounded:
                    incorrect_low_overlap += 1

            # Groundedness verification accuracy:
            # Correct answer grounded OR Incorrect answer ungrounded
            if (is_correct and is_grounded) or (not is_correct and not is_grounded):
                grounded_check_correct += 1

        n = len(qa_pairs)
        metrics = {metric: val / n for metric, val in results.items()}

        groundedness_stats = {
            "frac_correct_high_overlap": (correct_high_overlap / total_correct) if total_correct > 0 else 0.0,
            "frac_incorrect_low_overlap": (incorrect_low_overlap / total_incorrect) if total_incorrect > 0 else 0.0,
            "groundedness_check_accuracy": grounded_check_correct / n if n > 0 else 0.0
        }

        return metrics, groundedness_stats
