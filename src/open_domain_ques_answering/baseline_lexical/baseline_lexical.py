import spacy
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from open_domain_ques_answering.utils import compute_exact, compute_f1, compute_recall_at_k, normalize_answer

nlp = spacy.load("en_core_web_sm")

class ClassicalQA:
    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus  # List of {'id': ..., 'text': ...}
        self.tokenized_corpus = [doc['text'].lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1.5, b=0.75)

    def retrieve(self, query: str, top_k: int = 10) -> List[int]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return top_indices

    def extract_answer(self, query: str, passage_text: str) -> str:
        """Finds sentence with highest word overlap, then extracts NP/NUM span."""
        doc = nlp(passage_text)
        query_words = set(normalize_answer(query).split())

        best_sent = None
        max_overlap = -1
        for sent in doc.sents:
            sent_words = set(normalize_answer(sent.text).split())
            overlap = len(query_words.intersection(sent_words))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sent = sent

        if best_sent is None:
            return passage_text[:30]

        # Extract longest Noun Phrase or Number
        spans = [chunk.text for chunk in best_sent.noun_chunks] + \
                [ent.text for ent in best_sent.ents if ent.label_ in {"NUM", "CARDINAL", "DATE"}]

        if spans:
            return max(spans, key=len)
        return best_sent.text

    def evaluate(self, qa_pairs: List[Dict[str, Any]], k_list=[1, 3, 5, 10]):
        results = {"em": 0.0, "f1": 0.0}
        for k in k_list:
            results[f"recall@{k}"] = 0.0

        for item in qa_pairs:
            q, gts = item['question'], item['answers']
            retrieved_idx = self.retrieve(q, top_k=max(k_list))
            retrieved_passages = [self.corpus[i]['text'] for i in retrieved_idx]

            # Compute recall@k
            for k in k_list:
                if compute_recall_at_k(retrieved_passages[:k], gts):
                    results[f"recall@{k}"] += 1.0

            # Answer extraction from top-1 passage
            pred = self.extract_answer(q, retrieved_passages[0])
            results["em"] += compute_exact(pred, gts)
            results["f1"] += compute_f1(pred, gts)

        n = len(qa_pairs)
        return {metric: val / n for metric, val in results.items()}
