import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

from open_domain_ques_answering.utils import compute_exact, compute_f1, compute_recall_at_k

class RAGSystem:
    def __init__(self, corpus: List[Dict[str, str]], dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.corpus = corpus
        self.retriever = SentenceTransformer(dense_model_name)

        # Build FAISS Flat Index
        passage_texts = [d['text'] for d in corpus]
        print("Encoding passages for dense retrieval...")
        embeddings = self.retriever.encode(passage_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

        # Generator initialization
        self.gen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading Generator: {self.gen_model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(
            self.gen_model_name,
            dtype=torch.float16,
            device_map="auto"
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[int]:
        q_emb = self.retriever.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(np.array(q_emb, dtype=np.float32), top_k)
        return indices[0].tolist()

    def generate_answer(self, query: str, passages: List[str]) -> str:
        truncated_passages = [p[:200] for p in passages[:5]]
        context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(truncated_passages)])

        prompt = (
            f"Answer the question concisely based strictly on the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return generated_text.strip()

    def evaluate(self, qa_pairs: List[Dict[str, Any]], k_list=[1, 3, 5, 10]):
        results = {"em": 0.0, "f1": 0.0}
        for k in k_list:
            results[f"recall@{k}"] = 0.0

        for item in qa_pairs:
            q, gts = item['question'], item['answers']
            retrieved_idx = self.retrieve(q, top_k=max(k_list))
            retrieved_passages = [self.corpus[i]['text'] for i in retrieved_idx]

            for k in k_list:
                if compute_recall_at_k(retrieved_passages[:k], gts):
                    results[f"recall@{k}"] += 1.0

            pred = self.generate_answer(q, retrieved_passages[:5])
            results["em"] += compute_exact(pred, gts)
            results["f1"] += compute_f1(pred, gts)

        n = len(qa_pairs)
        return {metric: val / n for metric, val in results.items()}
