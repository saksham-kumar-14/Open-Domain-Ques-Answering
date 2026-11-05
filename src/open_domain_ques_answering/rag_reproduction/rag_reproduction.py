import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

from open_domain_ques_answering.utils import compute_exact, compute_f1, compute_recall_at_k

class RAGSystem:
    def __init__(self, corpus: List[Dict[str, str]], dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 load_generator: bool = True):
        self.corpus = corpus
        self.retriever = SentenceTransformer(dense_model_name)

        # Build FAISS Flat Index
        passage_texts = [d['text'] for d in corpus]
        print("Encoding passages for dense retrieval...")
        embeddings = self.retriever.encode(passage_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.array(embeddings, dtype=np.float32))

        # Retrieval-only callers, such as RRF weight tuning, skip loading the generator
        if not load_generator:
            return

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.gen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        print(f"Loading Generator: {self.gen_model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_name)

        self.generator = AutoModelForCausalLM.from_pretrained(
            self.gen_model_name,
            dtype=torch.float32 if self.device == "cpu" else torch.float16,
        ).to(self.device)

    def retrieve(self, query: str, top_k: int = 5) -> List[int]:
        q_emb = self.retriever.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(np.array(q_emb, dtype=np.float32), top_k)
        return indices[0].tolist()

    def truncate_passage(self, passage: str, max_tokens: int = 200) -> str:
        token_ids = self.tokenizer(passage, add_special_tokens=False, truncation=True, max_length=max_tokens).input_ids
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def generate_answer(self, query: str, passages: List[str]) -> str:
        truncated_passages = [self.truncate_passage(p) for p in passages[:5]]
        context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(truncated_passages)])

        messages = [{
            "role": "user",
            "content": (
                f"Answer the question based strictly on the provided context. "
                f"Reply with only the answer as a short phrase, not a full sentence. "
                f"If it is a yes/no question, reply with only yes or no.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            ),
        }]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.generator.device)
        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=64,
                # Removed temperature=0.0 because do_sample=False implies greedy decoding
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
