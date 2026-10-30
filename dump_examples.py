import json

from open_domain_ques_answering.data_loader import load_assigned_dataset
from open_domain_ques_answering.baseline_lexical.baseline_lexical import ClassicalQA
from open_domain_ques_answering.rag_reproduction.rag_reproduction import RAGSystem
from open_domain_ques_answering.hybrid.hybrid import HybridQA
from open_domain_ques_answering.utils import compute_exact, compute_f1

corpus, qa_pairs = load_assigned_dataset("nq_open", sample_size=300, seed=42)
t1 = ClassicalQA(corpus)
t2 = RAGSystem(corpus)
t3 = HybridQA(t1, t2, rrf_k=60)

rows = []
for item in qa_pairs:
    q, gts = item["question"], item["answers"]
    dense = [corpus[i] for i in t2.retrieve(q, top_k=5)]
    hybrid = [corpus[i] for i in t3.retrieve_rrf(q, top_k=5)]
    dense_texts = [p["text"] for p in dense]
    hybrid_texts = [p["text"] for p in hybrid]
    t2_answer = t2.generate_answer(q, dense_texts)
    t3_answer = t2.generate_answer(q, hybrid_texts)
    overlap = t3.compute_lexical_overlap(t3_answer, hybrid_texts)
    rows.append({
        "question": q,
        "gold": gts,
        "task2": {
            "answer": t2_answer,
            "em": compute_exact(t2_answer, gts),
            "f1": compute_f1(t2_answer, gts),
            "top_titles": [p["title"] for p in dense],
        },
        "task3": {
            "answer": t3_answer,
            "em": compute_exact(t3_answer, gts),
            "f1": compute_f1(t3_answer, gts),
            "overlap": overlap,
            "grounded": overlap >= 0.5,
            "top_titles": [p["title"] for p in hybrid],
        },
    })

with open("examples_nq_open_n300.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("Task 2 exact matches:", sum(r["task2"]["em"] for r in rows), "(report: 83)")
print("Task 3 exact matches:", sum(r["task3"]["em"] for r in rows), "(report: 77)")
print("Task 3 flagged:", sum(not r["task3"]["grounded"] for r in rows), "(report: 26)")
