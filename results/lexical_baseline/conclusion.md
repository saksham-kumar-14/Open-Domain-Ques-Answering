The stark contrast between the **Recall** scores and the **Exact Match / F1** scores exposes the fundamental limitations of classical, rule-based NLP pipelines.

### 1. The Retrieval Success (High Recall)
BM25 proved highly effective at identifying relevant documents when exact terminology was shared. For `nq_open` and `trivia_qa`, the script specifically injected "perfect" passages containing the gold answers into the index. BM25 reliably retrieved these injected documents, pushing Recall@5 and Recall@10 to a perfect **1.0 (100%)**. 

Even on `rag-mini-wikipedia`, which used a natural unmodified corpus, the system successfully located passages containing the answer in its top 10 results nearly half the time (43%).

### 2. The Extraction Failure (Low EM/F1)
Despite having the correct document in hand, the system consistently failed to extract the right answer, resulting in Exact Match scores near **0%**. 

Because the classical pipeline does not possess reading comprehension or semantic understanding, it relies on brittle grammatical heuristics (identifying the "longest noun phrase"). Instead of extracting a precise target (e.g., *"William Shakespeare"*), the parser frequently sweeps up large, irrelevant chunks of the sentence containing the answer (e.g., *"The officially recognized answer to the question..."*). 

### The Motivation for Task 2 (RAG)
These results conclusively demonstrate why modern QA systems have moved away from heuristic extraction. Finding the right text is a search problem that algorithms like BM25 can solve passably well. However, isolating the correct answer from that text requires actual semantic comprehension—setting the perfect stage for **Task 2**, where a Generative LLM will replace the brittle parser to read the retrieved context and synthesize a precise answer.
