# Open-Domain-Ques-Answering

[Primary Paper Link](https://arxiv.org/pdf/1704.00051)

## Part 1 -- Baseline: Classical Lexical Retrieve-and-Extract
Builds a fully classical, non-neural pipeline to solve Open-Domain Question Answering by splitting the problem into two distinct stages: finding the right text (`Retrieval`) and pulling out the exact answer string (`Extraction`).

### The Retrival Stage (BM25)
When a query comes in, the system cannot process the entire document corpus at once. It uses the BM25 algorithm to quickly filter the corpus down to the top k most relevant passages. 

BM25 ranks these documents based on exact word overlap, acting as an advanced version of `TF-IDF (Term Frequency-Inverse Document Frequency)`:

- It tokenizes the question and searches for those exact terms in every passage
- It applies a high weight to rare words because finding a match for a rare word is a strong mathematical signal of relevance.   
- It applies a low weight to common words to prevent them from skewing the document score.   
- *The Limitation:* Because BM25 relies purely on exact string matching, it is blind to semantic meaning. If a question asks about a "film" and the passage discusses a "movie," BM25 sees zero connection.


### The Extraction Stage (Rule-Based Heuristics)
Once BM25 retrieves the most relevant passage, the system must pinpoint a specific short answer without generating any new text. It does this using a rigid, two-step rule-based heuristic:

**1. Sentence Scoring** The system splits the top-retrieved passage into individual sentences and calculates the lexical overlap (shared vocabulary) between each sentence and the question. It assumes the sentence with the highest exact word overlap contains the target answer

**2. Span Selection** Once the highest-overlap sentence is isolated, the system uses a dependency parser (`like spaCy`) to analyze its grammar. It extracts the longest noun phrase or numerical span from that sentence and outputs it as the final answer


## Part 2 -- Paper Reproduction: Retrieval-Augmented Generation
Reproduce a dense-retrieval-plus-generation pipeline on the same index as Task 1.

[Paper Link](https://arxiv.org/pdf/2005.11401)

### Dense Retrieval
Instead of counting raw keyword overlaps, dense retrieval projects both queries and candidate passages into a shared continuous vector space using a neural encoder (`sentence-transformers/all-MiniLM-L6-v2`)
- **Semantic Embeddings:** Passages are encoded into dense vectors that capture semantic meaning rather than surface-level strings
- **Inner Product Search:** Passages are indexed into a FAISS vector database. At query time, the system computes the inner product (cosine similarity on normalized vectors) to retrieve the top k nearest neighbors.

### The Generative Stage
Once the top k passages are retrieved, the extraction phase is replaced by a generative Large Language Model (`Qwen2.5-1.5B-Instruct`).
- **Context Ingestion & Prompting:** The top retrieved passages are formatted directly into the LLM's prompt window as grounded context alongside the user query.
- **Reading Comprehension & Synthesis:** Instead of relying on brittle syntactic parsing rules (such as noun-phrase extraction in task 1), the language model comprehends the full context, resolves coreferences, and synthesizes a concise, factual answer string.
- **Deterministic Decoding:** Output generation uses greedy decoding (`do_sample=False`) to enforce temperature zero behavior (eliminating stochastic randomness), maximizing output factual accuracy and minimizing hallucinations.
