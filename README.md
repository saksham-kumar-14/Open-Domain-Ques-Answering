# Open-Domain-Ques-Answering

## Part 1 -- Baseline: Classical Lexical Retrieve-and-Extract
Builds a fully classical, non-neural pipeline to solve Open-Domain Question Answering by splitting the problem into two distinct stages: finding the right text (Retrieval) and pulling out the exact answer string (Extraction).

### The Retrival Stage (BM25)
When a query comes in, the system cannot process the entire document corpus at once. It uses the BM25 algorithm to quickly filter the corpus down to the top k most relevant passages. 

BM25 ranks these documents based on exact word overlap, acting as an advanced version of TF-IDF (Term Frequency-Inverse Document Frequency):

- It tokenizes the question and searches for those exact terms in every passage
- It applies a high weight to rare words because finding a match for a rare word is a strong mathematical signal of relevance.   
- It applies a low weight to common words to prevent them from skewing the document score.   
- *The Limitation:* Because BM25 relies purely on exact string matching, it is blind to semantic meaning. If a question asks about a "film" and the passage discusses a "movie," BM25 sees zero connection.


### The Extraction Stage (Rule-Based Heuristics)
Once BM25 retrieves the most relevant passage, the system must pinpoint a specific short answer without generating any new text. It does this using a rigid, two-step rule-based heuristic:

**1. Sentence Scoring** The system splits the top-retrieved passage into individual sentences and calculates the lexical overlap (shared vocabulary) between each sentence and the question. It assumes the sentence with the highest exact word overlap contains the target answer

**2. Span Selection** Once the highest-overlap sentence is isolated, the system uses a dependency parser (like spaCy) to analyze its grammar. It extracts the longest noun phrase or numerical span from that sentence and outputs it as the final answer
