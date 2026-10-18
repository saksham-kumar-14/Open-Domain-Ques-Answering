```
uv run task --dataset trivia_qa --sample_size 300
```

```
Loading TriviaQA (rc.nocontext) and sampling 300 items (seed=42)...
Ignoring part of dataset_info from the dataset card: The `splits` field in YAML is malformed or missing the name/num_examples/num_bytes fields
Ignoring part of dataset_info from the dataset card: The `splits` field in YAML is malformed or missing the name/num_examples/num_bytes fields
Ignoring part of dataset_info from the dataset card: The `splits` field in YAML is malformed or missing the name/num_examples/num_bytes fields
Ignoring part of dataset_info from the dataset card: The `splits` field in YAML is malformed or missing the name/num_examples/num_bytes fields
Resolving data files: 100%|████████████████████████████████████████████████████████| 26/26 [00:00<00:00, 13821.53it/s]
Constructing passage index for TriviaQA...
Dataset successfully prepared: 600 passages in index, 300 QA test pairs.

=== Running Task 1: Classical Lexical Retrieve-and-Extract on [trivia_qa] ===

Task 1 Final Metrics (trivia_qa):
{
  "em": 0.013333333333333334,
  "f1": 0.06823833573833572,
  "recall@1": 0.9733333333333334,
  "recall@3": 1.0,
  "recall@5": 1.0,
  "recall@10": 1.0
}
```
