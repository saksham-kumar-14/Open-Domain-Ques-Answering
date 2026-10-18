```
uv run task --dataset nq_open --sample_size 300
```

```
Loading Natural Questions (Open) and sampling 300 validation items (seed=42)...
Constructing passage index for sampled Natural Questions...
Dataset successfully prepared: 900 passages in index, 300 QA test pairs.

=== Running Task 1: Classical Lexical Retrieve-and-Extract on [nq_open] ===

Task 1 Final Metrics (nq_open):
{
  "em": 0.0,
  "f1": 0.0027777777777777775,
  "recall@1": 0.03333333333333333,
  "recall@3": 0.9966666666666667,
  "recall@5": 1.0,
  "recall@10": 1.0
}
```
