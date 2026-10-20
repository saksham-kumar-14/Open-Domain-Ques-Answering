```
uv run task --dataset nq_open --sample_size 30
```

```
=== Task 1: Classical Lexical Retrieve-and-Extract [nq_open] ===
Task 1 Metrics:
{
  "em": 0.0,
  "f1": 0.0,
  "recall@1": 0.0,
  "recall@3": 1.0,
  "recall@5": 1.0,
  "recall@10": 1.0
}

=== Task 2: Dense Retrieval + RAG [nq_open] ===
Loading weights: 100%|█████████████████████████████████████████████| 103/103 [00:00<00:00, 7270.38it/s]
Encoding passages for dense retrieval...
Batches: 100%|███████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  2.54it/s]
Loading Generator: Qwen/Qwen2.5-1.5B-Instruct on mps...
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|███████████████████████████████████████████████| 338/338 [00:06<00:00, 52.94it/s]
Task 2 Metrics:
{
  "em": 0.0,
  "f1": 0.0808315232986087,
  "recall@1": 0.13333333333333333,
  "recall@3": 1.0,
  "recall@5": 1.0,
  "recall@10": 1.0
}

=== Task 3: Hybrid Retrieval (RRF) & Groundedness Verification [nq_open] ===
Task 3 Hybrid Metrics:
{
  "em": 0.0,
  "f1": 0.08441826800092357,
  "recall@1": 0.0,
  "recall@3": 1.0,
  "recall@5": 1.0,
  "recall@10": 1.0
}
Task 3 Groundedness Analysis:
{
  "frac_correct_high_overlap": 0.0,
  "frac_incorrect_low_overlap": 0.5333333333333333,
  "groundedness_check_accuracy": 0.5333333333333333
}
```
