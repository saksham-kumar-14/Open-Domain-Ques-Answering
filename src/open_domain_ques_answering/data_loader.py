import json
import numpy as np
from datasets import load_dataset
from pathlib import Path
from typing import List, Dict, Any, Tuple

DATA_DIR = Path(__file__).parent / "data"


def index_paths(dataset_name: str, sample_size: int, seed: int) -> Tuple[Path, Path]:
    """Returns the passage index and manifest paths for one dataset sample."""
    stem = f"{dataset_name}_n{sample_size}_seed{seed}"
    return DATA_DIR / f"{stem}.jsonl", DATA_DIR / f"{stem}.manifest.json"


def sample_qa_pairs(dataset_name: str, sample_size: int, seed: int) -> List[Dict[str, Any]]:
    """Draws the fixed validation sample for nq_open or trivia_qa, with every acceptable answer flattened into a list."""
    if dataset_name == "nq_open":
        print(f"Loading Natural Questions (Open) and sampling {sample_size} validation items (seed={seed})...")
        ds_qa = load_dataset("google-research-datasets/nq_open", split="validation")
    elif dataset_name == "trivia_qa":
        print(f"Loading TriviaQA (rc.nocontext) and sampling {sample_size} items (seed={seed})...")
        ds_qa = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from ['nq_open', 'trivia_qa']")

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds_qa), size=min(sample_size, len(ds_qa)), replace=False)
    sampled_data = [ds_qa[int(i)] for i in indices]

    if dataset_name == "nq_open":
        return [{"question": item["question"], "answers": item["answer"]} for item in sampled_data]

    # Preprocessing: Combine value + aliases into one flat list
    qa_pairs = []
    for item in sampled_data:
        ans_dict = item["answer"]
        valid_answers = [ans_dict["value"]] + ans_dict.get("aliases", [])
        qa_pairs.append({
            "question": item["question"],
            "answers": list(dict.fromkeys(valid_answers)) # Flattened & deduplicated
        })
    return qa_pairs


def load_wikipedia_index(dataset_name: str, sample_size: int, seed: int) -> List[Dict[str, str]]:
    """Loads the frozen Wikipedia-paragraph index built by prepare_index.py."""
    index_path, _ = index_paths(dataset_name, sample_size, seed)
    if not index_path.exists():
        raise FileNotFoundError(
            f"No passage index at {index_path}. Build it first with:\n"
            f"  uv run prepare-index --dataset {dataset_name} --sample_size {sample_size} --seed {seed}"
        )
    with open(index_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_assigned_dataset(dataset_name: str, sample_size: int = 300, seed: int = 42) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Loads and normalizes one of the 3 assignable datasets for Project 10.
    Returns:
      corpus: List of dicts [{'id': str, 'text': str, ...}]
      qa_pairs: List of dicts [{'question': str, 'answers': List[str]}]
    """
    if dataset_name == "rag-mini-wikipedia":
        print("Loading rag-mini-wikipedia...")
        ds_corpus = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
        corpus = [{"id": str(i), "text": item["passage"]} for i, item in enumerate(ds_corpus["passages"])]

        ds_qa = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
        qa_pairs = [
            {"question": item["question"], "answers": [item["answer"]]}
            for item in ds_qa["test"]
        ]

    elif dataset_name in ("nq_open", "trivia_qa"):
        corpus = load_wikipedia_index(dataset_name, sample_size, seed)
        qa_pairs = sample_qa_pairs(dataset_name, sample_size, seed)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from ['rag-mini-wikipedia', 'nq_open', 'trivia_qa']")

    print(f"Dataset successfully prepared: {len(corpus)} passages in index, {len(qa_pairs)} QA test pairs.\n")
    return corpus, qa_pairs
