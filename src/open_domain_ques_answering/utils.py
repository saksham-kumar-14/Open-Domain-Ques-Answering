import re
import string
from collections import Counter
from typing import List, Union

# Lowercases, removes punctuation, articles, and extra whitespace
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(prediction: str, ground_truths: Union[str, List[str]]) -> int:
    if isinstance(ground_truths, str):      # sanity check
        ground_truths = [ground_truths]
    norm_pred = normalize_answer(prediction)
    return int(any(norm_pred == normalize_answer(gt) for gt in ground_truths))

def compute_f1(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str):      # sanity check
        ground_truths = [ground_truths]

    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()

    best_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = float(pred_tokens == gt_tokens)
        elif num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(gt_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

        best_f1 = max(best_f1, f1)
    return best_f1

def compute_recall_at_k(retrieved_passages: List[str], ground_truths: Union[str, List[str]]) -> bool:
    if isinstance(ground_truths, str):      # sanity check
        ground_truths = [ground_truths]

    for passage in retrieved_passages:
        norm_passage = normalize_answer(passage)
        for gt in ground_truths:
            if normalize_answer(gt) in norm_passage:
                return True
    return False
