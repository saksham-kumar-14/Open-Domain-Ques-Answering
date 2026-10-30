import re
import json
import time
import argparse
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterator, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm

from open_domain_ques_answering.data_loader import DATA_DIR, index_paths, sample_qa_pairs
from open_domain_ques_answering.utils import normalize_answer

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "Open-Domain-Ques-Answering/0.1 (course project; https://github.com/saksham-kumar-14/Open-Domain-Ques-Answering)"
REQUEST_DELAY = 0.1  # seconds between API calls
MAX_RETRIES = 5

MIN_WORDS = 100
MAX_WORDS = 200
MIN_TAIL_WORDS = 30  # shorter leftovers at the end of a section are dropped

SKIPPED_SECTIONS = {
    "references", "external links", "see also", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "works cited",
}


class WikipediaClient:
    """Wikipedia API wrapper that caches every response on disk, so an interrupted build can resume."""

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: Dict[str, Any] = {}
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry["value"]

    def _call(self, **params) -> dict:
        url = f"{API_URL}?{urllib.parse.urlencode({'action': 'query', 'format': 'json', **params})}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        for attempt in range(MAX_RETRIES):
            time.sleep(REQUEST_DELAY)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"Wikipedia request failed ({e}), retrying...")
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("unreachable")

    def _cached(self, key: str, fetch):
        if key not in self.cache:
            self.cache[key] = fetch()
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": self.cache[key]}) + "\n")
        return self.cache[key]

    def search(self, query: str, limit: int) -> List[str]:
        def fetch():
            data = self._call(list="search", srsearch=query, srlimit=limit, srprop="")
            return [hit["title"] for hit in data["query"]["search"]]
        return self._cached(f"search:{limit}:{query}", fetch)

    def page_text(self, title: str) -> str:
        def fetch():
            data = self._call(prop="extracts", explaintext=1, redirects=1, titles=title)
            pages = data["query"]["pages"].values()
            return next((p.get("extract", "") for p in pages), "")
        return self._cached(f"page:{title}", fetch)

    def random_titles(self, count: int) -> List[str]:
        titles: List[str] = []
        while len(titles) < count:
            data = self._call(list="random", rnnamespace=0, rnfilterredir="nonredirects", rnlimit=min(count - len(titles), 100))
            titles.extend(page["title"] for page in data["query"]["random"])
        return titles


def find_titles(client: WikipediaClient, question: str, pages_per_query: int) -> List[str]:
    """Finds pages from the question text alone. The answers must never reach this function."""
    titles = client.search(question, pages_per_query)

    # All-terms matching misses pages worded differently from the question, so also match on any keyword
    keywords = [w for w in re.findall(r"[\w']+", question.lower()) if w not in ENGLISH_STOP_WORDS]
    if len(keywords) > 1:
        titles = titles + client.search(" OR ".join(keywords), pages_per_query)

    return list(dict.fromkeys(titles))


def split_long_block(text: str) -> Iterator[str]:
    """Splits text over MAX_WORDS at sentence boundaries, hard-cutting any single oversized sentence."""
    chunk: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = sentence.split()
        while len(words) > MAX_WORDS:
            if chunk:
                yield " ".join(chunk)
                chunk = []
            yield " ".join(words[:MAX_WORDS])
            words = words[MAX_WORDS:]
        if len(chunk) + len(words) > MAX_WORDS:
            yield " ".join(chunk)
            chunk = []
        chunk.extend(words)
    if chunk:
        yield " ".join(chunk)


def segment_page(text: str) -> List[str]:
    """Splits a page's plain text into passages of roughly MIN_WORDS to MAX_WORDS, never crossing a section."""
    passages: List[str] = []
    pending: List[str] = []
    skipping = False

    def flush():
        if pending:
            joined = " ".join(pending)
            pending.clear()
            for chunk in split_long_block(joined):
                if len(chunk.split()) >= MIN_TAIL_WORDS:
                    passages.append(chunk)

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        heading = re.fullmatch(r"(=+)\s*(.*?)\s*\1", line)
        if heading:
            flush()
            # A subsection of a skipped section stays skipped
            if len(heading.group(1)) == 2:
                skipping = heading.group(2).lower() in SKIPPED_SECTIONS
            continue

        if skipping:
            continue

        pending.append(line)
        if sum(len(block.split()) for block in pending) >= MIN_WORDS:
            flush()

    flush()
    return passages


def build_index(dataset_name: str, sample_size: int, seed: int, pages_per_query: int, random_pages: int) -> None:
    qa_pairs = sample_qa_pairs(dataset_name, sample_size, seed)
    index_path, manifest_path = index_paths(dataset_name, sample_size, seed)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = WikipediaClient(DATA_DIR / "wikipedia_cache.jsonl")

    titles: List[str] = []
    for item in tqdm(qa_pairs, desc="Searching Wikipedia"):
        titles.extend(find_titles(client, item["question"], pages_per_query))
    titles = list(dict.fromkeys(titles))
    question_page_count = len(titles)

    # Unrelated pages, so that retrievers have to filter out irrelevant text
    distractors = client.random_titles(random_pages)
    titles = list(dict.fromkeys(titles + distractors))

    passages = []
    for title in tqdm(titles, desc="Fetching pages"):
        slug = title.replace(" ", "_")
        for i, text in enumerate(segment_page(client.page_text(title))):
            passages.append({
                "id": f"{slug}_p{i}",
                "title": title,
                "text": text,
                "source_url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(slug)}",
            })

    with open(index_path, "w", encoding="utf-8") as f:
        for passage in passages:
            f.write(json.dumps(passage, ensure_ascii=False) + "\n")

    # Gold answers are only read here, after the index is written, to report how many are findable at all
    all_text = f" {' '.join(normalize_answer(p['text']) for p in passages)} "
    covered = sum(
        any(f" {gt} " in all_text for gt in map(normalize_answer, item["answers"]) if gt)
        for item in qa_pairs
    )

    manifest = {
        "dataset": dataset_name,
        "sample_size": len(qa_pairs),
        "seed": seed,
        "source": "English Wikipedia, live MediaWiki API (plain-text page extracts)",
        "build_date": date.today().isoformat(),
        "page_selection": f"top {pages_per_query} results for the raw question, plus top {pages_per_query} for its keywords joined by OR",
        "segmentation": f"{MIN_WORDS}-{MAX_WORDS} words per passage, split within sections at sentence boundaries, tails under {MIN_TAIL_WORDS} words dropped",
        "skipped_sections": sorted(SKIPPED_SECTIONS),
        "question_pages": question_page_count,
        "random_distractor_pages": random_pages,
        "total_pages": len(titles),
        "total_passages": len(passages),
        "answer_coverage": covered / len(qa_pairs),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Index written to {index_path}")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Build the frozen Wikipedia-paragraph index for nq_open or trivia_qa")
    parser.add_argument("--dataset", type=str, required=True, choices=["nq_open", "trivia_qa"])
    parser.add_argument("--sample_size", type=int, default=300, help="Must match the sample size used with `task`")
    parser.add_argument("--seed", type=int, default=42, help="Must match the seed used with `task`")
    parser.add_argument("--pages_per_query", type=int, default=3)
    parser.add_argument("--random_pages", type=int, default=300, help="Unrelated distractor pages to add")
    args = parser.parse_args()

    build_index(args.dataset, args.sample_size, args.seed, args.pages_per_query, args.random_pages)


if __name__ == "__main__":
    main()
