from collections import OrderedDict
import heapq
import json
import os
import re

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "arxiv-metadata-oai-snapshot-002.json")
DATA_PATH = os.getenv("ARXIV_SNAPSHOT_PATH", DEFAULT_DATA_PATH)
MAX_SCAN_LINES = int(os.getenv("MAX_SCAN_LINES", "20000"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "20"))
MAX_RECOMMENDATIONS = int(os.getenv("MAX_RECOMMENDATIONS", "5"))
MAX_RECOMMEND_SCAN_LINES = int(os.getenv("MAX_RECOMMEND_SCAN_LINES", "20000"))

WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
}


class LruCache:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.data = OrderedDict()

    def get(self, key):
        if key in self.data:
            self.data.move_to_end(key)
            return self.data[key]
        return None

    def set(self, key, value):
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.max_size:
            self.data.popitem(last=False)


paper_cache = LruCache(max_size=500)


def dataset_available():
    return os.path.exists(DATA_PATH)


def extract_keywords(text):
    tokens = WORD_RE.findall((text or "").lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def build_search_blob(paper):
    return " ".join(
        [
            paper.get("title", ""),
            paper.get("authors", ""),
            paper.get("categories", ""),
            paper.get("abstract", ""),
        ]
    ).lower()


def iter_papers(path, max_lines=None):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if max_lines is not None and index >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def truncate(text, limit=280):
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def paper_summary(paper, score):
    return {
        "id": paper.get("id", ""),
        "title": paper.get("title", "Untitled"),
        "authors": paper.get("authors", ""),
        "categories": paper.get("categories", ""),
        "abstract_snippet": truncate(paper.get("abstract", ""), 260),
        "score": score,
    }


def search_papers(query):
    keywords = extract_keywords(query)
    if not keywords:
        return []

    heap = []
    counter = 0
    for paper in iter_papers(DATA_PATH, MAX_SCAN_LINES):
        blob = build_search_blob(paper)
        score = sum(1 for word in keywords if word in blob)
        if score == 0:
            continue
        item = (score, counter, paper)
        if len(heap) < MAX_SEARCH_RESULTS:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)
        counter += 1

    results = sorted(heap, key=lambda item: item[0], reverse=True)
    return [paper_summary(paper, score) for score, _, paper in results]


def find_paper_by_id(paper_id):
    cached = paper_cache.get(paper_id)
    if cached is not None:
        return cached

    for paper in iter_papers(DATA_PATH):
        if paper.get("id") == paper_id:
            paper_cache.set(paper_id, paper)
            return paper

    return None


def recommend_similar(paper):
    base_text = " ".join([paper.get("title", ""), paper.get("abstract", "")])
    base_keywords = extract_keywords(base_text)
    if not base_keywords:
        return []

    primary_category = ""
    categories = paper.get("categories", "")
    if categories:
        primary_category = categories.split()[0]

    heap = []
    counter = 0
    for candidate in iter_papers(DATA_PATH, MAX_RECOMMEND_SCAN_LINES):
        if candidate.get("id") == paper.get("id"):
            continue
        if primary_category and primary_category not in candidate.get("categories", ""):
            continue
        candidate_text = " ".join(
            [candidate.get("title", ""), candidate.get("abstract", "")]
        )
        candidate_keywords = extract_keywords(candidate_text)
        overlap = len(base_keywords & candidate_keywords)
        if overlap == 0:
            continue
        item = (overlap, counter, candidate)
        if len(heap) < MAX_RECOMMENDATIONS:
            heapq.heappush(heap, item)
        else:
            heapq.heappushpop(heap, item)
        counter += 1

    results = sorted(heap, key=lambda item: item[0], reverse=True)
    return [paper_summary(paper_item, score) for score, _, paper_item in results]


@app.route("/")
def index():
    query = request.args.get("q", "").strip()
    results = []
    error = ""

    if query:
        if not dataset_available():
            error = "Dataset not found."
        else:
            results = search_papers(query)
            if not results:
                error = "No matches found in the scanned range."

    return render_template(
        "index.html",
        query=query,
        results=results,
        error=error,
        max_scan_lines=MAX_SCAN_LINES,
    )


@app.route("/paper/<paper_id>")
def paper_detail(paper_id):
    if not dataset_available():
        return render_template(
            "paper.html",
            paper=None,
            recommendations=[],
            error="Dataset not found.",
        )

    paper = find_paper_by_id(paper_id)
    if not paper:
        return render_template(
            "paper.html",
            paper=None,
            recommendations=[],
            error="Paper not found in the dataset.",
        )

    recommendations = recommend_similar(paper)
    return render_template(
        "paper.html",
        paper=paper,
        recommendations=recommendations,
        error="",
    )


if __name__ == "__main__":
    app.run(debug=True)
