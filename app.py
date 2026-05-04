from collections import OrderedDict
import heapq
import json
import os
import re

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "arxiv-metadata-oai-snapshot-002.json")
MODEL_PATH = os.path.join(BASE_DIR, "scientific_recommender_gmm.pkl")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "hep_ph_train_embeddings.json")
MAX_SCAN_LINES = 20000
MAX_SEARCH_RESULTS = 20
MAX_RECOMMENDATIONS = 5
MAX_RECOMMEND_SCAN_LINES = 20000
MAX_RECOMMEND_CANDIDATES = 50
EMBEDDING_DIM = 768

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
recommender_assets = {
    "ready": False,
    "error": "",
    "model": None,
    "ids": [],
    "id_to_index": {},
    "vectors": None,
    "clusters": None,
}


def load_papers_by_ids(paper_ids, max_lines=None):
    remaining = {paper_id for paper_id in paper_ids if paper_id}
    found = {}

    for paper_id in list(remaining):
        cached = paper_cache.get(paper_id)
        if cached is not None:
            found[paper_id] = cached
            remaining.remove(paper_id)

    if not remaining:
        return found

    for paper in iter_papers(DATA_PATH, max_lines=max_lines):
        paper_id = paper.get("id")
        if paper_id in remaining:
            paper_cache.set(paper_id, paper)
            found[paper_id] = paper
            remaining.remove(paper_id)
            if not remaining:
                break

    return found


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


def paper_summary(paper, score, score_label=None):
    score_value = score
    score_display = ""
    try:
        score_value = float(score)
        if score_value.is_integer():
            score_value = int(score_value)
            score_display = str(score_value)
        else:
            score_display = f"{score_value:.3f}"
    except (TypeError, ValueError):
        score_display = str(score)
    return {
        "id": paper.get("id", ""),
        "title": paper.get("title", "Untitled"),
        "authors": paper.get("authors", ""),
        "categories": paper.get("categories", ""),
        "abstract_snippet": truncate(paper.get("abstract", ""), 260),
        "score": score_value,
        "score_display": score_display,
        "score_label": score_label,
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
    for score, _, paper in results:
        paper_id = paper.get("id")
        if paper_id:
            paper_cache.set(paper_id, paper)
    return [paper_summary(paper, score) for score, _, paper in results]


def find_paper_by_id(paper_id):
    return load_papers_by_ids({paper_id}).get(paper_id)


def load_recommender_assets():
    if recommender_assets["ready"] or recommender_assets["error"]:
        return recommender_assets

    if not os.path.exists(MODEL_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        recommender_assets["error"] = "Model or embeddings file not found."
        return recommender_assets

    try:
        import joblib
        import numpy as np
    except ImportError as exc:
        recommender_assets["error"] = f"Missing dependency: {exc}"
        return recommender_assets

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        recommender_assets["error"] = f"Failed to load model: {exc}"
        return recommender_assets

    ids = []
    embeddings = []
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            paper_id = row.get("id")
            vector = row.get("embedding")
            if not paper_id or not isinstance(vector, list):
                continue
            if len(vector) != EMBEDDING_DIM:
                continue
            ids.append(paper_id)
            embeddings.append(vector)

    if not ids:
        recommender_assets["error"] = "No embeddings loaded."
        return recommender_assets

    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    topic_vectors = None
    clusters = None
    if hasattr(model, "predict_proba"):
        try:
            topic_vectors = model.predict_proba(embeddings)
            topic_vectors = np.asarray(topic_vectors, dtype=np.float32)
            tnorms = np.linalg.norm(topic_vectors, axis=1, keepdims=True)
            tnorms[tnorms == 0] = 1.0
            topic_vectors = topic_vectors / tnorms
            clusters = topic_vectors.argmax(axis=1)
        except Exception:
            topic_vectors = None
            clusters = None

    if topic_vectors is None and hasattr(model, "predict"):
        try:
            clusters = model.predict(embeddings)
        except Exception:
            clusters = None

    if clusters is not None:
        clusters = np.asarray(clusters)

    vectors = topic_vectors if topic_vectors is not None else embeddings

    recommender_assets.update(
        {
            "ready": True,
            "model": model,
            "ids": ids,
            "id_to_index": {paper_id: index for index, paper_id in enumerate(ids)},
            "vectors": vectors,
            "clusters": clusters,
        }
    )
    return recommender_assets


def recommend_similar_model(paper):
    assets = load_recommender_assets()
    if not assets["ready"]:
        return []

    try:
        import numpy as np
    except ImportError:
        return []

    paper_id = paper.get("id")
    target_index = assets["id_to_index"].get(paper_id)
    if target_index is None:
        return []

    vectors = assets["vectors"]
    if vectors is None:
        return []

    candidate_indices = np.arange(vectors.shape[0])
    clusters = assets["clusters"]
    if clusters is not None:
        cluster_id = clusters[target_index]
        candidate_indices = np.where(clusters == cluster_id)[0]

    candidate_indices = candidate_indices[candidate_indices != target_index]
    if candidate_indices.size == 0:
        return []

    target_vector = vectors[target_index]
    scores = vectors[candidate_indices] @ target_vector

    top_k = min(MAX_RECOMMEND_CANDIDATES, candidate_indices.size)
    if top_k <= 0:
        return []

    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    ids = assets["ids"]
    candidate_ids = [ids[candidate_indices[index]] for index in top_indices]
    candidate_papers = load_papers_by_ids(
        candidate_ids,
        max_lines=MAX_RECOMMEND_SCAN_LINES,
    )

    recommendations = []
    for relative_index in top_indices:
        candidate_index = candidate_indices[relative_index]
        candidate_id = ids[candidate_index]
        candidate_paper = candidate_papers.get(candidate_id)
        if not candidate_paper:
            continue
        recommendations.append(
            paper_summary(
                candidate_paper,
                float(scores[relative_index]),
                score_label="Model similarity",
            )
        )
        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

    return recommendations


def recommend_similar(paper):
    recommendations = recommend_similar_model(paper)
    return recommendations[:MAX_RECOMMENDATIONS]


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
