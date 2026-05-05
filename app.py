from collections import OrderedDict
import argparse
import json
import os
import re
import sqlite3
import sys

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "arxiv-metadata-oai-snapshot-002.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "hep_ph_train_embeddings.json")
PAPER_DB_PATH = os.path.join(BASE_DIR, "papers.sqlite")
MAX_SCAN_LINES = 20000
MAX_SEARCH_RESULTS = 20
MAX_RECOMMENDATIONS = 5
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
    "ids": [],
    "id_to_index": {},
    "vectors": None,
}
embedding_id_cache = {
    "ready": False,
    "error": "",
    "ids_set": None,
    "synced_db": False,
}


def auto_build_cache_enabled():
    value = os.getenv("AUTO_BUILD_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def embeddings_cache_path():
    max_embeddings = recommender_max_embeddings()
    if max_embeddings:
        return os.path.join(
            BASE_DIR,
            f"hep_ph_train_embeddings_{max_embeddings}.npz",
        )
    return os.path.join(BASE_DIR, "hep_ph_train_embeddings.npz")


def embeddings_source_available():
    return os.path.exists(EMBEDDINGS_PATH) or os.path.exists(embeddings_cache_path())


def paper_db_available():
    return os.path.exists(PAPER_DB_PATH)


def get_paper_db_connection():
    conn = sqlite3.connect(PAPER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def build_paper_db(force=False):
    if not dataset_available():
        return False, "Dataset not found."

    if os.path.exists(PAPER_DB_PATH):
        if not force:
            return True, ""
        os.remove(PAPER_DB_PATH)

    conn = sqlite3.connect(PAPER_DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute(
            """
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                authors TEXT,
                categories TEXT,
                abstract TEXT,
                update_date TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE embedded_ids (
                id TEXT PRIMARY KEY
            )
            """
        )
        embedding_id_cache["synced_db"] = False

        batch = []
        insert_sql = (
            "INSERT OR REPLACE INTO papers "
            "(id, title, authors, categories, abstract, update_date) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        for paper in iter_papers(DATA_PATH):
            paper_id = paper.get("id")
            if not paper_id:
                continue
            batch.append(
                (
                    paper_id,
                    paper.get("title", ""),
                    paper.get("authors", ""),
                    paper.get("categories", ""),
                    paper.get("abstract", ""),
                    paper.get("update_date", ""),
                )
            )
            if len(batch) >= 5000:
                conn.executemany(insert_sql, batch)
                conn.commit()
                batch.clear()

        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
    except Exception as exc:
        return False, f"Failed to build paper index: {exc}"
    finally:
        conn.close()

    return True, ""


def ensure_paper_db():
    if paper_db_available():
        return True
    if not auto_build_cache_enabled():
        return False
    ok, _ = build_paper_db()
    return ok


def build_embeddings_cache(force=False):
    cache_path = embeddings_cache_path()
    if os.path.exists(cache_path):
        if not force:
            return True, ""
        os.remove(cache_path)

    if not os.path.exists(EMBEDDINGS_PATH):
        return False, "Embeddings file not found."

    try:
        import numpy as np
    except ImportError as exc:
        return False, f"Missing dependency: {exc}"

    ids = []
    embeddings = []
    max_embeddings = recommender_max_embeddings()
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
            if max_embeddings and len(ids) >= max_embeddings:
                break

    if not ids:
        return False, "No embeddings loaded."

    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    np.savez(cache_path, ids=np.asarray(ids), vectors=vectors)

    if paper_db_available() or auto_build_cache_enabled():
        sync_embedded_ids(set(ids))

    recommender_assets.update(
        {
            "ready": False,
            "error": "",
            "ids": [],
            "id_to_index": {},
            "vectors": None,
        }
    )
    embedding_id_cache.update(
        {"ready": False, "error": "", "ids_set": None, "synced_db": False}
    )
    return True, ""


def ensure_embeddings_cache():
    cache_path = embeddings_cache_path()
    if os.path.exists(cache_path):
        return True, ""
    if not auto_build_cache_enabled():
        return False, "Embeddings cache missing. Run python app.py --build-cache."
    return build_embeddings_cache()


def build_all_caches(force=False):
    paper_ok, paper_error = build_paper_db(force=force)
    embed_ok, embed_error = build_embeddings_cache(force=force)
    errors = [error for error in (paper_error, embed_error) if error]
    return paper_ok and embed_ok, "; ".join(errors)


def normalize_embedding_ids(ids_array):
    ids_list = []
    for item in ids_array.tolist():
        if isinstance(item, bytes):
            ids_list.append(item.decode("utf-8"))
        else:
            ids_list.append(str(item))
    return ids_list


def load_embedding_ids():
    if embedding_id_cache["ready"] or embedding_id_cache["error"]:
        return embedding_id_cache

    if not recommender_enabled():
        embedding_id_cache["error"] = "Recommendations are disabled."
        return embedding_id_cache

    cache_path = embeddings_cache_path()
    if not os.path.exists(cache_path):
        ok, error = ensure_embeddings_cache()
        if not ok:
            embedding_id_cache["error"] = error
            return embedding_id_cache

    try:
        import numpy as np
    except ImportError as exc:
        embedding_id_cache["error"] = f"Missing dependency: {exc}"
        return embedding_id_cache

    try:
        with np.load(cache_path, allow_pickle=True) as payload:
            ids = payload["ids"]
    except Exception as exc:  # noqa: BLE001
        embedding_id_cache["error"] = f"Failed to load embeddings ids: {exc}"
        return embedding_id_cache

    if ids is None or len(ids) == 0:
        embedding_id_cache["error"] = "No embeddings loaded."
        return embedding_id_cache

    embedding_id_cache.update(
        {
            "ready": True,
            "error": "",
            "ids_set": set(normalize_embedding_ids(ids)),
            "synced_db": embedding_id_cache.get("synced_db", False),
        }
    )
    return embedding_id_cache


def sync_embedded_ids(embedded_ids=None):
    if embedding_id_cache.get("synced_db") and paper_db_available():
        return True, ""

    if embedded_ids is None:
        cache = load_embedding_ids()
        if not cache["ready"]:
            return False, cache["error"] or "Embeddings are not available."
        embedded_ids = cache["ids_set"] or set()

    if not embedded_ids:
        return False, "No embedded ids available."

    if not ensure_paper_db():
        return False, "Paper index not available."

    conn = None
    try:
        conn = get_paper_db_connection()
        conn.execute("CREATE TABLE IF NOT EXISTS embedded_ids (id TEXT PRIMARY KEY)")
        conn.execute("DELETE FROM embedded_ids")

        batch = []
        for paper_id in embedded_ids:
            batch.append((paper_id,))
            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT OR IGNORE INTO embedded_ids (id) VALUES (?)",
                    batch,
                )
                conn.commit()
                batch.clear()

        if batch:
            conn.executemany(
                "INSERT OR IGNORE INTO embedded_ids (id) VALUES (?)",
                batch,
            )
            conn.commit()
    except sqlite3.Error as exc:
        return False, f"Failed to sync embedded ids: {exc}"
    finally:
        if conn is not None:
            conn.close()

    embedding_id_cache["synced_db"] = True
    return True, ""


def get_embedded_stats():
    cache = load_embedding_ids()
    if not cache["ready"]:
        return 0, cache["error"] or "Embeddings are not available."
    return len(cache["ids_set"] or set()), ""


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

    if remaining and not paper_db_available():
        ensure_paper_db()

    if remaining and paper_db_available():
        conn = None
        try:
            conn = get_paper_db_connection()
            placeholders = ",".join("?" for _ in remaining)
            query = (
                "SELECT id, title, authors, categories, abstract, update_date "
                f"FROM papers WHERE id IN ({placeholders})"
            )
            rows = conn.execute(query, tuple(remaining)).fetchall()
            for row in rows:
                paper = dict(row)
                paper_id = paper.get("id")
                if not paper_id:
                    continue
                paper_cache.set(paper_id, paper)
                found[paper_id] = paper
                remaining.discard(paper_id)
        except sqlite3.Error:
            pass
        finally:
            if conn is not None:
                conn.close()

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


def recommender_enabled():
    value = os.getenv("RECOMMENDER_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def recommender_max_embeddings():
    raw = os.getenv("RECOMMENDER_MAX_EMBEDDINGS", "").strip()
    if not raw:
        return 0
    try:
        limit = int(raw)
    except ValueError:
        return 0
    return max(limit, 0)


def recommender_available():
    if not recommender_enabled():
        return False
    cache_path = embeddings_cache_path()
    if os.path.exists(cache_path):
        return True
    if auto_build_cache_enabled():
        return os.path.exists(EMBEDDINGS_PATH)
    return False


def recommender_unavailable_reason():
    if not recommender_enabled():
        return "Recommendations are disabled to keep the app lightweight."
    cache_path = embeddings_cache_path()
    if os.path.exists(cache_path):
        return ""
    if os.path.exists(EMBEDDINGS_PATH):
        if not auto_build_cache_enabled():
            return "Embeddings cache missing. Run python app.py --build-cache."
        return ""
    if not embeddings_source_available():
        return "Embeddings file not found."
    return ""


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
    limit = max_lines if max_lines and max_lines > 0 else None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
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


def paper_summary(paper, _score=None, _score_label=None):
    return {
        "id": paper.get("id", ""),
        "title": paper.get("title", "Untitled"),
        "authors": paper.get("authors", ""),
        "categories": paper.get("categories", ""),
        "abstract_snippet": truncate(paper.get("abstract", ""), 260),
    }


def search_papers_db(keywords, limit):
    if not paper_db_available():
        return []

    if not keywords:
        return []

    terms = sorted(set(keywords))
    blob = (
        "lower(coalesce(p.title, '') || ' ' || coalesce(p.authors, '') || ' ' || "
        "coalesce(p.categories, '') || ' ' || coalesce(p.abstract, ''))"
    )
    where_expr = " OR ".join([f"{blob} LIKE ?" for _ in terms])

    sql = (
        "SELECT p.id, p.title, p.authors, p.categories, p.abstract, p.update_date "
        "FROM papers p "
        "JOIN embedded_ids e ON p.id = e.id "
        f"WHERE {where_expr} "
        "ORDER BY p.update_date DESC "
        "LIMIT ?"
    )

    params = [f"%{term}%" for term in terms]
    params.append(limit)

    conn = None
    try:
        conn = get_paper_db_connection()
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        if conn is not None:
            conn.close()

    results = []
    for row in rows:
        paper = dict(row)
        results.append(paper_summary(paper))
        paper_id = paper.get("id")
        if paper_id:
            paper_cache.set(paper_id, paper)
    return results


def search_papers(query):
    keywords = extract_keywords(query)
    if not keywords:
        return [], ""

    embedding_cache = load_embedding_ids()
    if not embedding_cache["ready"]:
        return [], embedding_cache["error"] or "Embeddings are not available."

    ok, error = sync_embedded_ids(embedding_cache["ids_set"])
    if not ok:
        return [], error

    results = search_papers_db(keywords, MAX_SEARCH_RESULTS)
    return results, ""


def find_paper_by_id(paper_id):
    return load_papers_by_ids({paper_id}).get(paper_id)


def load_recommender_assets():
    if recommender_assets["ready"] or recommender_assets["error"]:
        return recommender_assets

    if not recommender_enabled():
        recommender_assets["error"] = "Recommendations are disabled."
        return recommender_assets

    try:
        import numpy as np
    except ImportError as exc:
        recommender_assets["error"] = f"Missing dependency: {exc}"
        return recommender_assets
    cache_path = embeddings_cache_path()
    if not os.path.exists(cache_path):
        ok, error = ensure_embeddings_cache()
        if not ok:
            recommender_assets["error"] = error
            return recommender_assets

    try:
        with np.load(cache_path, allow_pickle=True) as payload:
            ids = payload["ids"]
            vectors = payload["vectors"]
    except Exception as exc:  # noqa: BLE001
        recommender_assets["error"] = f"Failed to load embeddings cache: {exc}"
        return recommender_assets

    if ids is None or vectors is None or len(ids) == 0:
        recommender_assets["error"] = "No embeddings loaded."
        return recommender_assets

    if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIM:
        recommender_assets["error"] = "Embeddings cache has an unexpected shape."
        return recommender_assets

    if len(ids) != vectors.shape[0]:
        recommender_assets["error"] = "Embeddings cache is inconsistent."
        return recommender_assets

    ids_list = normalize_embedding_ids(ids)

    recommender_assets.update(
        {
            "ready": True,
            "error": "",
            "ids": ids_list,
            "id_to_index": {
                paper_id: index for index, paper_id in enumerate(ids_list)
            },
            "vectors": vectors,
        }
    )
    return recommender_assets


def recommend_similar_model(paper, assets=None):
    if assets is None:
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
    if vectors is None or vectors.shape[0] <= 1:
        return []

    target_vector = vectors[target_index]
    scores = vectors @ target_vector
    scores[target_index] = -1.0e9

    candidate_count = scores.size - 1
    if candidate_count <= 0:
        return []

    candidate_pool = max(MAX_RECOMMEND_CANDIDATES, MAX_RECOMMENDATIONS * 25)
    top_k = min(candidate_pool, candidate_count)
    top_indices = np.argpartition(scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    ids = assets["ids"]
    candidate_ids = [ids[index] for index in top_indices]
    candidate_papers = load_papers_by_ids(candidate_ids)

    recommendations = []
    for candidate_index in top_indices:
        candidate_id = ids[candidate_index]
        candidate_paper = candidate_papers.get(candidate_id)
        if not candidate_paper:
            candidate_paper = {
                "id": candidate_id,
                "title": "Metadata unavailable",
                "authors": "",
                "categories": "",
                "abstract": "",
            }
        recommendations.append(
            paper_summary(
                candidate_paper,
                float(scores[candidate_index]),
            )
        )
        if len(recommendations) >= MAX_RECOMMENDATIONS:
            break

    return recommendations


def recommend_similar(paper, assets=None):
    recommendations = recommend_similar_model(paper, assets=assets)
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
            results, search_error = search_papers(query)
            if search_error:
                error = search_error
            elif not results:
                error = "No matches found in the embedded papers."

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
            recs_available=False,
            recs_unavailable_reason=recommender_unavailable_reason(),
        )

    paper = find_paper_by_id(paper_id)
    if not paper:
        return render_template(
            "paper.html",
            paper=None,
            recommendations=[],
            error="Paper not found in the dataset.",
            recs_available=recommender_available(),
            recs_unavailable_reason=recommender_unavailable_reason(),
        )

    recs_available = recommender_available()
    recs_unavailable_reason = recommender_unavailable_reason()
    if recs_available:
        embedding_cache = load_embedding_ids()
        if not embedding_cache["ready"]:
            recs_available = False
            recs_unavailable_reason = (
                embedding_cache["error"]
                or "Embeddings are not available."
            )
        else:
            if paper.get("id") not in (embedding_cache["ids_set"] or set()):
                recs_available = False
                recs_unavailable_reason = (
                    "This paper is not in the embeddings index. "
                    "Recommendations are limited to embedded papers."
                )

    return render_template(
        "paper.html",
        paper=paper,
        recommendations=[],
        error="",
        recs_available=recs_available,
        recs_unavailable_reason=recs_unavailable_reason,
    )


@app.route("/paper/<paper_id>/recommendations")
def paper_recommendations(paper_id):
    if not dataset_available():
        return jsonify({"recommendations": [], "error": "Dataset not found."}), 404

    if not recommender_available():
        reason = recommender_unavailable_reason() or "Recommendations are not available."
        return jsonify({"recommendations": [], "error": reason}), 400

    assets = load_recommender_assets()
    if not assets["ready"]:
        error = assets["error"] or "Recommendations are not available."
        return jsonify({"recommendations": [], "error": error}), 400

    paper = find_paper_by_id(paper_id)
    if not paper:
        return (
            jsonify(
                {
                    "recommendations": [],
                    "error": "Paper not found in the dataset.",
                }
            ),
            404,
        )

    if paper.get("id") not in assets["id_to_index"]:
        return (
            jsonify(
                {
                    "recommendations": [],
                    "error": (
                        "This paper is not in the embeddings index. "
                        "Recommendations are limited to embedded papers."
                    ),
                }
            ),
            400,
        )

    recommendations = recommend_similar(paper, assets=assets)
    return jsonify({"recommendations": recommendations, "error": ""})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Finder server")
    parser.add_argument(
        "--build-cache",
        action="store_true",
        help="Build the SQLite and embeddings caches",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild caches even if they already exist",
    )
    args = parser.parse_args()

    if args.build_cache:
        ok, error = build_all_caches(force=args.force)
        if not ok:
            print(error or "Failed to build caches.", file=sys.stderr)
            sys.exit(1)
        print("Cache build complete.")
        sys.exit(0)

    app.run(debug=True)
