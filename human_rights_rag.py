#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Optional .env loading
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


# ----------------------------
# Environment / small helpers
# ----------------------------

def load_env() -> None:
    """Load env vars from common .env locations if python-dotenv exists."""
    if load_dotenv is None:
        return
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env",
        here / "human_rights_crew" / ".env",
        Path.cwd() / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_url(url: str) -> str:
    """Normalize URLs so evaluation doesn't fail on tiny differences."""
    try:
        p = urlparse(url.strip())
        scheme = "https"
        netloc = (p.netloc or "").lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = p.path or ""
        if path.endswith("/") and path != "/":
            path = path[:-1]
        return urlunparse((scheme, netloc, path, "", p.query or "", ""))
    except Exception:
        return url.strip()


# ----------------------------
# Scraping / cleaning
# ----------------------------

NOISE_PATTERNS = [
    "This site was archived on 2023-02-01 and is no longer receiving updates",
    "Links, accessibility, and other functionality may be limited.",
    "University of Minnesota Human Rights Library",
    "Human Rights Library",
]
DROP_LINE_PREFIXES = {"home", "search", "contact", "copyright"}


def strip_noise(text: str) -> str:
    for pat in NOISE_PATTERNS:
        text = text.replace(pat, "")
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = []
    for ln in lines:
        if not ln:
            continue
        low = ln.lower().strip()
        first = low.split(" ")[0] if low.split(" ") else low
        if first in DROP_LINE_PREFIXES and len(low.split()) <= 5:
            continue
        cleaned.append(ln)
    return "\n".join(cleaned)


def fetch_and_clean(session: requests.Session, url: str, timeout: int = 20) -> str:
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"[WARN] Failed to fetch {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    raw = soup.get_text(separator="\n")
    raw = strip_noise(raw)

    lines = []
    for ln in raw.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            lines.append(ln)
    return "\n".join(lines)


def chunk_text(text: str, max_words: int = 400, overlap: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(1, max_words - overlap)
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
        start += step
    return chunks


def cache_path(out_dir: str, url: str) -> str:
    ensure_dir(os.path.join(out_dir, "cache"))
    return os.path.join(out_dir, "cache", f"{sha1(url)}.txt")


def build_docs_from_csv(csv_path: str, out_dir: str, timeout: int = 20, sleep_s: float = 0.0):
    df = pd.read_csv(csv_path)
    if "URL" not in df.columns or "Title" not in df.columns:
        raise ValueError("CSV must contain columns: URL, Title")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (HumanRightsRAG/1.0)"})

    docs = []
    for i, row in df.iterrows():
        url = str(row["URL"]).strip()
        title = str(row["Title"]).strip()
        if not url or url.lower() == "nan":
            continue

        cpath = cache_path(out_dir, url)
        if os.path.exists(cpath):
            txt = open(cpath, "r", encoding="utf-8", errors="ignore").read().strip()
            if txt:
                docs.append({"doc_id": int(i), "url": url, "title": title, "text": txt})
            continue

        print(f"[FETCH] {i}: {title} | {url}")
        txt = fetch_and_clean(session, url, timeout=timeout)
        with open(cpath, "w", encoding="utf-8") as f:
            f.write(txt)

        if txt:
            docs.append({"doc_id": int(i), "url": url, "title": title, "text": txt})

        if sleep_s > 0:
            time.sleep(sleep_s)

    return docs


def build_corpus(docs, max_words=400, overlap=80):
    corpus = []
    for d in docs:
        chunks = chunk_text(d["text"], max_words=max_words, overlap=overlap)
        for j, ch in enumerate(chunks):
            corpus.append({
                "doc_id": d["doc_id"],
                "url": d["url"],
                "url_norm": canonicalize_url(d["url"]),
                "title": d["title"],
                "chunk_id": j,
                "text": ch
            })
    return corpus


# ----------------------------
# Embeddings + FAISS
# ----------------------------

def embed_and_index(corpus, out_dir: str, model_name: str):
    from sentence_transformers import SentenceTransformer
    import faiss

    ensure_dir(out_dir)
    texts = [c["text"] for c in corpus]
    if not texts:
        raise RuntimeError("Corpus is empty")

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    faiss.normalize_L2(emb)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    save_json(os.path.join(out_dir, "corpus.json"), corpus)
    save_json(os.path.join(out_dir, "meta.json"), {
        "embedding_model": model_name,
        "dim": int(dim),
        "n_chunks": int(index.ntotal)
    })
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    print("[OK] index saved to", os.path.join(out_dir, "index.faiss"))


# ----------------------------
# Cached runtime state (FAST)
# ----------------------------

@dataclass
class LoadedStore:
    corpus: List[Dict[str, Any]]
    meta: Dict[str, Any]
    index: Any
    model: Any


_STORE_CACHE: Dict[str, LoadedStore] = {}


def _abs_out_dir(out_dir: str) -> str:
    return str(Path(out_dir).resolve())


def load_store(out_dir: str) -> LoadedStore:
    """Load FAISS index + corpus + embedding model, cached per out_dir."""
    import faiss
    from sentence_transformers import SentenceTransformer

    out_abs = _abs_out_dir(out_dir)
    if out_abs in _STORE_CACHE:
        return _STORE_CACHE[out_abs]

    corpus_path = os.path.join(out_abs, "corpus.json")
    meta_path = os.path.join(out_abs, "meta.json")
    index_path = os.path.join(out_abs, "index.faiss")

    if not (os.path.exists(corpus_path) and os.path.exists(meta_path) and os.path.exists(index_path)):
        raise FileNotFoundError(
            f"Missing index files in {out_abs}. Need corpus.json, meta.json, index.faiss."
        )

    corpus = load_json(corpus_path)
    meta = load_json(meta_path)
    index = faiss.read_index(index_path)
    model = SentenceTransformer(meta["embedding_model"])

    store = LoadedStore(corpus=corpus, meta=meta, index=index, model=model)
    _STORE_CACHE[out_abs] = store
    return store


# ----------------------------
# Retrieval + helpers
# ----------------------------

def infer_must_title(question: str) -> str:
    q = question.lower()

    if "udhr" in q or "universal declaration" in q:
        return "Universal Declaration of Human Rights"
    if "iccpr" in q or ("civil" in q and "political" in q and "covenant" in q):
        return "International Covenant on Civil and Political Rights"
    if "icescr" in q or ("economic" in q and "social" in q and "cultural" in q and "covenant" in q):
        return "International Covenant on Economic Social and Cultural Rights"
    if "cedaw" in q:
        return "CEDAW"
    if "torture" in q or re.search(r"\bcat\b", q):
        return "Convention against Torture"
    if "genocide" in q:
        return "Convention on the Prevention and Punishment of the Crime of Genocide"

    return ""


def filter_by_title(results: List[Dict[str, Any]], must_title: str) -> List[Dict[str, Any]]:
    if not must_title:
        return results
    key = must_title.lower().strip()
    return [r for r in results if key in (r.get("title", "").lower())]


def prefer_title_rerank(results: List[Dict[str, Any]], prefer_title: str) -> List[Dict[str, Any]]:
    """Stable rerank: keep similarity but bubble preferred-title matches first."""
    if not prefer_title:
        return results
    key = prefer_title.lower().strip()
    return sorted(results, key=lambda r: (key not in r.get("title", "").lower(), -float(r.get("score", 0.0))))


def extract_article_number(question: str) -> Optional[int]:
    q = question.lower()
    m = re.search(r"\barticle\s*(\d{1,3})\b", q)
    if m:
        return int(m.group(1))
    m = re.search(r"\bart\.?\s*(\d{1,3})\b", q)
    if m:
        return int(m.group(1))
    return None


def filter_for_article(question: str, results: List[Dict[str, Any]], want_top: int, article: Optional[int] = None) -> List[Dict[str, Any]]:
    if not results:
        return results

    art = article if article is not None else extract_article_number(question)
    if art is None:
        return results[:want_top]

    a = str(art)
    needles = [
        f"article {a}",
        f"article {a}.",
        f"article {a}:",
        f"art. {a}",
        f"art {a}",
    ]

    hits = [r for r in results if any(n in r.get("text", "").lower() for n in needles)]
    if hits:
        seen = set(id(x) for x in hits)
        tail = [r for r in results if id(r) not in seen]
        return (hits + tail)[:want_top]

    return results[:want_top]


def _tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    toks = re.findall(r"[a-z0-9]+", s)
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "what", "does", "say", "about", "explain", "give", "me", "is", "are",
    }
    return [t for t in toks if t not in stop and len(t) >= 3]


def lexical_score(query: str, title: str, text: str) -> float:
    """Cheap lexical scoring to help when embeddings miss."""
    q = (query or "").lower()
    t = (title or "").lower()
    x = (text or "").lower()
    toks = _tokenize(query)

    s = 0.0
    for tok in toks:
        s += 2.0 * t.count(tok)
        s += 1.0 * x.count(tok)

    # Extra boosts for common high-value patterns
    if "freedom of expression" in q:
        if "universal declaration of human rights" in t:
            s += 80.0
        if "international covenant on civil and political rights" in t:
            s += 60.0
        if "everyone has the right to freedom of opinion and expression" in x:
            s += 120.0
        if "article 19" in x:
            s += 80.0

    art = extract_article_number(query)
    if art is not None:
        if f"article {art}" in x:
            s += 50.0

    return s


def keyword_fallback_retrieve(query: str, out_dir: str, k: int = 40) -> List[Dict[str, Any]]:
    """Fallback search across corpus when FAISS misses the right doc."""
    store = load_store(out_dir)
    scored = []
    for item in store.corpus:
        title = item.get("title", "")
        text = item.get("text", "")
        kw = lexical_score(query, title=title, text=text)
        if kw > 0:
            scored.append({
                "rank": 0,
                "score": 0.0,
                "title": title,
                "url": item.get("url", ""),
                "url_norm": item.get("url_norm") or canonicalize_url(item.get("url", "")),
                "text": text,
                "_kw": kw,
            })
    scored.sort(key=lambda r: r["_kw"], reverse=True)
    out = scored[:max(k, 10)]
    for r in out:
        r.pop("_kw", None)
    return out


def retrieve(
    query: str,
    out_dir: str,
    k: int = 5,
    prefer_title: str = "",
    must_title: str = "",
    probe_k: int = 120,
) -> List[Dict[str, Any]]:
    import faiss

    store = load_store(out_dir)
    qv = store.model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)

    k_probe = max(k, int(probe_k))
    D, I = store.index.search(qv, k_probe)

    raw: List[Dict[str, Any]] = []
    for rank, idx in enumerate(I[0]):
        if idx < 0:
            continue
        item = store.corpus[int(idx)]
        sim = float(D[0][rank])
        title = item.get("title", "")
        text = item.get("text", "")
        kw = lexical_score(query, title=title, text=text)

        # hybrid: similarity + small lexical factor
        combined = sim + 0.02 * kw

        raw.append({
            "rank": rank,
            "score": sim,
            "score_combined": combined,
            "title": title,
            "url": item.get("url", ""),
            "url_norm": item.get("url_norm") or canonicalize_url(item.get("url", "")),
            "text": text,
        })

    filtered = filter_by_title(raw, must_title) if must_title else raw

    # If the top candidates do not even mention any key words, do fallback merge
    top_text = " ".join([r.get("text", "").lower() for r in filtered[:10]])
    toks = _tokenize(query)
    misses = sum(1 for t in toks[:6] if t not in top_text)
    need_fallback = misses >= max(2, len(toks[:6]) // 2)

    merged = filtered
    if need_fallback:
        extra = keyword_fallback_retrieve(query, out_dir, k=80)
        seen = set((r["title"], r["url_norm"], (r["text"] or "")[:80]) for r in merged)
        for r in extra:
            key = (r["title"], r["url_norm"], (r["text"] or "")[:80])
            if key not in seen:
                r["score_combined"] = r["score"] + 0.02 * lexical_score(query, r["title"], r["text"])
                merged.append(r)
                seen.add(key)

    merged.sort(key=lambda r: r.get("score_combined", 0.0), reverse=True)

    # Apply prefer_title AFTER merge
    reranked = prefer_title_rerank(merged, prefer_title) if prefer_title else merged

    out = reranked[:k]
    for i, r in enumerate(out):
        r["rank"] = i
        r.pop("score_combined", None)
    return out


# ----------------------------
# Prompting + LLM
# ----------------------------

def build_context(chunks, max_chars=7000):
    parts = []
    total = 0
    for c in chunks:
        snippet = f"[{c['title']} | {c['url']}]\n{c['text']}\n\n"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "".join(parts)


def make_prompt(question, retrieved_chunks):
    context = build_context(retrieved_chunks)
    return f"""
You are a precise legal assistant specialized in international human rights law.

Use ONLY the context below to answer the question.
- If the answer is not clearly supported by the context, say: "Not sure based on the provided context."
- Do not add outside knowledge.
- Keep the answer short and direct.
- Cite the instrument name and (if possible) the article number if present in the context.

Context:
{context}

Question: {question}

Answer:
""".strip()


def call_llm(prompt: str, provider: str, model: str) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        print("[LLM ERROR] openai package not installed:", e)
        return ""

    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or ""
    ).strip() or None

    if provider == "github_models":
        token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not token:
            print("[LLM ERROR] Missing env var: GITHUB_TOKEN (or OPENAI_API_KEY fallback)")
            return ""

        client = OpenAI(
            api_key=token,
            base_url=base_url or "https://models.github.ai/inference",
            default_headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

        # GitHub Models expects "openai/<model>"
        if "/" not in model:
            model = f"openai/{model}"

    else:
        token = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not token:
            print("[LLM ERROR] Missing env var: OPENAI_API_KEY")
            return ""
        client = OpenAI(api_key=token, base_url=base_url)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise legal assistant specialized in international human rights law."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=350,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("[LLM ERROR]", repr(e))
        return ""


def answer_question(
    question: str,
    out_dir: str,
    k: int,
    use_llm: bool,
    provider: str,
    model: str,
    prefer_title: str = "",
    must_title: str = "",
    article: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:

    if not must_title:
        must_title = infer_must_title(question)

    # Probe more so article rerank has candidates
    probed = retrieve(
        query=question,
        out_dir=out_dir,
        k=max(k, 60),
        prefer_title=prefer_title,
        must_title=must_title,
        probe_k=140,
    )

    retrieved = filter_for_article(question, probed, want_top=k, article=article)

    if not retrieved:
        return "No results found in the index.", []

    baseline = retrieved[0]["text"]

    if not use_llm:
        return baseline, retrieved

    prompt = make_prompt(question, retrieved)
    ans = call_llm(prompt, provider=provider, model=model)
    if not ans:
        return baseline, retrieved

    return ans, retrieved


# ----------------------------
# CLI commands
# ----------------------------

def cmd_build(args):
    print("[BUILD] starting")
    os.makedirs(args.out_dir, exist_ok=True)
    print("[BUILD] out_dir created:", os.path.abspath(args.out_dir))

    docs = build_docs_from_csv(args.csv, args.out_dir, timeout=args.timeout, sleep_s=args.sleep)
    print("[BUILD] docs =", len(docs))

    corpus = build_corpus(docs, max_words=args.max_words, overlap=args.overlap)
    print("[BUILD] chunks =", len(corpus))

    embed_and_index(corpus, args.out_dir, model_name=args.embed_model)
    print("[BUILD] done")


def cmd_ask(args):
    print("[ASK] starting")

    ans, res = answer_question(
        question=args.question,
        out_dir=args.out_dir,
        k=args.k,
        use_llm=args.use_llm,
        provider=args.provider,
        model=args.model,
        prefer_title=args.prefer_title,
        must_title=args.must_title,
        article=args.article,
    )

    print("\nANSWER:")
    print(ans)

    print("\nSOURCES (top-k):")
    for r in res:
        print(f"{r['rank']} score={r.get('score', 0.0):.3f} | {r['title']} -> {r['url']}")


def cmd_eval(args):
    print("[EVAL] starting")
    print("[EVAL] TODO")


def main():
    load_env()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("[INFO] running:", __file__)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--csv", required=True)
    b.add_argument("--out_dir", required=True)
    b.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    b.add_argument("--timeout", type=int, default=20)
    b.add_argument("--sleep", type=float, default=0.0)
    b.add_argument("--max_words", type=int, default=400)
    b.add_argument("--overlap", type=int, default=80)
    b.set_defaults(func=cmd_build)

    a = sub.add_parser("ask")
    a.add_argument("--out_dir", required=True)
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--question", required=True)
    a.add_argument("--use_llm", action="store_true")
    a.add_argument("--provider", choices=["openai", "github_models"], default="github_models")
    a.add_argument("--model", default="openai/gpt-4o-mini")
    a.add_argument("--prefer_title", default="", help="Soft preference (rerank) if title contains this.")
    a.add_argument("--must_title", default="", help="Hard filter: only keep chunks whose title contains this.")
    a.add_argument("--article", type=int, default=None, help="Force an Article number (e.g., 19).")
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("eval")
    e.add_argument("--out_dir", required=True)
    e.add_argument("--test_set", required=True)
    e.add_argument("--k", type=int, default=5)
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


# ----------------------------
# FastMCP tool (cloud)
# ----------------------------

def create_mcp_app():
    from fastmcp import FastMCP

    m = FastMCP("Human Rights RAG")

    @m.tool()
    def rag_ask(
        question: str,
        out_dir: str = "rag_out",
        k: int = 8,
        use_llm: bool = True,
        provider: str = "github_models",
        model: str = "openai/gpt-4o-mini",
        prefer_title: str = "",
        must_title: str = "",
        article: Optional[int] = None,
    ) -> dict:
        load_env()
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        ans, res = answer_question(
            question=question,
            out_dir=out_dir,
            k=k,
            use_llm=use_llm,
            provider=provider,
            model=model,
            prefer_title=prefer_title,
            must_title=must_title,   # auto-inferred inside answer_question if empty
            article=article,
        )
        return {"answer": ans, "sources": res}

    return m


mcp = create_mcp_app()

if __name__ == "__main__":
    main()
