#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse
import argparse
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from pathlib import Path

# Load env vars from human_rights_crew/.env (relative to this file)
ENV_PATH = Path(__file__).resolve().parent / "human_rights_crew" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

NOISE_PATTERNS = [
    "This site was archived on 2023-02-01 and is no longer receiving updates",
    "Links, accessibility, and other functionality may be limited.",
    "University of Minnesota Human Rights Library",
    "Human Rights Library",
]

DROP_LINE_PREFIXES = {"home", "search", "contact", "copyright"}


def canonicalize_url(url: str) -> str:
    """Normalize URLs so evaluation doesn't fail on tiny differences."""
    try:
        p = urlparse(url.strip())
        scheme = "https"
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = p.path or ""
        if path.endswith("/") and path != "/":
            path = path[:-1]
        return urlunparse((scheme, netloc, path, "", p.query or "", ""))
    except Exception:
        return url.strip()


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

    # Keep newlines so "Article X" structure survives better
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


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cache_path(out_dir: str, url: str) -> str:
    ensure_dir(os.path.join(out_dir, "cache"))
    return os.path.join(out_dir, "cache", f"{sha1(url)}.txt")


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    save_json(os.path.join(out_dir, "meta.json"), {"embedding_model": model_name, "dim": int(dim), "n_chunks": int(index.ntotal)})
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))

    print("[OK] index saved to", os.path.join(out_dir, "index.faiss"))


def load_index(out_dir: str):
    import faiss
    corpus = load_json(os.path.join(out_dir, "corpus.json"))
    meta = load_json(os.path.join(out_dir, "meta.json"))
    index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
    return corpus, meta, index


def retrieve(query: str, out_dir: str, k: int = 5):
    from sentence_transformers import SentenceTransformer
    import faiss

    corpus, meta, index = load_index(out_dir)
    model = SentenceTransformer(meta["embedding_model"])

    q = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q)
    D, I = index.search(q, k)

    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0:
            continue
        item = corpus[int(idx)]
        results.append({
            "rank": rank,
            "score": float(D[0][rank]),
            "title": item["title"],
            "url": item["url"],
            "url_norm": item["url_norm"],
            "text": item["text"],
        })
    return results


def eval_retrieval(test_set_path: str, out_dir: str, k: int = 5):
    test_set = load_json(test_set_path)
    success = 0
    rrs = []
    precisions = []

    for ex in test_set:
        q = ex["question"]
        gold = ex.get("source_urls", ex.get("source_url"))
        if isinstance(gold, str):
            gold = [gold]
        gold_norm = {canonicalize_url(u) for u in (gold or [])}

        results = retrieve(q, out_dir=out_dir, k=k)
        rel = [i for i, r in enumerate(results) if r["url_norm"] in gold_norm]

        if rel:
            success += 1
            rrs.append(1.0 / (rel[0] + 1))
        else:
            rrs.append(0.0)

        denom = max(1, min(k, len(results)))
        precisions.append(len(rel) / denom)

    n = len(test_set)
    print(f"Questions    : {n}")
    print(f"Success@{k}  : {success/n:.3f}")
    print(f"MRR          : {sum(rrs)/n:.3f}")
    print(f"Precision@{k}: {sum(precisions)/n:.3f}")


def build_context(chunks, max_chars=5000):
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
- Cite the instrument name and (if possible) the article number if present in the context.

Context:
{context}

Question: {question}

Answer:
""".strip()

def call_llm(prompt: str, provider: str, model: str) -> str:
    """
    provider:
      - "openai"        -> uses OPENAI_API_KEY
      - "github_models" -> uses GITHUB_TOKEN and base_url=https://models.github.ai/inference
    """
    try:
        from openai import OpenAI
    except Exception as e:
        print("[LLM ERROR] openai package not installed:", e)
        return ""

    if provider == "github_models":
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            print("[LLM ERROR] Missing env var: GITHUB_TOKEN")
            return ""
        client = OpenAI(
            api_key=token,
            base_url="https://models.github.ai/inference",
            default_headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
    else:
        token = os.environ.get("OPENAI_API_KEY", "").strip()
        if not token:
            print("[LLM ERROR] Missing env var: OPENAI_API_KEY")
            return ""
        client = OpenAI(api_key=token)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise legal assistant specialized in international human rights law."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        msg = resp.choices[0].message
        content = getattr(msg, "content", None)
        return (content or "").strip()
    except Exception as e:
        print("[LLM ERROR]", repr(e))
        return ""

def answer_question(question: str, k: int, out_dir: str, use_llm: bool, provider: str, model: str):
    retrieved = retrieve(question, out_dir=out_dir, k=k)
    if not retrieved:
        return "No results found in the index.", []

    # retrieval-only baseline
    baseline = retrieved[0]["text"]

    if not use_llm:
        return baseline, retrieved

    prompt = make_prompt(question, retrieved)
    ans = call_llm(prompt, provider=provider, model=model)
    if not ans:
        # fallback if LLM fails
        return baseline, retrieved

    return ans, retrieved



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
        args.question,
        out_dir=args.out_dir,
        k=args.k,
        use_llm=args.use_llm,
        provider=args.provider,
        model=args.model
    )

    print("\nANSWER:")
    print(ans)

    print("\nSOURCES (top-k):")
    for r in res:
        print(f"{r['rank']} score={r['score']:.3f} | {r['title']} -> {r['url']}")


def cmd_eval(args):
    print("[EVAL] starting")
    # eval_retrieval(args.test_set, out_dir=args.out_dir, k=args.k)

def main():
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
    b.set_defaults(func=cmd_build)   # IMPORTANT

    a = sub.add_parser("ask")
    a.add_argument("--out_dir", required=True)
    a.add_argument("--k", type=int, default=5)
    a.add_argument("--question", required=True)
    a.add_argument("--use_llm", action="store_true")
    a.add_argument("--provider", choices=["openai", "github_models"], default="github_models")
    a.add_argument("--model", default="openai/gpt-4o-mini")
    a.set_defaults(func=cmd_ask)


    e = sub.add_parser("eval")
    e.add_argument("--out_dir", required=True)
    e.add_argument("--test_set", required=True)
    e.add_argument("--k", type=int, default=5)
    e.set_defaults(func=cmd_eval)    # IMPORTANT

    args = ap.parse_args()
    args.func(args)                  # IMPORTANT (this runs build/ask/eval)


def create_mcp_app():
    from fastmcp import FastMCP

    m = FastMCP("Human Rights RAG")

    @m.tool()
    def rag_ask(
        question: str,
        out_dir: str = "rag_out",
        k: int = 5,
        use_llm: bool = True,
        provider: str = "github_models",
        model: str = "openai/gpt-4o-mini",
    ) -> dict:
        ans, res = answer_question(
            question=question,
            out_dir=out_dir,
            k=k,
            use_llm=use_llm,
            provider=provider,
            model=model,
        )
        return {"answer": ans, "sources": res}

    return m

mcp = create_mcp_app()



if __name__ == "__main__":
    main()


