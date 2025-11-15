"""
Build a sharded RAG vector database from Blender documentation (current).
- Crawls Blender Python API (current) and Blender Manual (en/current)
- Extracts text, chunks, embeds with Ollama (nomic-embed-text)
- Writes shards under blender_assistant_mcp/rag_db with index.json and report.json

Run inside Blender via Preferences > RAG > Rebuild RAG Database or from CLI:
    python blender_assistant_mcp/build_rag_database.py

Optional: use local docs ZIPs to avoid crawling the web
    python blender_assistant_mcp/build_rag_database.py --api-zip path/to/api.zip --manual-zip path/to/manual.zip
"""

from __future__ import annotations
import sys, json, time, os, random
from pathlib import Path
from typing import List, Dict, Tuple
import requests
import shutil

import zipfile
import argparse

from urllib.parse import urljoin, urlparse
import urllib.robotparser as robotparser
from collections import deque
from bs4 import BeautifulSoup

# Reporting
BLOCKED_URLS: List[str] = []
FAILED_URLS: List[str] = []

# Constants
API_BASE = "https://docs.blender.org/api/current/"
MANUAL_BASE = "https://docs.blender.org/manual/en/current/"
EMBED_MODEL = "nomic-embed-text"
OUT_DIR = Path(__file__).resolve().parent / "rag_db"
TEMP_DIR = Path(__file__).resolve().parent.parent / "temp_docs"

CHUNKS_CACHE = TEMP_DIR / "chunks_cache.json"


# Map common/ollama names to HuggingFace SentenceTransformers names
def _resolve_torch_model_candidates(name: str | None) -> list[str]:
    if not name or name in {"nomic-embed-text", "nomic-ai/nomic-embed-text"}:
        # Try official Nomic HF ids, then a strong baseline fallback
        return [
            "nomic-ai/nomic-embed-text-v1",
            "nomic-ai/nomic-embed-text-v1.5",
            "BAAI/bge-base-en-v1.5",
        ]
    return [name]


def _is_allowed(rp: robotparser.RobotFileParser, url: str) -> bool:
    try:
        return rp.can_fetch('*', url)
    except Exception:
        return True


def _extract_links(html: str, page_url: str) -> List[str]:
    links: List[str] = []
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a.get('href')
            if not href:
                continue
            links.append(urljoin(page_url, href))
    except Exception:
        pass
    return links


def crawl_site(base_url: str, out_dir: Path, max_pages: int = 10000, delay: float = 0.15) -> List[Tuple[str, str, str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(base_url)
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        pass

    q = deque([base_url])
    seen = {base_url}
    results: List[Tuple[str, str, str]] = []
    base_netloc, base_path = parsed.netloc, parsed.path
    count = 0

    while q and count < max_pages:
        url = q.popleft()
        try:
            if rp and not _is_allowed(rp, url):
                BLOCKED_URLS.append(url)
                print(f"[blocked] {url}")
                continue
            r = requests.get(url, timeout=30)
            if r.status_code != 200 or 'text/html' not in r.headers.get('Content-Type', ''):
                continue
            html = r.text
            up = urlparse(url)
            path_parts = [p for p in up.path.split('/') if p]
            section = '__root__'
            if len(path_parts) >= 2:
                section = '_'.join(path_parts[:2])
            elif path_parts:
                section = path_parts[0]
            safe_name = up.path.strip('/') or 'index.html'
            if not safe_name.endswith('.html'):
                safe_name = safe_name.replace('/', '_') + '.html'
            out_file = out_dir / safe_name.replace('/', '_')
            out_file.write_text(html, encoding='utf-8')
            results.append((str(out_file), url, section))
            count += 1
            print(f"[crawl] {count}: {url}")

            for link in _extract_links(html, url):
                try:
                    lp = urlparse(link)
                    if link in seen:
                        continue
                    if lp.netloc and lp.netloc != base_netloc:
                        continue
                    if not lp.path.startswith(base_path):
                        continue
                    seen.add(link)
                    q.append(link)
                except Exception:
                    continue
            time.sleep(delay)
        except Exception:
            FAILED_URLS.append(url)
            print(f"[fail] {url}")
            continue
    print(f"✓ Crawled {count} pages from {base_url}")
    return results


def enumerate_zip_html(zip_path: str, out_dir: Path, source_label: str) -> List[Tuple[str, str, str]]:
    """Extract a docs ZIP to out_dir and enumerate all HTML files.
    Returns [(file_path, synthetic_url, section), ...] and prints one-line progress for each file.
    """
    # Clean target directory to avoid stale files
    if out_dir.exists():
        try:
            shutil.rmtree(out_dir)
        except Exception:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)

    files: List[Tuple[str, str, str]] = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(out_dir)
    except Exception as e:
        print(f"❌ Failed to extract ZIP {zip_path}: {e}")
        return files

    i = 0
    for p in sorted(out_dir.rglob('*.html')):
        try:
            if not p.is_file():
                continue
            rel = p.relative_to(out_dir).as_posix()
            parts = [part for part in rel.split('/') if part]
            section = '__root__'
            if len(parts) >= 2:
                section = '_'.join(parts[:2])
            elif parts:
                section = parts[0]
            url = f"local_zip:{source_label}/{rel}"
            files.append((str(p), url, section))
            i += 1
            print(f"[zip] {i}: {url}")
        except Exception:
            continue
    print(f"✓ Enumerated {i} HTML files from {zip_path}")
    return files


def extract_text_from_html(html_file: str) -> str:
    """Legacy simple extractor (kept for compatibility)."""
    soup = BeautifulSoup(Path(html_file).read_text(encoding='utf-8'), 'html.parser')
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(" ")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)


def extract_filtered_text(html: str, page_type: str, is_api: bool) -> str:
    """Aggressively remove UI/chrome and keep main content.
    Applies light page-type-specific hints for API/Manual.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Drop obvious chrome
    for sel in [
        'header', 'footer', 'nav', 'aside',
        '.sphinxsidebar', '.bd-sidebar', '.bd-toc', '.toc', '.toctree-wrapper',
        '.related', '.sidebar', '#searchbox', '.wy-side-nav-search', '.navbar',
        '.breadcrumbs', '.bd-breadcrumbs', '.theme-toggle', '.theme-switch'
    ]:
        for el in soup.select(sel):
            el.decompose()

    # Drop generic navigation roles
    for el in soup.find_all(attrs={"role": lambda v: v in ("navigation", "complementary")}):
        try:
            el.decompose()
        except Exception:
            pass

    # Prefer a main/content container if present
    content = None
    for sel in ['main', 'div.bd-content', 'article', 'div.document', 'div.body']:
        content = soup.select_one(sel)
        if content:
            break
    root = content if content else soup

    # For API, ensure we keep sections like Parameters/Returns/Attributes/Methods
    if is_api:
        keep_sections = []
        headings = root.find_all(['h2', 'h3'])
        wanted = {"parameters", "returns", "attributes", "properties", "methods", "examples"}
        for h in headings:
            name = (h.get_text(" ") or "").strip().lower()
            if any(w in name for w in wanted):
                # Include heading and sibling block until next heading of same level
                sec = [h]
                sib = h.find_next_sibling()
                while sib and sib.name not in ('h2','h3','h1'):
                    sec.append(sib)
                    sib = sib.find_next_sibling()
                frag = BeautifulSoup("<div></div>", 'html.parser')
                container = frag.find('div') or frag
                if hasattr(container, 'append'):
                    for node in sec:
                        try:
                            container.append(node)
                        except Exception:
                            pass
                    keep_sections.append(container)
        # Build a minimal doc of title + kept sections if we detected any
        if keep_sections:
            title = root.find(['h1'])
            frag = BeautifulSoup("<div></div>", 'html.parser')
            container = frag.find('div') or frag
            if title and hasattr(container, 'append'):
                try:
                    container.append(title)
                except Exception:
                    pass
            for ks in keep_sections:
                try:
                    container.append(ks)
                except Exception:
                    pass
            root = container

    text = root.get_text(" ")
    # Normalize whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return ' '.join(chunk for chunk in chunks if chunk)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    out: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if end < n:
            cut = max(chunk.rfind('.'), chunk.rfind('\n'))
            if cut > chunk_size * 0.5:
                end = start + cut + 1
                chunk = text[start:end]
        out.append(chunk.strip())
        start = max(end - overlap, end)
    return out

# --- Sampling and page analysis utilities ---

def _analyze_page_features(html: str) -> Dict:
    """Heuristically classify a Blender docs page and extract structural features.
    Returns a dict with: page_type, has_nav, has_sidebar, has_breadcrumbs, has_toc,
    has_parameters, has_returns, has_attributes, has_methods, has_examples,
    has_code_blocks, h1,h2,h3 counts, tables count.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')

    def has(sel: str) -> bool:
        try:
            return bool(soup.select_one(sel))
        except Exception:
            return False

    text = soup.get_text(" ") if soup else ""
    lower = text.lower()
    # Signals
    has_params = ("parameters" in lower) or bool(soup.select('dt:contains("Parameters")'))
    has_returns = ("returns" in lower) or ("return type" in lower)
    has_examples = ("example" in lower) or bool(soup.select('div.highlight, pre, code'))

    # Headings and tables
    h1 = len(soup.find_all('h1'))
    h2 = len(soup.find_all('h2'))
    h3 = len(soup.find_all('h3'))
    tables = len(soup.find_all('table'))

    # API vs Manual quick heuristic
    body_text = lower
    is_api = ("bpy.ops." in body_text) or ("bpy.types." in body_text) or ("bmesh.ops" in body_text) or ("mathutils" in body_text)

    # API subtypes
    page_type = "manual_article"
    if is_api:
        if "bpy.ops." in body_text or "bmesh.ops" in body_text:
            page_type = "api_operator"
        elif "bpy.types." in body_text:
            page_type = "api_type"
        else:
            page_type = "api_module"
    else:
        # Manual special cases
        if "index" in lower[:200] or has('div.toctree-wrapper'):
            page_type = "manual_index"
        elif has_params or has_examples:
            page_type = "manual_reference"
        else:
            page_type = "manual_article"

    return {
        "page_type": page_type,
        "has_nav": has('nav, header'),
        "has_sidebar": has('aside, .sphinxsidebar, .bd-sidebar'),
        "has_breadcrumbs": has('.breadcrumbs, .bd-breadcrumbs'),
        "has_toc": has('.toctree-wrapper, nav[role="navigation"] .toc'),
        "has_parameters": has_params,
        "has_returns": has_returns,
        "has_attributes": ("attributes" in lower) or ("properties" in lower),
        "has_methods": ("methods" in lower) or ("functions" in lower),
        "has_examples": has_examples,
        "has_code_blocks": bool(soup.find_all(['pre','code','div'], class_=['highlight'])),
        "h1": h1, "h2": h2, "h3": h3, "tables": tables,
    }


def run_sample_report(api_zip: str | None, manual_zip: str | None, size_api: int, size_manual: int, out_path: Path = TEMP_DIR / 'sample_report.json') -> Path:
    """Enumerate docs, sample N pages per source, analyze, and write a JSON report."""
    api_dir = TEMP_DIR / "api"; manual_dir = TEMP_DIR / "manual"
    api_files = enumerate_zip_html(api_zip, api_dir, "API") if api_zip else crawl_site(API_BASE, api_dir)
    manual_files = enumerate_zip_html(manual_zip, manual_dir, "Manual") if manual_zip else crawl_site(MANUAL_BASE, manual_dir)

    import random as _rand
    def _sample(files, k):
        return _rand.sample(files, min(k, len(files)))

    api_sample = _sample(api_files, size_api)
    manual_sample = _sample(manual_files, size_manual)

    def _analyze_list(lst, label):
        out = []
        for (path, url, section) in lst:
            try:
                html = Path(path).read_text(encoding='utf-8')
                feats = _analyze_page_features(html)
                # Refine page_type using filename/url hints for API vs Manual
                base = Path(path).name.lower()
                u = (url or '').lower()
                is_api = ('local_zip:api/' in u) or ('/api/' in u)
                if is_api:
                    if base in ('index.html', 'search.html', 'genindex.html', 'py-modindex.html'):
                        feats['page_type'] = 'api_index'
                    elif 'bpy.ops.' in base or 'bmesh.ops' in base:
                        feats['page_type'] = 'api_operator'
                    elif 'bpy.types.' in base:
                        feats['page_type'] = 'api_type'
                    else:
                        feats['page_type'] = feats.get('page_type','api_module')
                else:
                    if base in ('index.html', 'search.html', 'genindex.html', 'py-modindex.html'):
                        feats['page_type'] = 'manual_index'
                # Include a short excerpt of the filtered content for inspection
                try:
                    excerpt = extract_filtered_text(html, feats.get('page_type','manual_article'), is_api)
                    excerpt = (excerpt or "").strip()[:600]
                except Exception:
                    excerpt = ""
                out.append({
                    "file": Path(path).name,
                    "url": url,
                    "section": section,
                    **feats,
                    "excerpt": excerpt,
                })
            except Exception as e:
                out.append({"file": Path(path).name, "url": url, "section": section, "error": str(e)})
        return out

    api_report = _analyze_list(api_sample, "API")
    manual_report = _analyze_list(manual_sample, "Manual")

    def _summary(items):
        from collections import Counter
        c = Counter(x.get("page_type","?") for x in items)
        return dict(c)

    report = {
        "summary": {
            "api_counts": _summary(api_report),
            "manual_counts": _summary(manual_report),
            "api_sample_size": len(api_report),
            "manual_sample_size": len(manual_report),
        },
        "api": api_report,
        "manual": manual_report,
    }

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"✓ Wrote sample report: {out_path}")
    print(f"API types: {report['summary']['api_counts']}")
    print(f"Manual types: {report['summary']['manual_counts']}")
    return out_path



def generate_embeddings_ollama(texts: List[str], model: str = EMBED_MODEL, progress_interval: int = 25, verbose: bool = False) -> List[List[float]]:
    print(f"=== Generating embeddings with Ollama ({model}) ===")
    embs: List[List[float]] = []
    total = len(texts)
    for i, t in enumerate(texts, 1):
        if verbose or (progress_interval > 0 and (i % progress_interval == 0 or i == total)):
            print(f"[embed] {i}/{total}")
        try:
            resp = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": t},
                timeout=45,
            )
            resp.raise_for_status()
            embs.append(resp.json().get('embedding', [0.0] * 768))
        except Exception as e:
            print(f"  ⚠ embedding failed @ {i}: {e}")
            embs.append([0.0] * 768)
    return embs


def generate_embeddings_torch(
    texts: List[str],
    model_name: str = "nomic-embed-text",
    batch_size: int = 128,
    normalize: bool = True,
    verbose: bool = False,
    progress_interval: int = 1,
) -> List[List[float]]:
    """Generate embeddings using SentenceTransformers with optional CUDA.
    Falls back to CPU if CUDA not available. Prints progress per batch.
    Tries Nomic aliases and falls back to a strong baseline if needed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except Exception as e:
        raise RuntimeError(
            "SentenceTransformers not available. Install with `pip install sentence-transformers`"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve candidate HF model ids
    candidates = _resolve_torch_model_candidates(model_name)
    last_err: Exception | None = None
    model = None
    for cand in candidates:
        try:
            if verbose:
                print(f"[embed:init] trying {cand} on {device}")
            # Some models (e.g., nomic-ai/nomic-embed-text-*) require trust_remote_code
            model = SentenceTransformer(cand, device=device, trust_remote_code=True)
            model_name = cand
            break
        except Exception as e:
            last_err = e
            if verbose:
                print(f"  ⚠ failed to load {cand}: {e}")
    if model is None:
        raise RuntimeError(f"Could not load any embedding model from candidates: {candidates}. Last error: {last_err}")

    if verbose:
        print(f"=== Generating embeddings with SentenceTransformers ({model_name}) on {device} ===")

    embs: List[List[float]] = []
    total = len(texts)
    cpu_fallback = False
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        try:
            batch_embs = model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
            embs.extend(e.tolist() for e in batch_embs)
        except Exception as e:
            msg = str(e)
            # Handle common CUDA-arch mismatch by falling back to CPU and retrying once
            if (not cpu_fallback) and ("no kernel image is available" in msg.lower() or "cuda error" in msg.lower()):
                if verbose:
                    print("  ⚠ CUDA kernel error detected; falling back to CPU for remaining batches")
                try:
                    # Re-initialize model on CPU and retry this batch
                    model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
                    cpu_fallback = True
                    batch_embs = model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize,
                        show_progress_bar=False,
                    )
                    embs.extend(e.tolist() for e in batch_embs)
                except Exception as e2:
                    zeros = [0.0] * (len(embs[0]) if embs else 768)
                    embs.extend([zeros] * len(batch))
                    print(f"  ⚠ torch embedding failed on CPU retry for batch starting @{i}: {e2}")
            else:
                # On failure with no viable fallback, pad zeros for this batch
                zeros = [0.0] * (len(embs[0]) if embs else 768)
                embs.extend([zeros] * len(batch))
                print(f"  ⚠ torch embedding failed for batch starting @{i}: {e}")
        done = min(i + len(batch), total)
        if verbose or (progress_interval > 0 and (done % progress_interval == 0 or done == total)):
            print(f"[embed] {done}/{total}")
    return embs



def create_sharded_database(documents: List[Dict], out_dir: Path = OUT_DIR, verbose: bool = False) -> Path:
    print("=== Writing sharded JSON database ===")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Group by source + section
    shards: Dict[str, Dict[str, List]] = {}
    for doc in documents:
        md = doc['metadata']
        # Normalize source if unknown using URL hint
        src = md.get('source', 'Unknown')
        if src == 'Unknown':
            u = (md.get('url') or '').lower()
            if 'local_zip:api/' in u or '/api/' in u:
                md['source'] = 'API'
            elif 'local_zip:manual/' in u or '/manual/' in u:
                md['source'] = 'Manual'
        source = md.get('source', 'Unknown')
        section = md.get('section', md.get('file', '__root__'))
        key = f"{source}_{section}".replace('/', '_').replace(' ', '_')
        b = shards.setdefault(key, {"documents": [], "embeddings": []})
        b["documents"].append({"content": doc["content"], "metadata": md})
        b["embeddings"].append(doc["embedding"])

    index = {"version": 1, "total_documents": len(documents), "shards": []}
    for name, data in shards.items():
        p = out_dir / f"{name}.json"
        p.write_text(json.dumps(data), encoding='utf-8')
        index["shards"].append({"name": name, "documents": len(data['documents']), "path": p.name})
        if verbose:
            print(f"[shard] {name}: {len(data['documents'])} -> {p.name}")

    (out_dir / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    report = {"blocked_urls": BLOCKED_URLS, "failed_urls": FAILED_URLS}
    (out_dir / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    print(f"✓ Shards: {len(shards)} | Docs: {len(documents)} | Index: {out_dir / 'index.json'}")
    return out_dir / 'index.json'


def build(
    api_zip: str | None = None,
    manual_zip: str | None = None,
    reuse_chunks: bool = False,
    verbose: bool = False,
    embed_interval: int = 25,
    process_interval: int = 200,
    embed_backend: str = "ollama",
    embed_model: str | None = None,
    embed_batch_size: int = 128,
    include_types: str = "",
    exclude_types: str = "manual_index",
) -> Path:
    print("=" * 60)
    print("Blender RAG Database Builder (current)")
    print("=" * 60)

    # Acquire docs: from ZIPs if provided, otherwise crawl
    api_dir = TEMP_DIR / "api"; manual_dir = TEMP_DIR / "manual"
    if api_zip:
        print(f"=== Using API ZIP: {api_zip}")
        api_files = enumerate_zip_html(api_zip, api_dir, "API")
    else:
        api_files = crawl_site(API_BASE, api_dir)

    if manual_zip:
        print(f"=== Using Manual ZIP: {manual_zip}")
        manual_files = enumerate_zip_html(manual_zip, manual_dir, "Manual")
    else:
        manual_files = crawl_site(MANUAL_BASE, manual_dir)

    all_files = api_files + manual_files
    if not all_files:
        raise RuntimeError("No documentation sources found (ZIPs empty or crawl returned none)")

    # Extract + chunk (or reuse)
    documents: List[Dict] = []
    if reuse_chunks and (CHUNKS_CACHE.exists()):
        print("=== Reusing chunk cache ===")
        try:
            documents = json.loads(CHUNKS_CACHE.read_text(encoding='utf-8'))
            # Require page_type in cache; otherwise force reprocess
            if not documents or not all(isinstance(d, dict) and isinstance(d.get('metadata', {}), dict) and d['metadata'].get('page_type') for d in documents[:50]):
                print("  ⚠ cached chunks missing page_type; ignoring cache and reprocessing with new filters")
                documents = []
            else:
                print(f"✓ Loaded {len(documents)} cached chunks")
        except Exception as e:
            print(f"  ⚠ failed to load chunk cache, falling back to re-chunking: {e}")

    if not documents:
        print("=== Processing documentation ===")
        inc = set([t.strip() for t in (include_types or '').split(',') if t.strip()])
        exc = set([t.strip() for t in (exclude_types or '').split(',') if t.strip()])
        total_chunks = 0
        for (file_path, url, section) in all_files:
            try:
                html = Path(file_path).read_text(encoding='utf-8')
                feats = _analyze_page_features(html)
                base = Path(file_path).name.lower()
                u = (url or '').lower()
                is_api = ('local_zip:api/' in u) or ('/api/' in u)
                # Refine classification
                if is_api:
                    if base in ('index.html', 'search.html', 'genindex.html', 'py-modindex.html'):
                        feats['page_type'] = 'api_index'
                    elif 'bpy.ops.' in base or 'bmesh.ops' in base:
                        feats['page_type'] = 'api_operator'
                    elif 'bpy.types.' in base:
                        feats['page_type'] = 'api_type'
                    else:
                        feats['page_type'] = feats.get('page_type','api_module')
                else:
                    if base in ('index.html', 'search.html', 'genindex.html', 'py-modindex.html'):
                        feats['page_type'] = 'manual_index'
                page_type = feats.get('page_type', 'manual_article')

                # Filtering include/exclude
                if page_type in exc:
                    if verbose:
                        print(f"[skip] {base} excluded (type={page_type})")
                    continue
                if inc and (page_type not in inc):
                    if verbose:
                        print(f"[skip] {base} not in include set (type={page_type})")
                    continue

                txt = extract_filtered_text(html, page_type, is_api)
                if not txt or len(txt.strip()) < 40:
                    if verbose:
                        print(f"[skip] {base} produced too little content after filtering")
                    continue
                chunks = chunk_text(txt, 1000, 200)
                source = "API" if is_api else "Manual"
                filename = Path(file_path).name
                if verbose:
                    print(f"[process] {filename}: {len(chunks)} chunks (type={page_type})")
                for i, ch in enumerate(chunks):
                    documents.append({
                        "content": ch,
                        "metadata": {"source": source, "file": filename, "url": url, "section": section, "chunk": i, "page_type": page_type},
                        "embedding": None,
                    })
                    total_chunks += 1
                    if verbose or (process_interval > 0 and total_chunks % process_interval == 0):
                        print(f"[chunk] {total_chunks}")
            except Exception as e:
                print(f"  ⚠ process failed for {file_path}: {e}")
        print(f"✓ Created {len(documents)} chunks")
        # Write cache to speed up retries
        try:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            CHUNKS_CACHE.write_text(json.dumps(documents), encoding='utf-8')
            if verbose:
                print(f"[cache] wrote {CHUNKS_CACHE} ({len(documents)} entries)")
        except Exception as e:
            print(f"  ⚠ failed to write chunk cache: {e}")

    # Embed
    texts = [d["content"] for d in documents]
    if embed_backend == "torch":
        torch_model = embed_model or "nomic-embed-text"
        embs = generate_embeddings_torch(
            texts,
            model_name=torch_model,
            batch_size=embed_batch_size,
            normalize=True,
            verbose=verbose,
            progress_interval=embed_interval,
        )
    else:
        ollama_model = embed_model or EMBED_MODEL
        embs = generate_embeddings_ollama(
            texts,
            model=ollama_model,
            progress_interval=embed_interval,
            verbose=verbose,
        )
    for d, e in zip(documents, embs):
        d["embedding"] = e

    # Write shards


    idx = create_sharded_database(documents, OUT_DIR, verbose=verbose)

    print("=" * 60)
    print("✓ RAG Database Build Complete")
    print("=" * 60)
    print(f"Index: {idx}")
    print(f"Report: {OUT_DIR / 'report.json'}")
    return idx


def main():
    parser = argparse.ArgumentParser(description="Build sharded RAG DB from Blender docs")
    parser.add_argument("--api-zip", type=str, default=None, help="Path to API docs zip (optional)")
    parser.add_argument("--manual-zip", type=str, default=None, help="Path to Manual docs zip (optional)")
    parser.add_argument("--verbose", action="store_true", help="Verbose progress (per-file/per-shard); combine with intervals below")
    parser.add_argument("--embed-interval", type=int, default=25, help="Embedding progress interval (set 1 for per-item)")
    parser.add_argument("--process-interval", type=int, default=200, help="Chunk progress interval (set 1 for per-chunk)")
    parser.add_argument("--embed-backend", choices=["ollama", "torch"], default="ollama", help="Embedding backend: ollama (default) or torch (CUDA capable)")
    parser.add_argument("--embed-model", type=str, default=None, help="Model name: Ollama model (ollama) or HF model id (torch). Defaults to nomic-embed-text")
    parser.add_argument("--embed-batch-size", type=int, default=128, help="Batch size for torch backend")
    parser.add_argument("--reuse-chunks", action="store_true", help="Reuse cached chunks from temp_docs/chunks_cache.json to skip re-chunking")
    # Filtering controls
    parser.add_argument("--include-types", type=str, default="", help="Comma-separated page types to include (api_operator,api_type,api_module,api_index,manual_reference,manual_article,manual_index). Empty=all")
    parser.add_argument("--exclude-types", type=str, default="manual_index,api_index", help="Comma-separated page types to exclude. Default: manual_index,api_index")
    # Sampling/report flags
    parser.add_argument("--sample-report", action="store_true", help="Generate a sample analysis report (no embedding) and exit")
    parser.add_argument("--sample-size-api", type=int, default=100, help="Number of API pages to sample (default 100)")
    parser.add_argument("--sample-size-manual", type=int, default=100, help="Number of Manual pages to sample (default 100)")

    args = parser.parse_args()

    if args.sample_report:
        try:
            run_sample_report(args.api_zip, args.manual_zip, args.sample_size_api, args.sample_size_manual)
            return
        except Exception as e:
            print(f"❌ Sample report failed: {e}")
            sys.exit(1)
    try:
        build(
            api_zip=args.api_zip,
            manual_zip=args.manual_zip,
            reuse_chunks=args.reuse_chunks,
            verbose=args.verbose,
            embed_interval=args.embed_interval,
            process_interval=args.process_interval,
            embed_backend=args.embed_backend,
            embed_model=args.embed_model,
            embed_batch_size=args.embed_batch_size,
            include_types=args.include_types,
            exclude_types=args.exclude_types,
        )
    except Exception as e:
        print(f"❌ Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

