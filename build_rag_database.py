"""
Build RAG vector database from Blender documentation.

This script:
1. Downloads Blender Python API documentation
2. Downloads Blender manual
3. Chunks the documentation
4. Generates embeddings using Ollama
5. Creates ChromaDB vector database
6. Saves to blender_assistant_mcp/blender_docs_db/

Run this before building the extension to include RAG support.
"""

import json
import os
import sys
import time
import urllib.robotparser as robotparser
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

# --- New full-site crawler helpers (API + Manual, "current") ---
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BLOCKED_URLS = []
FAILED_URLS = []


def _is_same_site(url: str, base_netloc: str, base_path: str) -> bool:
    try:
        p = urlparse(url)
        if p.netloc and p.netloc != base_netloc:
            return False
        # Constrain to subtree path
        return p.path.startswith(base_path)
    except Exception:
        return False


def _extract_links(html: str, page_url: str) -> List[str]:
    links = []
    try:
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue
            abs_url = urljoin(page_url, href)
            links.append(abs_url)
    except Exception:
        pass
    return links


def crawl_site(
    base_url: str,
    out_dir: Path,
    source_label: str,
    max_pages: int = 10000,
    delay: float = 0.2,
):
    """BFS crawl under base_url, saving HTML pages to out_dir.

    Returns list of tuples: (saved_file_path:str, url:str, section:str)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(base_url)
    rp = robotparser.RobotFileParser()
    rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        # If robots can't be read, proceed cautiously and still rate-limit
        pass

    q = deque([base_url])
    seen = set([base_url])
    results = []
    count = 0

    while q and count < max_pages:
        url = q.popleft()
        try:
            # Respect robots.txt if available
            if rp and hasattr(rp, "can_fetch") and not rp.can_fetch("*", url):
                BLOCKED_URLS.append(url)
                continue

            headers = {
                "User-Agent": "BlenderAssistant-RAG/1.0 (+https://github.com/ChrisWilliamson11/blender-local-mcp)"
            }
            r = requests.get(url, timeout=30, headers=headers)

            if r.status_code != 200 or "text/html" not in r.headers.get(
                "Content-Type", ""
            ):
                continue

            html = r.text
            # Derive a simple section from path segments for sharding
            up = urlparse(url)
            path_parts = [p for p in up.path.split("/") if p]
            section = "__root__"
            if len(path_parts) >= 2:
                section = "_".join(path_parts[0:2])  # e.g., api/current or manual/en
            elif path_parts:
                section = path_parts[0]

            # Save file (stable naming)
            safe_name = up.path.strip("/")
            if not safe_name:
                safe_name = "index.html"
            if not safe_name.endswith(".html"):
                safe_name = safe_name.replace("/", "_") + ".html"
            out_file = out_dir / safe_name.replace("/", "_")
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(html)

            results.append((str(out_file), url, section))

            count += 1

            if count % 10 == 0:
                print(
                    f"[crawl] {source_label}: {count} pages, queue {len(q)}, failures {len(FAILED_URLS)} (last: {url})"
                )

            # Enqueue in-domain links under base path

            base_netloc = parsed.netloc
            base_path = parsed.path
            for link in _extract_links(html, url):
                if link in seen:
                    continue
                if _is_same_site(link, base_netloc, base_path):
                    seen.add(link)
                    q.append(link)

            time.sleep(delay)
        except Exception:
            FAILED_URLS.append(url)
            continue

    print(f"✓ Crawled {count} pages from {base_url}")
    if BLOCKED_URLS:
        print(f"⚠ Robots disallowed {len(BLOCKED_URLS)} URLs (see rag_db/report.json)")
    if FAILED_URLS:
        print(f"⚠ Failed to fetch {len(FAILED_URLS)} URLs (see rag_db/report.json)")
    return results


def download_blender_api_docs(api_zip_url: Optional[str] = None):
    """Fetch Blender Python API docs as a zip if available, else crawl.

    Returns list of tuples: (file_path, url, section)

    """

    print("\n=== Downloading Blender Python API Documentation (current) ===")

    docs_dir = Path("temp_docs/api")

    docs_dir.mkdir(parents=True, exist_ok=True)
    # (Removed erroneous Manual zip block from API downloader)

    # Prefer explicit API zip if provided
    if api_zip_url:
        try:
            import zipfile
            from io import BytesIO

            headers = {"User-Agent": "BlenderAssistant-RAG/1.0 (+repo)"}
            print(f"[zip] Using API zip: {api_zip_url} (downloading...)")
            import requests

            with requests.get(
                api_zip_url, stream=True, timeout=300, headers=headers
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", "0") or "0")
                buf = BytesIO()
                chunk = 1024 * 256
                downloaded = 0
                for part in resp.iter_content(chunk_size=chunk):
                    if part:
                        buf.write(part)
                        downloaded += len(part)
                        if total and downloaded % (chunk * 8) == 0:
                            pct = downloaded * 100 // total
                            print(
                                f"[zip] API download {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct}%)"
                            )
                data = buf.getvalue()
            print("[zip] Extracting API zip...")
            with zipfile.ZipFile(BytesIO(data)) as zf:
                zf.extractall(docs_dir)
            results = []
            for p in docs_dir.rglob("*.html"):
                rel = p.relative_to(docs_dir)
                parts = [s for s in rel.parts if s]
                section = "__root__"
                if len(parts) >= 2:
                    section = "_".join(parts[:2])
                elif parts:
                    section = parts[0]
                results.append((str(p), None, section))
            print(f"✓ Downloaded and extracted API zip, pages: {len(results)}")
            return results
        except Exception as e:
            print(f"[zip] Provided API zip failed, falling back: {e}")

    # Try to locate a ZIP link on the API index page, download, and extract.
    try:
        import re
        import zipfile
        from io import BytesIO

        import requests

        index_url = "https://docs.blender.org/api/current/"
        headers = {"User-Agent": "BlenderAssistant-RAG/1.0 (+repo)"}
        r = requests.get(index_url, timeout=30, headers=headers)
        r.raise_for_status()
        html = r.text

        # Heuristic: find any .zip link on the API index page
        m = re.search(r'href="([^"]+\\.zip)"', html, flags=re.IGNORECASE)
        if m:
            zip_url = m.group(1)
            if not zip_url.startswith("http"):
                from urllib.parse import urljoin

                zip_url = urljoin(index_url, zip_url)
            print(f"[zip] Found API zip: {zip_url} (downloading...)")

            with requests.get(
                zip_url, stream=True, timeout=120, headers=headers
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", "0") or "0")
                buf = BytesIO()
                chunk = 1024 * 256
                downloaded = 0
                for part in resp.iter_content(chunk_size=chunk):
                    if part:
                        buf.write(part)
                        downloaded += len(part)
                        if total and downloaded % (chunk * 8) == 0:
                            pct = downloaded * 100 // total
                            print(
                                f"[zip] API download {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct}%)"
                            )
                data = buf.getvalue()

            print("[zip] Extracting API zip...")
            with zipfile.ZipFile(BytesIO(data)) as zf:
                zf.extractall(docs_dir)

            # Collect extracted HTML files
            results = []
            for p in docs_dir.rglob("*.html"):
                # Section: derive from relative path (first two segments)
                rel = p.relative_to(docs_dir)
                parts = [s for s in rel.parts if s]
                section = "__root__"
                if len(parts) >= 2:
                    section = "_".join(parts[:2])
                elif parts:
                    section = parts[0]
                results.append((str(p), None, section))

            print(f"✓ Downloaded and extracted API zip, pages: {len(results)}")
            return results
        else:
            print("[zip] No API zip link found on index, falling back to crawl.")
    except Exception as e:
        print(f"[zip] API zip download failed, falling back to crawl: {e}")

    # Fallback to crawling
    base_url = "https://docs.blender.org/api/current/"
    results = crawl_site(base_url, docs_dir, source_label="API")
    print(f"✓ Downloaded {len(results)} API pages (crawl)")
    return results


def download_blender_manual(manual_zip_url: Optional[str] = None):
    """Fetch Blender Manual (current, en) as a zip if available, else crawl.

    Returns list of tuples: (file_path, url, section)

    """

    print("\n=== Downloading Blender Manual (current, en) ===")

    docs_dir = Path("temp_docs/manual")

    docs_dir.mkdir(parents=True, exist_ok=True)

    # Prefer explicit Manual zip if provided
    if manual_zip_url:
        try:
            import zipfile
            from io import BytesIO

            headers = {"User-Agent": "BlenderAssistant-RAG/1.0 (+repo)"}
            print(f"[zip] Using Manual zip: {manual_zip_url} (downloading...)")
            import requests

            with requests.get(
                manual_zip_url, stream=True, timeout=300, headers=headers
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", "0") or "0")
                buf = BytesIO()
                chunk = 1024 * 256
                downloaded = 0
                for part in resp.iter_content(chunk_size=chunk):
                    if part:
                        buf.write(part)
                        downloaded += len(part)
                        if total and downloaded % (chunk * 8) == 0:
                            pct = downloaded * 100 // total
                            print(
                                f"[zip] Manual download {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct}%)"
                            )
                data = buf.getvalue()
            print("[zip] Extracting Manual zip...")
            with zipfile.ZipFile(BytesIO(data)) as zf:
                zf.extractall(docs_dir)
            results = []
            for p in docs_dir.rglob("*.html"):
                rel = p.relative_to(docs_dir)
                parts = [s for s in rel.parts if s]
                section = "__root__"
                if len(parts) >= 2:
                    section = "_".join(parts[:2])
                elif parts:
                    section = parts[0]
                results.append((str(p), None, section))
            print(f"✓ Downloaded and extracted Manual zip, pages: {len(results)}")
            return results
        except Exception as e:
            print(f"[zip] Provided Manual zip failed, falling back: {e}")

    # Try to locate a ZIP link on the Manual index page, download, and extract.
    try:
        import re
        import zipfile
        from io import BytesIO
        from urllib.parse import urljoin

        import requests

        index_url = "https://docs.blender.org/manual/en/current/"
        headers = {"User-Agent": "BlenderAssistant-RAG/1.0 (+repo)"}
        r = requests.get(index_url, timeout=30, headers=headers)
        r.raise_for_status()
        html = r.text

        # Heuristic: find any .zip link on the Manual page
        m = re.search(r'href="([^"]+\\.zip)"', html, flags=re.IGNORECASE)
        if m:
            zip_url = m.group(1)
            if not zip_url.startswith("http"):
                zip_url = urljoin(index_url, zip_url)
            print(f"[zip] Found Manual zip: {zip_url} (downloading...)")

            with requests.get(
                zip_url, stream=True, timeout=300, headers=headers
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", "0") or "0")
                buf = BytesIO()
                chunk = 1024 * 256
                downloaded = 0
                for part in resp.iter_content(chunk_size=chunk):
                    if part:
                        buf.write(part)
                        downloaded += len(part)
                        if total and downloaded % (chunk * 8) == 0:
                            pct = downloaded * 100 // total
                            print(
                                f"[zip] Manual download {downloaded // 1024 // 1024}MB / {total // 1024 // 1024}MB ({pct}%)"
                            )
                data = buf.getvalue()

            print("[zip] Extracting Manual zip...")
            with zipfile.ZipFile(BytesIO(data)) as zf:
                zf.extractall(docs_dir)

            # Collect extracted HTML files
            results = []
            for p in docs_dir.rglob("*.html"):
                rel = p.relative_to(docs_dir)
                parts = [s for s in rel.parts if s]
                section = "__root__"
                if len(parts) >= 2:
                    section = "_".join(parts[:2])
                elif parts:
                    section = parts[0]
                results.append((str(p), None, section))

            print(f"✓ Downloaded and extracted Manual zip, pages: {len(results)}")
            return results
        else:
            print("[zip] No Manual zip link found on index, falling back to crawl.")
    except Exception as e:
        print(f"[zip] Manual zip download failed, falling back to crawl: {e}")

    # Fallback to crawling
    base_url = "https://docs.blender.org/manual/en/current/"
    results = crawl_site(base_url, docs_dir, source_label="Manual")
    print(f"✓ Downloaded {len(results)} manual pages (crawl)")
    return results


def extract_text_from_html(html_file: str) -> List[Dict[str, str]]:
    """Extract text content from HTML documentation, filtering menus/sidebars/navigation."""

    from bs4 import BeautifulSoup

    with open(html_file, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Aggressively strip non-content elements (headers/footers/menus/sidebars/navigation/forms/etc.)
    for tag in ("script", "style", "nav", "footer", "header", "aside", "form"):
        for el in list(soup.find_all(tag)):
            el.decompose()

    # Remove common noise by class/id/role hints (Sphinx/Docs specific: toctree, sidebar, breadcrumbs, search, related, navbar)
    noise_keywords = {
        "sidebar",
        "sphinxsidebar",
        "menu",
        "navbar",
        "breadcrumb",
        "breadcrumbs",
        "toctree",
        "toc",
        "navigation",
        "related",
        "search",
        "prev",
        "next",
        "skip",
        "version",
        "language",
        "copyright",
    }
    for el in list(soup.find_all(True)):
        # Be robust: some parsers can yield non-Tag or transient nodes
        if el is None or not hasattr(el, "get"):
            continue
        attrs = []
        try:
            for key in ("class", "id", "role", "aria-label"):
                val = el.get(key)
                if isinstance(val, (list, tuple)):
                    attrs.extend([str(x).lower() for x in val])
                elif val:
                    attrs.append(str(val).lower())
            tag_name = (getattr(el, "name", "") or "").lower()
            hint = " ".join(attrs)
            if tag_name in {"nav", "header", "footer", "aside"} or any(
                kw in hint for kw in noise_keywords
            ):
                el.decompose()
        except Exception:
            # If anything goes wrong inspecting this node, skip it
            continue

    # Prefer main content containers when present
    main_candidates = [
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
        soup.find("article"),
        soup.find("div", id="content"),
        soup.find("div", id="document"),
        soup.find(
            "div",
            class_=lambda c: c
            and any(
                x in " ".join(c) if isinstance(c, (list, tuple)) else str(c)
                for x in ("document", "content", "main", "article", "body")
            ),
        ),
    ]
    root = next((c for c in main_candidates if c), None)
    if root is None:
        root = soup

    # Extract text from the chosen root
    try:
        text = root.get_text(separator="\n")
    except Exception:
        text = ""

    # Clean up whitespace and drop common navigation/license noise lines
    lines = [ln.strip() for ln in text.splitlines()]
    filtered = []
    skip_terms = {
        "previous",
        "next",
        "navigation",
        "table of contents",
        "license",
        "creativecommons",
        "©",
        "copyright",
    }
    for ln in lines:
        if not ln or len(ln) <= 2:
            continue
        low = ln.lower()
        if any(term in low for term in skip_terms):
            continue
        filtered.append(ln)

    text = " ".join(filtered)
    return text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind(".")
            last_newline = chunk.rfind("\n")
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                chunk = chunk[: break_point + 1]
                end = start + break_point + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def generate_embeddings_ollama(
    texts: List[str], model: str = "nomic-embed-text"
) -> List[List[float]]:
    """Generate embeddings using Ollama."""
    print(f"\n=== Generating embeddings using Ollama ({model}) ===")

    embeddings = []
    total = len(texts)

    for i, text in enumerate(texts, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{total} ({i * 100 // total}%)")

        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            embeddings.append(embedding)

        except Exception as e:
            print(f"  ⚠ Failed to generate embedding for chunk {i}: {e}")
            # Use zero vector as fallback
            embeddings.append([0.0] * 768)  # nomic-embed-text dimension

    print(f"✓ Generated {len(embeddings)} embeddings")
    return embeddings


def create_vector_database(
    documents: List[Dict], output_file: str = "blender_assistant_mcp/blender_docs.json"
):
    """Create JSON vector database."""
    print(f"\n=== Creating JSON vector database ===")

    # Prepare data
    db_data = {
        "documents": [
            {"content": doc["content"], "metadata": doc["metadata"]}
            for doc in documents
        ],
        "embeddings": [doc["embedding"] for doc in documents],
    }

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Writing {len(documents)} documents to JSON...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(db_data, f)

    # Get file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"✓ Created vector database with {len(documents)} documents")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    print(f"✓ Saved to: {output_path}")

    return output_path


def main():
    """Main build process."""
    print("=" * 60)

    print("Blender RAG Database Builder")

    print("=" * 60)

    # CLI options

    import argparse

    parser = argparse.ArgumentParser(description="Build Blender RAG database")

    parser.add_argument(
        "--api-zip",
        dest="api_zip",
        default=None,
        help="Explicit URL to Blender API docs zip (e.g., https://docs.blender.org/api/current/blender_python_reference_4_5.zip)",
    )

    parser.add_argument(
        "--manual-zip",
        dest="manual_zip",
        default=None,
        help="Explicit URL to Blender Manual zip (e.g., https://docs.blender.org/manual/en/latest/blender_manual_html.zip)",
    )

    args = parser.parse_args()

    # Step 1: Download documentation

    api_files = download_blender_api_docs(api_zip_url=args.api_zip)

    manual_files = download_blender_manual(manual_zip_url=args.manual_zip)

    all_files = api_files + manual_files

    if not all_files:
        print("\n❌ No documentation downloaded. Exiting.")
        sys.exit(1)

    # Step 2: Extract and chunk text
    print("\n=== Processing documentation ===")
    documents = []

    for entry in all_files:
        # entry may be a file path (legacy) or a tuple (file_path, url, section)
        if isinstance(entry, tuple):
            file_path, url, section = entry
        else:
            file_path, url, section = entry, None, None

        print(f"  Processing {Path(file_path).name}...")

        try:
            # Extract text
            text = extract_text_from_html(file_path)

            # Chunk text
            chunks = chunk_text(text, chunk_size=1000, overlap=200)

            # Create document entries
            source = (
                "API"
                if "temp_docs/api" in str(file_path)
                else ("Manual" if "temp_docs/manual" in str(file_path) else "Unknown")
            )
            filename = Path(file_path).name

            for i, chunk in enumerate(chunks):
                documents.append(
                    {
                        "content": chunk,
                        "metadata": {
                            "source": source,
                            "file": filename,
                            "url": url,
                            "section": section,
                            "chunk": i,
                        },
                        "embedding": None,  # Will be filled later
                    }
                )

        except Exception as e:
            print(f"  ⚠ Failed to process {file_path}: {e}")

    print(f"✓ Created {len(documents)} document chunks")

    # Step 3: Generate embeddings
    texts = [doc["content"] for doc in documents]
    embeddings = generate_embeddings_ollama(texts)

    # Add embeddings to documents
    for doc, embedding in zip(documents, embeddings):
        doc["embedding"] = embedding

    # Step 4: Create sharded vector database and report
    index_path = create_sharded_database(documents)

    print("\n" + "=" * 60)
    print("✓ RAG Database Build Complete!")
    print("=" * 60)
    print(f"Documents: {len(documents)}")
    print(f"Index: {index_path}")
    print("\nNext step: Run 'python build_extension.py' to bundle with extension")


def create_sharded_database(
    documents: List[Dict], output_dir: str = "blender_assistant_mcp/rag_db"
):
    """Create sharded JSON vector database and a build report.

    Shards are grouped by source + section (if present in metadata).
    Writes:
      - rag_db/index.json
      - rag_db/<shard_name>.json
      - rag_db/report.json (blocked/failed URLs)
    """
    print(f"\n=== Creating sharded JSON vector database ===")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group documents into shards
    shards: Dict[str, Dict[str, List]] = {}
    for doc in documents:
        md = doc.get("metadata", {})
        source = md.get("source", "Unknown")
        section = md.get("section", md.get("file", "__root__"))
        # Normalize shard key
        shard_key = f"{source}_{section}".replace("/", "_").replace(" ", "_")
        bucket = shards.setdefault(shard_key, {"documents": [], "embeddings": []})
        bucket["documents"].append({"content": doc["content"], "metadata": md})
        bucket["embeddings"].append(doc["embedding"])

    index = {"version": 1, "total_documents": len(documents), "shards": []}

    for shard_name, shard_data in shards.items():
        shard_path = out_dir / f"{shard_name}.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(shard_data, f)
        index["shards"].append(
            {
                "name": shard_name,
                "documents": len(shard_data["documents"]),
                "path": shard_path.name,
            }
        )

    # Write index
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    # Write report
    report = {"blocked_urls": BLOCKED_URLS, "failed_urls": FAILED_URLS}
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f)

    print(f"\u2713 Created {len(shards)} shards, total documents: {len(documents)}")
    print(f"\u2713 Index written to {out_dir / 'index.json'}")
    return out_dir / "index.json"


if __name__ == "__main__":
    main()
