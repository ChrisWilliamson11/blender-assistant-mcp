"""Web-based MCP tools.

This module contains tools for web searches and other internet-based functionality.
"""

import bpy
import httpx

from . import mcp_tools


def web_search(query: str, num_results: int = 5) -> dict:
    """Search the web for information using DuckDuckGo HTML (no API key required).

    Args:
        query: The search query (e.g., "today's news", "blender tutorials")
        num_results: Number of results to return (default: 5, max: 10)

    Returns:
        Search results as dict with titles, URLs, and descriptions
    """
    try:
        import re
        from html import unescape
        from urllib.parse import quote_plus

        num_results = min(num_results, 10)

        # Use DuckDuckGo HTML (simpler structure, more reliable)
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        with httpx.Client(follow_redirects=True) as client:
            response = client.get(search_url, headers=headers, timeout=15.0)
            response.raise_for_status()
            html = response.text
            print(f"[Web] Fetched {len(html)} chars from DDG")

        results = []
        meta_fetches = 0

        # Parse DuckDuckGo HTML results
        # DDG uses <div class="result"> for each result
        result_blocks = re.findall(
            r'<div[^>]+class="[^"]*\bresult\b[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        print(f"[Web] Found {len(result_blocks)} result blocks")

        for block in result_blocks[:num_results]:
            # Extract title and URL from <a class="result__a" href="...">title</a>
            title_match = re.search(
                r'<a[^>]+class="[^"]*\bresult__a\b[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                re.IGNORECASE | re.DOTALL,
            )
            # Extract snippet from <a class="result__snippet">...</a>
            snippet_match = re.search(
                r'<(?:a|div|span)[^>]+class="[^"]*\bresult__snippet\b[^"]*"[^>]*>(.*?)</(?:a|div|span)>',
                block,
                re.IGNORECASE | re.DOTALL,
            )

            if title_match:
                url = unescape(title_match.group(1))
                title_html = unescape(title_match.group(2))
                # Strip any nested tags in the title text
                title = re.sub(r"<[^>]+>", "", title_html).strip()
                snippet = ""

                if snippet_match:
                    snippet_html = unescape(snippet_match.group(1))
                    # Strip any nested tags in the snippet text
                    snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()

                # DuckDuckGo uses redirect URLs like //duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com
                # Extract the actual URL from the uddg parameter
                from urllib.parse import parse_qs, unquote, urlparse

                if "uddg=" in url:
                    # Extract uddg parameter
                    try:
                        if url.startswith("//"):
                            url = "https:" + url
                        parsed = urlparse(url)
                        params = parse_qs(parsed.query)
                        if "uddg" in params:
                            url = unquote(params["uddg"][0])
                    except Exception:
                        pass  # Keep original URL if parsing fails

                # Fallback: if snippet empty, try fetching page meta description (limited quick fetches, no API keys)
                if not snippet and url and meta_fetches < 3:
                    try:
                        with httpx.Client(follow_redirects=True) as client:
                            resp2 = client.get(url, headers=headers, timeout=4.0)
                            if resp2.status_code == 200:
                                page = resp2.text
                                # Prefer standard meta description, then og:description
                                md = re.search(
                                    r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
                                    page,
                                    re.IGNORECASE | re.DOTALL,
                                )
                                if not md:
                                    md = re.search(
                                        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
                                        page,
                                        re.IGNORECASE | re.DOTALL,
                                    )
                                if md:
                                    meta_txt = re.sub(
                                        r"<[^>]+>", "", unescape(md.group(1))
                                    ).strip()
                                    if meta_txt:
                                        snippet = meta_txt
                    except Exception:
                        pass
                    meta_fetches += 1
                results.append({"title": title.strip(), "url": url, "snippet": snippet})

        if not results:
            # Fallback: Try DuckDuckGo Instant Answer API
            api_url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }

            with httpx.Client() as client:
                response = client.get(api_url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()

            # Get instant answer if available
            if data.get("AbstractText"):
                results.append(
                    {
                        "title": data.get("Heading", "Instant Answer"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("AbstractText", ""),
                    }
                )

            # Get related topics
            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append(
                        {
                            "title": topic.get("Text", "").split(" - ")[0],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", ""),
                        }
                    )

        if not results:
            return {
                "results": [],
                "message": f"No results found for '{query}'. Try a different search term.",
                "query": query,
            }

        # Format results as text
        formatted = f"Search results for '{query}':\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            if result["url"]:
                formatted += f"   URL: {result['url']}\n"
            if result["snippet"]:
                formatted += f"   {result['snippet']}\n"
            formatted += "\n"

        return {
            "results": results,
            "formatted": formatted,
            "query": query,
            "count": len(results),
        }

    except httpx.TimeoutException:
        return {"error": "Search request timed out. Please try again."}
    except httpx.HTTPError as e:
        return {"error": f"HTTP error during search: {str(e)}"}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def search_wikimedia_image(query: str, apply_to_active: bool = True) -> dict:
    """Search Wikimedia Commons for free images and download as texture.

    Wikimedia Commons has: nature photos, textures, historical images, architecture.
    Does NOT have: modern stock photos, satellite imagery.

    Args:
        query: What to search for (e.g., 'wood grain', 'brick wall', 'mountain landscape', 'Paris aerial')
        apply_to_active: Whether to apply texture to active object (default: True)

    Returns:
        Dict with success status and image info
    """
    try:
        # Use Wikimedia Commons API - free, reliable, no rate limits
        # API docs: https://www.mediawiki.org/wiki/API:Search
        api_url = "https://commons.wikimedia.org/w/api.php"

        # Step 1: Search for images
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": f"{query} filetype:bitmap",
            "srnamespace": "6",  # File namespace
            "srlimit": "10",
        }

        with httpx.Client() as client:
            response = client.get(api_url, params=search_params, timeout=15.0)
            response.raise_for_status()
            search_data = response.json()

        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return {
                "error": f"No images found for '{query}' on Wikimedia Commons",
                "hint": "Wikimedia has nature, textures, architecture, historical photos. Try simpler terms like 'wood texture', 'brick wall', 'mountain'. For modern stock photos, use search_stock_photos (requires API key) or download_image_as_texture with a direct URL.",
            }

        print(f"[DEBUG] Found {len(search_results)} images on Wikimedia Commons")

        # Step 2: Get image URLs for each result
        for result in search_results[:5]:  # Try first 5 results
            title = result.get("title", "")
            if not title.startswith("File:"):
                continue

            # Get image info
            info_params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "imageinfo",
                "iiprop": "url|size",
                "iiurlwidth": "1920",  # Request high-res version
            }

            with httpx.Client() as client:
                response = client.get(api_url, params=info_params, timeout=15.0)
                response.raise_for_status()
                info_data = response.json()

            pages = info_data.get("query", {}).get("pages", {})
            for page_data in pages.values():
                imageinfo = page_data.get("imageinfo", [])
                if not imageinfo:
                    continue

                img_url = imageinfo[0].get("url")
                if not img_url:
                    continue

                print(f"[DEBUG] Attempting to download: {img_url[:100]}")

                # Step 3: Download the image
                try:
                    result = download_image_as_texture(
                        img_url, apply_to_active, pack_image=True
                    )
                    if "error" not in result:
                        return {
                            "success": True,
                            "query": query,
                            "image_url": img_url,
                            "source": "Wikimedia Commons",
                            "title": title,
                            "message": f"Downloaded '{title}' from Wikimedia Commons",
                            **result,
                        }
                    else:
                        print(f"[DEBUG] Download failed: {result.get('error')}")
                except Exception as e:
                    print(f"[DEBUG] Download exception: {str(e)}")
                    continue

        return {
            "error": f"Could not download images for '{query}'",
            "hint": "Try a more specific search term or use download_image_as_texture with a direct URL",
        }

    except Exception as e:
        return {"error": f"Image search failed: {str(e)}"}


def fetch_webpage(url: str, max_length: int = 10000) -> dict:
    """Fetch and extract text content from a webpage.

    Args:
        url: URL of the webpage to fetch
        max_length: Maximum length of text to return (default: 10000 chars)

    Returns:
        Dictionary with extracted text content
    """
    try:
        import re
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            """Simple HTML parser to extract text."""

            def __init__(self):
                super().__init__()
                self.text = []
                self.skip_tags = {"script", "style", "meta", "link", "noscript"}
                self.current_tag = None

            def handle_starttag(self, tag, attrs):
                self.current_tag = tag

            def handle_endtag(self, tag):
                self.current_tag = None

            def handle_data(self, data):
                if self.current_tag not in self.skip_tags:
                    text = data.strip()
                    if text:
                        self.text.append(text)

            def get_text(self):
                return " ".join(self.text)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        with httpx.Client(follow_redirects=True) as client:
            response = client.get(url, headers=headers, timeout=15.0)
            response.raise_for_status()
            html = response.text

        # Extract text from HTML
        parser = TextExtractor()
        parser.feed(html)
        text = parser.get_text()

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length] + "... (truncated)"

        return {"success": True, "url": url, "text": text, "length": len(text)}

    except httpx.TimeoutException:
        return {"error": f"Request timed out while fetching {url}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error {e.response.status_code} while fetching {url}"}
    except Exception as e:
        return {"error": f"Failed to fetch webpage: {str(e)}"}


def download_image_as_texture(
    image_url: str, apply_to_active: bool = True, pack_image: bool = True
) -> dict:
    """Download an image from a URL and optionally apply it as a texture.



    Args:

        image_url: Direct URL to the image

        apply_to_active: If True, apply as texture to active object (default: True)

        pack_image: If True, pack image into .blend file (default: True)



    Returns:

        Dictionary with download result and image name

    """

    try:
        import os
        import tempfile
        from urllib.parse import urlparse

        # Basic URL validation

        parsed = urlparse(image_url)

        if parsed.scheme not in ("http", "https"):
            return {
                "error": "Invalid image_url: must be http(s) URL",
                "received": image_url,
            }

        # Prepare robust HTTP headers (helps avoid 403/521/CDN blocks)
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        referer = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else None
        headers = {
            "User-Agent": ua,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        if referer:
            headers["Referer"] = referer

        # Download image
        with httpx.Client(follow_redirects=True) as client:
            response = client.get(image_url, timeout=30.0, headers=headers)
            response.raise_for_status()

            # Determine file extension from URL or content-type

            content_type = (response.headers.get("content-type") or "").lower()
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"

            elif "png" in content_type:
                ext = ".png"

            elif "webp" in content_type:
                ext = ".webp"

            else:
                # Try to get from URL

                ext = os.path.splitext(image_url.split("?")[0])[1].lower()

                if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tga"]:
                    ext = ".jpg"  # Default

            # Save to temp file

            temp_dir = tempfile.gettempdir()

            temp_file = os.path.join(temp_dir, f"web_image_{hash(image_url)}{ext}")

            with open(temp_file, "wb") as f:
                f.write(response.content)

        # Load image into Blender (fail fast if Blender cannot load)

        image_name = f"WebImage_{hash(image_url)}"

        # Remove any stale with same name
        if image_name in bpy.data.images:
            try:
                bpy.data.images.remove(bpy.data.images[image_name])

            except Exception:
                pass

        try:
            img = bpy.data.images.load(temp_file)

        except Exception as load_err:
            # Clean up temp file if load fails
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return {
                "error": f"Blender failed to load image file: {str(load_err)}",
                "source_url": image_url,
                "content_type": content_type,
                "temp_file": temp_file,
            }

        img.name = image_name

        # Validate the loaded image (width/height must be non-zero)
        try:
            w, h = int(img.size[0]), int(img.size[1])
        except Exception:
            w, h = 0, 0
        if w <= 0 or h <= 0:
            # Remove invalid image datablock to avoid confusion with "Render Result"
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass
            try:
                os.remove(temp_file)

            except Exception:
                pass

            return {
                "error": "Loaded image has invalid size (0x0).",
                "source_url": image_url,
                "content_type": content_type,
                "temp_file": temp_file,
            }

        # Pack image if requested
        if pack_image:
            try:
                img.pack()
                # Can now delete temp file
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            except Exception as pack_err:
                # Packing failed; still continue but report
                pass

        result = {
            "success": True,
            "image_name": image_name,
            "packed": bool(pack_image),
            "size": f"{w}x{h}",
            "source_url": image_url,
        }

        # Apply as texture if requested

        if apply_to_active:
            obj = bpy.context.active_object

            if obj and obj.type == "MESH":
                # Create material if needed

                if not obj.data.materials:
                    mat = bpy.data.materials.new(name=f"Material_{obj.name}")

                    mat.use_nodes = True

                    obj.data.materials.append(mat)

                else:
                    mat = obj.data.materials[0]

                    if not mat.use_nodes:
                        mat.use_nodes = True

                # Get nodes

                nodes = mat.node_tree.nodes

                links = mat.node_tree.links

                # Find or create Principled BSDF

                bsdf = None

                for node in nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        bsdf = node

                        break

                if not bsdf:
                    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

                # Create image texture node

                tex_node = nodes.new("ShaderNodeTexImage")

                tex_node.image = img

                tex_node.location = (bsdf.location[0] - 300, bsdf.location[1])

                # Connect to Base Color

                links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

                result["applied_to"] = obj.name

                result["material"] = mat.name

                result["message"] = (
                    f"✓ Downloaded and applied image to '{obj.name}' (packed: {pack_image})"
                )

            else:
                result["applied_to"] = None

                result["message"] = (
                    f"✓ Downloaded image '{image_name}' (no active mesh to apply to)"
                )

        else:
            result["message"] = f"✓ Downloaded image '{image_name}' (not applied)"

        return result

    except httpx.TimeoutException:
        return {"error": f"Request timed out while fetching {image_url}"}
    except httpx.HTTPStatusError as e:
        return {
            "error": f"HTTP error {e.response.status_code} while fetching {image_url}",
            "status_code": e.response.status_code,
        }
    except Exception as e:
        return {"error": f"Failed to download image: {str(e)}"}


def extract_image_urls(url: str, min_width: int = 400, max_images: int = 10) -> dict:
    """Fetch a page and extract likely content image URLs (jpg/png/webp).

    Heuristics:
    - Parse <img>, srcset, og:image, twitter:image, and <link rel="image_src">
    - Resolve relative URLs against the page URL
    - Filter out SVGs and likely non-content images (logo/icon/sprite/thumb/avatar)
    - Use declared width/srcset hints to filter images smaller than min_width when available
    - Rank by inferred width (desc) and penalize likely non-content terms
    """
    try:
        import re
        from html import unescape
        from urllib.parse import urljoin

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        with httpx.Client(follow_redirects=True) as client:
            resp = client.get(url, headers=headers, timeout=15.0)
            resp.raise_for_status()
            html = resp.text

        # Collect candidates
        candidates: dict[str, dict] = {}

        def add_candidate(
            raw_url: str, width_hint: int | None = None, source: str = ""
        ):
            if not raw_url:
                return
            u = unescape(raw_url.strip())
            # Ignore data URLs
            if u.lower().startswith("data:"):
                return
            # Resolve against page URL
            u = urljoin(url, u)
            # Filter by extension
            ul = u.lower().split("?")[0].split("#")[0]
            if any(ul.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                # Exclude likely non-content by pattern
                bad_terms = [
                    "sprite",
                    "logo",
                    "icon",
                    "avatar",
                    "thumb",
                    "thumbnail",
                    "small",
                    "tiny",
                    "pixel",
                    "badge",
                ]
                if any(bt in ul for bt in bad_terms):
                    penalty = 1
                else:
                    penalty = 0
                # Store best width hint
                if u in candidates:
                    prev = candidates[u]
                    prev_w = prev.get("width_hint") or 0
                    if (width_hint or 0) > prev_w:
                        prev["width_hint"] = width_hint or prev_w
                        prev["penalty"] = penalty
                        prev["source"] = source or prev.get("source", "")
                else:
                    candidates[u] = {
                        "url": u,
                        "width_hint": width_hint or 0,
                        "penalty": penalty,
                        "source": source,
                    }

        # Regex helpers
        img_tag_re = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
        src_re = re.compile(r'\bsrc\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
        width_re = re.compile(r'\bwidth\s*=\s*["\']?(\d{2,5})["\']?', re.IGNORECASE)
        srcset_re = re.compile(r'\bsrcset\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)

        for tag in img_tag_re.findall(html):
            # src
            m = src_re.search(tag)
            w_hint = None
            mw = width_re.search(tag)
            if mw:
                try:
                    w_hint = int(mw.group(1))
                except Exception:
                    w_hint = None
            if m:
                add_candidate(m.group(1), w_hint, "img")
            # srcset (take widest)
            ms = srcset_re.search(tag)
            if ms:
                items = [p.strip() for p in ms.group(1).split(",") if p.strip()]
                best_url = None
                best_w = 0
                for it in items:
                    parts = it.split()
                    u = parts[0]
                    w = 0
                    if len(parts) >= 2 and parts[1].lower().endswith("w"):
                        try:
                            w = int(re.sub(r"[^\d]", "", parts[1]))
                        except Exception:
                            w = 0
                    if w > best_w:
                        best_w = w
                        best_url = u
                if best_url:
                    add_candidate(best_url, max(best_w, w_hint or 0), "srcset")

        # OpenGraph/Twitter
        for prop in ["og:image", "twitter:image", "twitter:image:src"]:
            for m in re.finditer(
                rf'<meta[^>]+(?:property|name)=["\']{prop}["\'][^>]+content=["\']([^"\']+)["\']',
                html,
                re.IGNORECASE,
            ):
                add_candidate(m.group(1), None, prop)

        # <link rel="image_src" href="...">
        for m in re.finditer(
            r'<link[^>]+rel=["\']image_src["\'][^>]+href=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE,
        ):
            add_candidate(m.group(1), None, "link:image_src")

        # Filter by min_width when we have hints, and remove duplicates while preserving ranking
        items = list(candidates.values())

        def score(it: dict) -> tuple:
            w = it.get("width_hint") or 0
            pen = it.get("penalty") or 0
            # Prefer larger widths; penalize likely non-content
            return (-w, pen)

        # Apply min_width filter if width hints exist; otherwise rely on penalties only
        filtered = []
        for it in items:
            w = it.get("width_hint") or 0
            if w and w < int(min_width):
                continue
            filtered.append(it)
        # Sort and clip
        filtered.sort(key=score)
        top = filtered[: max(1, int(max_images))]
        urls = [it["url"] for it in top]

        return {
            "success": True,
            "url": url,
            "count": len(urls),
            "images": urls,
        }
    except httpx.TimeoutException:
        return {"error": f"Request timed out while fetching {url}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error {e.response.status_code} while fetching {url}"}
    except Exception as e:
        return {"error": f"Failed to extract images: {str(e)}"}


def register():
    """Register all web tools with the MCP registry."""

    mcp_tools.register_tool(
        "web_search",
        web_search,
        "Search the web for information using DuckDuckGo (useful for Blender documentation, tutorials, best practices)",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., 'blender parent objects selection order')",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
        category="Web",
    )

    mcp_tools.register_tool(
        "search_wikimedia_image",
        search_wikimedia_image,
        "Search Wikimedia Commons for free images and download as texture. Good for: nature, textures, architecture, historical photos. NOT for: modern stock photos, satellite imagery. Use simple search terms like 'wood grain', 'brick wall', 'mountain landscape'.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Simple search term (e.g., 'wood texture', 'brick wall', 'mountain', 'Paris aerial view'). Avoid brand names, 'unsplash', '4k', 'satellite'.",
                },
                "apply_to_active": {
                    "type": "boolean",
                    "description": "Whether to apply the texture to the active object (default: true)",
                    "default": True,
                },
            },
            "required": ["query"],
        },
        category="Web",
    )

    mcp_tools.register_tool(
        "fetch_webpage",
        fetch_webpage,
        "Fetch and extract text content from a webpage. Useful for reading articles, documentation, or getting detailed information from search results.",
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the webpage to fetch (from web_search results)",
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum length of text to return in characters (default: 10000)",
                    "default": 10000,
                    "minimum": 1000,
                    "maximum": 50000,
                },
            },
            "required": ["url"],
        },
        category="Web",
    )

    mcp_tools.register_tool(
        "extract_image_urls",
        extract_image_urls,
        "Extract likely content image URLs from a webpage (jpg/png/webp). Filters out tiny/logos/sprites and returns top N.",
        {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Webpage URL to analyze",
                },
                "min_width": {
                    "type": "integer",
                    "description": "Minimum width to consider (uses HTML hints when available)",
                    "default": 400,
                    "minimum": 1,
                    "maximum": 4096,
                },
                "max_images": {
                    "type": "integer",
                    "description": "Maximum number of image URLs to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["url"],
        },
        category="Web",
    )

    mcp_tools.register_tool(
        "download_image_as_texture",
        download_image_as_texture,
        "Download an image from a URL and optionally apply it as a texture. Images are packed into the .blend file by default.",
        {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "Direct URL to the image (from search_wikimedia_image results)",
                },
                "apply_to_active": {
                    "type": "boolean",
                    "description": "Apply as texture to active object (default: true)",
                    "default": True,
                },
                "pack_image": {
                    "type": "boolean",
                    "description": "Pack image into .blend file so it's self-contained (default: true)",
                    "default": True,
                },
            },
            "required": ["image_url"],
        },
        category="Web",
    )
