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

        results = []
        meta_fetches = 0

        # Parse DuckDuckGo HTML results
        # DDG uses <div class="result"> for each result
        result_blocks = re.findall(
            r'<div[^>]+class="[^"]*\bresult\b[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html,
            re.DOTALL | re.IGNORECASE,
        )

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
    Does NOT have: modern stock photos, satellite imagery, AI-generated content.

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

        # Download image
        with httpx.Client() as client:
            response = client.get(image_url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()

            # Determine file extension from URL or content-type
            content_type = response.headers.get("content-type", "")
            if "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            elif "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"
            else:
                # Try to get from URL
                ext = os.path.splitext(image_url.split("?")[0])[1]
                if not ext or ext not in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                    ".bmp",
                    ".tga",
                ]:
                    ext = ".jpg"  # Default

            # Save to temp file
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"web_image_{hash(image_url)}{ext}")

            with open(temp_file, "wb") as f:
                f.write(response.content)

        # Load image into Blender
        image_name = f"WebImage_{hash(image_url)}"

        # Check if image already exists
        if image_name in bpy.data.images:
            bpy.data.images.remove(bpy.data.images[image_name])

        img = bpy.data.images.load(temp_file)

        img.name = image_name

        # Pack image if requested
        if pack_image:
            img.pack()
            # Can now delete temp file
            try:
                os.remove(temp_file)
            except:
                pass

        result = {
            "success": True,
            "image_name": image_name,
            "packed": pack_image,
            "size": f"{img.size[0]}x{img.size[1]}",
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

    except Exception as e:
        return {"error": f"Failed to download image: {str(e)}"}


def register_tools():
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
