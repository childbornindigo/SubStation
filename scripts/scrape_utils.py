"""
scrape_utils.py — Unified scraping interface.

Engines:
  1. Firecrawl (API-based, used when FIRECRAWL_API_KEY is set)
  2. ScrapLing (CLI-based, always available at /opt/homebrew/bin/scrapling)
"""

import os
import sys
import subprocess
import tempfile
import time
import re


SCRAPLING_BIN = "/opt/homebrew/bin/scrapling"


def _log(msg: str):
    print(msg, file=sys.stderr)


def _try_firecrawl(url: str, output_format: str = "md") -> dict | None:
    """Attempt scrape via Firecrawl. Returns result dict or None on failure."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        return None
    try:
        from firecrawl import FirecrawlApp
        app = FirecrawlApp(api_key=api_key)
        result = app.scrape_url(url, params={"formats": ["markdown"]})
        content = result.get("markdown", "") if isinstance(result, dict) else str(result)
        if not content:
            _log(f"[firecrawl] empty response for {url}")
            return None
        _log(f"[firecrawl] OK {url}")
        return {"content": content, "engine": "firecrawl", "success": True, "error": None}
    except Exception as e:
        _log(f"[firecrawl] failed {url}: {e}")
        return None


def _try_scrapling(url: str, timeout: int = 30000, stealth: bool = False) -> dict:
    """Scrape via ScrapLing CLI. Always returns a result dict."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        mode = "stealthy-fetch" if stealth else "fetch"
        cmd = [SCRAPLING_BIN, "extract", mode, url, tmp_path, "--timeout", str(timeout)]
        if stealth:
            cmd += ["--real-chrome", "--solve-cloudflare", "--block-webrtc", "--hide-canvas"]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout / 1000 + 30)

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            _log(f"[scrapling] failed {url}: exit {proc.returncode} — {stderr}")
            return {"content": "", "engine": "scrapling", "success": False, "error": stderr or f"exit code {proc.returncode}"}

        with open(tmp_path, "r") as f:
            content = f.read()

        if not content.strip():
            _log(f"[scrapling] empty output for {url}")
            return {"content": "", "engine": "scrapling", "success": False, "error": "empty output"}

        _log(f"[scrapling] OK {url}")
        return {"content": content, "engine": "scrapling", "success": True, "error": None}

    except subprocess.TimeoutExpired:
        _log(f"[scrapling] timed out {url}")
        return {"content": "", "engine": "scrapling", "success": False, "error": "timeout"}
    except Exception as e:
        _log(f"[scrapling] error {url}: {e}")
        return {"content": "", "engine": "scrapling", "success": False, "error": str(e)}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _is_rate_limited(result: dict) -> bool:
    """Check if a failure looks like rate limiting."""
    err = (result.get("error") or "").lower()
    return any(hint in err for hint in ["429", "too many", "rate limit"])


def smart_scrape(url: str, output_format: str = "md", timeout: int = 30000, stealth: bool = False) -> dict:
    """
    Scrape a URL. Returns {"content": str, "engine": str, "success": bool, "error": str|None}.

    If FIRECRAWL_API_KEY is set, tries Firecrawl first.
    On failure (or if no key), falls back to ScrapLing.

    stealth=True uses ScrapLing's stealthy-fetch with
    --real-chrome --solve-cloudflare --block-webrtc --hide-canvas
    """
    # Try Firecrawl first
    result = _try_firecrawl(url, output_format)
    if result:
        return result

    # Fall back to ScrapLing
    return _try_scrapling(url, timeout, stealth)


def batch_scrape(urls: list[str], stealth: bool = False, delay: float = 2.0) -> list[dict]:
    """Scrape multiple URLs with delay between requests. Returns list of smart_scrape results."""
    results = []
    current_delay = delay

    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(current_delay)

        result = smart_scrape(url, stealth=stealth)
        results.append(result)

        if not result["success"] and _is_rate_limited(result):
            current_delay *= 2
            _log(f"[batch] rate limited — delay increased to {current_delay}s")

    return results
