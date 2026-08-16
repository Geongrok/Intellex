"""Web search module with multiple backends.

Search strategy (first available wins):
  1. OpenRouter Web Search (built-in tool)   - uses existing OPENROUTER_API_KEY
  2. Tavily    (TAVILY_API_KEY)  - high-quality search API
  3. Exa       (EXA_API_KEY)     - neural web search API
  4. SerpAPI   (SERPAPI_API_KEY) - Google results API
  5. DuckDuckGo (free, no key)   - always-on fallback

Any configured keyed backend is used automatically when its env var is set.
If none are set, DuckDuckGo handles everything for free.
"""

import os
import urllib.parse
from typing import Dict, List

try:
    from dotenv import load_dotenv

    _ENV_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    load_dotenv(_ENV_PATH)
except Exception:
    pass


class WebSearch:
    """Performs web searches across configurable backends."""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.exa_key = os.getenv("EXA_API_KEY", "").strip()
        self.serpapi_key = os.getenv("SERPAPI_API_KEY", "").strip()
        self._ddgs_available = self._check_ddgs()
        self._backend = self._detect_backend()

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_ddgs() -> bool:
        try:
            from ddgs import DDGS  # noqa: F401

            return True
        except ImportError:
            return False

    def _detect_backend(self) -> str:
        forced = os.getenv("WEB_SEARCH_BACKEND", "").strip().lower()
        if forced:
            return forced
        if self.openrouter_key:
            return "openrouter"
        if self.tavily_key:
            return "tavily"
        if self.exa_key:
            return "exa"
        if self.serpapi_key:
            return "serpapi"
        return "duckduckgo"

    @property
    def available(self) -> bool:
        return True  # duckduckgo fallback always available

    def backend_name(self) -> str:
        return self._backend

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = None) -> List[Dict]:
        """Return a list of {title, url, snippet} dicts."""
        n = max_results or self.max_results
        try:
            if self._backend == "openrouter" and self.openrouter_key:
                results = self._search_openrouter(query, n)
            elif self._backend == "tavily" and self.tavily_key:
                results = self._search_tavily(query, n)
            elif self._backend == "exa" and self.exa_key:
                results = self._search_exa(query, n)
            elif self._backend == "serpapi" and self.serpapi_key:
                results = self._search_serpapi(query, n)
            else:
                results = self._search_duckduckgo(query, n)
        except Exception as exc:
            # Graceful fallback to DuckDuckGo, then to a search link.
            try:
                results = self._search_duckduckgo(query, n)
            except Exception:
                link = "https://duckduckgo.com/?q=" + urllib.parse.quote(query)
                results = [
                    {
                        "title": "Open search results in DuckDuckGo",
                        "url": link,
                        "snippet": f"Automatic search failed ({exc}). Click to open results.",
                    }
                ]
        return results[:n]

    # ------------------------------------------------------------------ #
    # OpenRouter (built-in web search tool)
    # ------------------------------------------------------------------ #
    def _search_openrouter(self, query: str, n: int) -> List[Dict]:
        import json
        import urllib.request

        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Search the web for the following query and return "
                        f"a JSON list (no markdown) of the top {n} results. "
                        'Each item must have exactly the keys "title", "url", '
                        '"snippet". Query: ' + query,
                    ),
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for current information",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openrouter_key}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Intellex",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]

        # If the model used the web_search tool, extract tool calls.
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            results = []
            for tc in tool_calls:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                q = args.get("query", query)
                results.extend(self._search_duckduckgo(q, n))
            if results:
                return results

        # Otherwise parse the model's JSON answer.
        content = (msg.get("content") or "").strip()
        parsed = self._parse_json_list(content)
        if parsed:
            return parsed
        # If we still got nothing usable, use DuckDuckGo.
        return self._search_duckduckgo(query, n)

    @staticmethod
    def _parse_json_list(text: str) -> List[Dict]:
        import json
        import re

        if not text:
            return []
        # Strip markdown fences if present.
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\[.*\]", text, re.S)
            if not match:
                return []
            try:
                data = json.loads(match.group(0))
            except Exception:
                return []
        results = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("url"):
                    results.append(
                        {
                            "title": str(item.get("title", "")).strip(),
                            "url": str(item.get("url", "")).strip(),
                            "snippet": str(item.get("snippet", "")).strip(),
                        }
                    )
        return results

    # ------------------------------------------------------------------ #
    # Tavily
    # ------------------------------------------------------------------ #
    def _search_tavily(self, query: str, n: int) -> List[Dict]:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "api_key": self.tavily_key,
                "query": query,
                "max_results": n,
                "search_depth": "basic",
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        for r in data.get("results", [])[:n]:
            results.append(
                {
                    "title": r.get("title", "").strip(),
                    "url": r.get("url", "").strip(),
                    "snippet": r.get("content", "").strip(),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # Exa
    # ------------------------------------------------------------------ #
    def _search_exa(self, query: str, n: int) -> List[Dict]:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "query": query,
                "numResults": n,
                "contents": {"text": {"maxCharacters": 400}},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.exa.ai/search",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.exa_key,
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        for r in data.get("results", [])[:n]:
            snippet = r.get("text", "") or ""
            results.append(
                {
                    "title": r.get("title", "").strip(),
                    "url": r.get("url", "").strip(),
                    "snippet": snippet.strip()[:400],
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # SerpAPI (Google)
    # ------------------------------------------------------------------ #
    def _search_serpapi(self, query: str, n: int) -> List[Dict]:
        import json
        import urllib.parse
        import urllib.request

        params = urllib.parse.urlencode(
            {
                "engine": "google",
                "q": query,
                "api_key": self.serpapi_key,
                "num": n,
            }
        )
        req = urllib.request.Request(f"https://serpapi.com/search.json?{params}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        for r in data.get("organic_results", [])[:n]:
            results.append(
                {
                    "title": r.get("title", "").strip(),
                    "url": r.get("link", "").strip(),
                    "snippet": r.get("snippet", "").strip(),
                }
            )
        return results

    # ------------------------------------------------------------------ #
    # DuckDuckGo (free fallback)
    # ------------------------------------------------------------------ #
    def _search_duckduckgo(self, query: str, n: int) -> List[Dict]:
        from ddgs import DDGS

        results: List[Dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=n):
                results.append(
                    {
                        "title": r.get("title", "").strip(),
                        "url": r.get("href", r.get("url", "")).strip(),
                        "snippet": (r.get("body", "") or "").strip(),
                    }
                )
        return results[:n]

