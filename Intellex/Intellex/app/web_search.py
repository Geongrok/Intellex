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
import time
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


    @staticmethod
    def _perf(label: str, elapsed: float) -> None:
        if os.getenv("INTELLEX_PERF_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}:
            print(f"[PERF] web.{label}: {elapsed:.3f}s", flush=True)

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
    # Relevance / query helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tokens(text: str) -> List[str]:
        import re
        stop = {
            "what", "is", "are", "the", "a", "an", "of", "for", "to", "and",
            "or", "in", "on", "at", "how", "why", "does", "do", "can", "could",
            "would", "should", "explain", "tell", "me", "please", "define",
            "definition", "meaning", "about", "from", "with", "give", "find",
            "value", "calculate", "calculate", "show", "used", "use",
        }
        out = []
        for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", (text or "").lower()):
            t = t.strip("_- ")
            if len(t) >= 3 and t not in stop:
                if t.endswith("ies") and len(t) > 5:
                    t = t[:-3] + "y"
                elif t.endswith("s") and len(t) > 4:
                    t = t[:-1]
                out.append(t)
        return out

    @classmethod
    def _rank_relevant(cls, query: str, results: List[Dict], n: int) -> List[Dict]:
        """Remove obviously unrelated search hits and rank the rest by query overlap."""
        import re
        q = " ".join(cls._tokens(query))
        q_terms = set(cls._tokens(query))
        if not q_terms:
            return results[:n]

        scored = []
        for idx, item in enumerate(results):
            hay = " ".join([
                str(item.get("title", "")),
                str(item.get("snippet", "")),
                str(item.get("url", "")),
            ])
            terms = set(cls._tokens(hay))
            overlap = q_terms & terms
            score = len(overlap) / max(1, len(q_terms))

            # Exact concept phrases are much stronger than isolated generic words.
            if len(q_terms) >= 2:
                phrase_terms = [t for t in cls._tokens(query)]
                title = str(item.get("title", "")).lower()
                snippet = str(item.get("snippet", "")).lower()
                raw = title + " " + snippet
                if " ".join(phrase_terms) in raw:
                    score += 0.35

            # Prefer authoritative technical sources when relevance is similar.
            url = str(item.get("url", "")).lower()
            if any(d in url for d in ("nasa.gov", "faa.gov", "mit.edu", "nist.gov", "esa.int")):
                score += 0.08

            # For a real question, zero concept overlap is almost certainly noise.
            if not overlap:
                continue
            scored.append((score, -idx, item))

        scored.sort(reverse=True, key=lambda x: (x[0], x[1]))
        return [item for _, _, item in scored[:n]]

    @staticmethod
    def _needs_aerospace_context(query: str) -> bool:
        q = (query or "").lower()
        terms = (
            "mach", "airfoil", "aerofoil", "aerodynamic", "aircraft", "wing",
            "lift", "drag", "reynolds", "naca", "compressible", "shock wave",
            "pitot", "airspeed", "propulsion", "thrust", "rocket", "flight",
            "boundary layer", "fuselage", "tailplane", "stability", "fem",
            "finite element", "structural", "turbine", "compressor", "nozzle",
        )
        return any(t in q for t in terms)

    @classmethod
    def _web_query(cls, query: str) -> str:
        q = " ".join((query or "").split()).strip()
        if cls._needs_aerospace_context(q):
            # Search engines are less likely to drift into unrelated meanings of
            # short terms such as "Mach", "lift", or "CL" when the domain is explicit.
            return f"{q} aerospace aerodynamics engineering"
        return q

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def search(self, query: str, max_results: int = None) -> List[Dict]:
        """Return a list of {title, url, snippet} dicts."""
        n = max_results or self.max_results
        search_query = self._web_query(query)
        results = []
        errors = []
        try:
            t0 = time.perf_counter()
            if self._backend == "openrouter" and self.openrouter_key:
                results = self._search_openrouter(search_query, n)
            elif self._backend == "tavily" and self.tavily_key:
                results = self._search_tavily(search_query, n)
            elif self._backend == "exa" and self.exa_key:
                results = self._search_exa(search_query, n)
            elif self._backend == "serpapi" and self.serpapi_key:
                results = self._search_serpapi(search_query, n)
            else:
                results = self._search_duckduckgo(search_query, n)
            self._perf(f"primary[{self._backend}]", time.perf_counter() - t0)
        except Exception as exc:
            self._perf(f"primary[{self._backend}] FAILED", time.perf_counter() - t0)
            errors.append(str(exc))

        relevant = self._rank_relevant(query, results, n)

        # Free DDGS can be rate-limited on hosted IPs. Try independent engines
        # one at a time so one failed engine does not discard another engine's hits.
        if len(relevant) < min(3, n):
            for engine in ("bing", "brave", "startpage", "yahoo", "duckduckgo"):
                try:
                    t0 = time.perf_counter()
                    retry = self._search_ddgs_engine(search_query, n, engine)
                    self._perf(f"retry[{engine}]", time.perf_counter() - t0)
                    candidate = self._rank_relevant(query, retry, n)
                    if len(candidate) > len(relevant):
                        relevant = candidate
                    if len(relevant) >= min(3, n):
                        break
                except Exception as exc:
                    errors.append(f"{engine}: {exc}")

        # Try exact wording if domain-expanded search was too restrictive.
        if len(relevant) < min(3, n) and search_query != query:
            for engine in ("bing", "brave", "startpage", "duckduckgo"):
                try:
                    t0 = time.perf_counter()
                    retry = self._search_ddgs_engine(query, n, engine)
                    self._perf(f"exact_retry[{engine}]", time.perf_counter() - t0)
                    candidate = self._rank_relevant(query, retry, n)
                    if len(candidate) > len(relevant):
                        relevant = candidate
                    if len(relevant) >= min(3, n):
                        break
                except Exception as exc:
                    errors.append(f"{engine}: {exc}")

        # Never fill the source list with unrelated results.
        if not relevant:
            link = "https://duckduckgo.com/?q=" + urllib.parse.quote(search_query)
            detail = errors[0][:180] if errors else "No relevant web results were returned."
            return [{
                "title": "Open web search",
                "url": link,
                "snippet": f"Intellex could not retrieve relevant web results automatically. {detail}",
                "_search_error": True,
            }]
        return relevant[:n]

    # ------------------------------------------------------------------ #
    # OpenRouter (built-in web search tool)
    # ------------------------------------------------------------------ #
    def _search_openrouter(self, query: str, n: int) -> List[Dict]:
        """Use OpenRouter's current server-side web search when explicitly enabled."""
        import json
        import urllib.request
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"Find {n} relevant web sources for: {query}. Use them as evidence."}],
            "tools": [{"type": "openrouter:web_search"}],
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openrouter_key}",
                "HTTP-Referer": "https://intellex.app",
                "X-Title": "Intellex",
            },
        )
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = []
        for choice in data.get("choices", []):
            msg = choice.get("message", {}) or {}
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {}) or {}
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except Exception:
                    args = {}
                for item in args.get("results", []) if isinstance(args, dict) else []:
                    if isinstance(item, dict) and item.get("url"):
                        results.append({
                            "title": str(item.get("title", "")).strip(),
                            "url": str(item.get("url", "")).strip(),
                            "snippet": str(item.get("snippet", item.get("content", ""))).strip(),
                        })
        return results[:n] or self._search_duckduckgo(query, n)

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
    # DDGS metasearch engines
    # ------------------------------------------------------------------ #
    def _search_ddgs_engine(self, query: str, n: int, engine: str) -> List[Dict]:
        from ddgs import DDGS
        results: List[Dict] = []
        with DDGS(timeout=12) as ddgs:
            for r in ddgs.text(query, max_results=n, backend=engine):
                results.append({
                    "title": r.get("title", "").strip(),
                    "url": r.get("href", r.get("url", "")).strip(),
                    "snippet": (r.get("body", "") or "").strip(),
                })
        return results[:n]

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

