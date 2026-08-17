"""Core Intellex routing: database first, specialist calculation second, web last.

Case 1: No database answer and no suitable AeroCalc calculation -> web search.
Case 2: Relevant content found in the database -> answer from the database.
Case 3: Database miss + explicit numerical aerospace calculation -> AeroCalc.
"""

from typing import Dict, List, Optional
import os
import time


from .knowledge_base import KnowledgeBase, STOP_WORDS
from .web_search import WebSearch
from .llm import LLMEngine
from . import aerocalc_bridge

# A DB chunk whose keyword-coverage score is at or above this is
# considered "relevant". 0..1, i.e. at least 55% of the query's meaningful
# terms (after removing stop words) must appear in the chunk. This avoids
# false positives where an unrelated question shares only generic words.
RELEVANCE_THRESHOLD = 0.55


class ChatBot:
    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        web_results: int = 5,
        threshold: float = RELEVANCE_THRESHOLD,
    ):
        # Passing None lets KnowledgeBase fall back to its project-root-anchored
        # defaults, so documents are found regardless of the working directory.
        self.kb = KnowledgeBase(data_dir=data_dir, cache_dir=cache_dir)
        self.web = WebSearch(max_results=web_results)
        self.llm = LLMEngine()
        self.threshold = threshold
        self._index_ready = False
        self.aerocalc = aerocalc_bridge

    def ensure_index(self, force: bool = False) -> None:
        """Load or build the knowledge-base index once."""
        if self._index_ready and not force:
            return
        self.kb.build_index(force=force)
        self._index_ready = True


    @staticmethod
    def _perf(label: str, elapsed: float) -> None:
        """Emit optional request-stage timing without changing behavior."""
        if os.getenv("INTELLEX_PERF_LOG", "1").strip().lower() not in {"0", "false", "no", "off"}:
            print(f"[PERF] {label}: {elapsed:.3f}s", flush=True)

    @staticmethod
    def _is_explicit_calculation_request(question: str) -> bool:
        """Return True only when the user is actually asking for a calculation.

        AeroCalc is a specialist numerical engine, not a glossary.  A bare
        concept such as "what is Mach number?" must therefore go to DB/web,
        even though AeroCalc has a Mach-number calculator.
        """
        import re

        q = re.sub(r"\s+", " ", (question or "").lower()).strip()

        # Direct calculation verbs are the strongest signal.
        if re.search(
            r"\b(?:calculate|compute|determine|solve|derive|evaluate|work\s+out|find)\b",
            q,
        ):
            return True

        # A numerical input combined with an aerospace/calculation concept is
        # also an explicit calculation, e.g. "density at 11000 m" or
        # "normal shock at Mach 2.5".
        has_number = bool(re.search(r"(?<![a-z])[-+]?\d+(?:[.,]\d+)?", q))
        if not has_number:
            return False

        # A bare NACA designation is a concept/identity question, not a
        # numerical request.  "What is NACA 2412?" must therefore stay in
        # Intellex DB/Web rather than opening the AeroCalc airfoil calculator.
        if re.fullmatch(r"what\s+is\s+(?:the\s+)?naca\s+\d{4}\??", q):
            return False

        calc_concepts = (
            "mach", "naca", "airfoil", "shock", "isentropic", "fanno",
            "rayleigh", "nozzle", "reynolds", "density", "pressure",
            "temperature", "airspeed", "velocity", "speed of sound",
            "l/d", "lift to drag", "orbital", "orbit", "hohmann",
            "delta-v", "delta v", "rocket equation", "escape velocity",
            "pipe flow", "friction factor", "pitot",
        )
        return any(term in q for term in calc_concepts)

    # ------------------------------------------------------------------ #
    # Answer
    # ------------------------------------------------------------------ #
    def answer(self, question: str, rebuild_index: bool = False) -> Dict:
        """Answer using Intellex's cascade: DB -> AeroCalc (when appropriate) -> web.

        The knowledge base remains the primary source. AeroCalc is a specialist
        calculator and is only considered after a DB miss and only for an
        explicit numerical aerospace calculation. Everything else falls back
        to web search.
        """
        request_started = time.perf_counter()
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Please ask a question.",
                "case": None,
                "db_results": [],
                "web_results": [],
                "aerocalc": None,
                "mode": self.llm.mode,
            }

        t0 = time.perf_counter()
        self.ensure_index(force=rebuild_index)
        self._perf("index_ready", time.perf_counter() - t0)

        # -------------------------------------------------------------- #
        # 1) DATABASE FIRST -- this is Intellex's primary purpose.
        # -------------------------------------------------------------- #
        # Expand common aerospace abbreviations for retrieval only. The user's
        # original question is still passed to the answer generator, so the
        # response remains natural. For example, "Cmo" becomes searchable as
        # "Cmo / Cm0 / pitching moment coefficient at zero lift".
        t0 = time.perf_counter()
        retrieval_question = self.llm.normalize_query(question)
        self._perf("query_normalization", time.perf_counter() - t0)

        # Retrieve a wider candidate set, then VALIDATE whether the retrieved
        # passages actually answer this question. Semantic similarity alone is
        # not enough for a question bank: an unrelated aerodynamics question
        # can be close in embedding space to "what is aerodynamic lift".
        t0 = time.perf_counter()
        db_candidates = self.kb.search(retrieval_question, top_k=8)
        self._perf(f"db_search ({len(db_candidates)} candidates)", time.perf_counter() - t0)
        # Validate against the ORIGINAL question. Expansion is retrieval-only;
        # otherwise an alias such as Cmo -> "pitching moment coefficient at zero
        # lift" could make the relevance gate demand words that are not present
        # in a perfectly valid source that uses the abbreviation Cmo.
        t0 = time.perf_counter()
        relevant = self.llm.filter_relevant_db_chunks(
            question,
            db_candidates,
            min_score=self.threshold,
        )
        self._perf(f"db_relevance ({len(relevant)} accepted)", time.perf_counter() - t0)

        if relevant:
            # Keep source cards focused on the evidence sentence that passed
            # validation, not the entire question-bank chunk.
            t0 = time.perf_counter()
            relevant = self.llm.attach_evidence_excerpts(question, relevant)
            self._perf("db_evidence_excerpts", time.perf_counter() - t0)
            # A database hit is authoritative for the Intellex workflow. Do
            # not waste a web request or let AeroCalc override the answer.
            t0 = time.perf_counter()
            answer_text = self.llm.generate(question, relevant, [])
            self._perf(f"answer_generation mode={self.llm.mode}", time.perf_counter() - t0)
            # A validated retrieval is only a candidate evidence set. If the
            # answer generator cannot turn it into an actual answer, continue
            # through AeroCalc/web rather than exposing the raw DB chunk.
            if answer_text and answer_text.strip():
                self._perf("TOTAL", time.perf_counter() - request_started)
                return {
                    "answer": answer_text,
                    "case": 2,
                    "source": "database",
                    "db_results": relevant,
                    "web_results": [],
                    "aerocalc": None,
                    "mode": self.llm.mode,
                }

        # -------------------------------------------------------------- #
        # 2) DATABASE MISS -> AeroCalc only for genuine calculations.
        # -------------------------------------------------------------- #
        calc = None
        if self._is_explicit_calculation_request(question):
            t0 = time.perf_counter()
            calc = self.aerocalc.compute(question)
            self._perf("aerocalc", time.perf_counter() - t0)
        if calc is not None:
            if calc.get("error"):
                self._perf("TOTAL", time.perf_counter() - request_started)
                return {
                    "answer": (
                        f"AeroCalc matched this as **{calc['match_name']}**, "
                        "but the calculation could not be completed."
                    ),
                    "case": 3,
                    "source": "aerocalc",
                    "db_results": [],
                    "web_results": [],
                    "aerocalc": calc,
                    "mode": self.llm.mode,
                }

            if not calc.get("summary"):
                suggestions = calc.get("suggestions") or []
                answer = (
                    f"AeroCalc matched your question to **{calc['match_name']}**, "
                    "but I need more input to calculate it."
                )
                if suggestions:
                    answer += "\n\n" + "\n".join(f"- {x}" for x in suggestions)
                self._perf("TOTAL", time.perf_counter() - request_started)
                return {
                    "answer": answer,
                    "case": 3,
                    "source": "aerocalc",
                    "db_results": [],
                    "web_results": [],
                    "aerocalc": calc,
                    "mode": self.llm.mode,
                }

            self._perf("TOTAL", time.perf_counter() - request_started)
            return {
                "answer": (
                    f"**AeroCalc result — {calc['match_name']}**\n\n"
                    f"{calc['summary']}"
                ),
                "case": 3,
                "source": "aerocalc",
                "db_results": [],
                "web_results": [],
                "aerocalc": calc,
                "mode": self.llm.mode,
            }

        # -------------------------------------------------------------- #
        # 3) DB miss + no suitable calculation -> WEB SEARCH.
        # -------------------------------------------------------------- #
        # Use the expanded terminology for web retrieval too. This prevents
        # "Cmo" from being interpreted by a general search engine as unrelated
        # non-aerospace acronyms such as Commercial Market Outlook.
        t0 = time.perf_counter()
        web_results = self.web.search(retrieval_question)
        self._perf(f"web_search ({len(web_results)} results)", time.perf_counter() - t0)
        usable_web = [r for r in web_results if not r.get("_search_error")]

        if not usable_web:
            return {
                "answer": (
                    "I couldn't retrieve relevant web sources for this question "
                    "right now. The local knowledge base did not contain a "
                    "sufficient match, and the web-search provider did not return "
                    "usable results. Please try again in a moment."
                ),
                "case": 1,
                "source": "web_unavailable",
                "db_results": [],
                "web_results": web_results,
                "aerocalc": None,
                "mode": self.llm.mode,
            }

        answer_text = self.llm.generate(question, [], usable_web)
        self._perf("TOTAL", time.perf_counter() - request_started)
        return {
            "answer": answer_text,
            "case": 1,
            "source": "web",
            "db_results": [],
            "web_results": usable_web,
            "aerocalc": None,
            "mode": self.llm.mode,
        }

    # ------------------------------------------------------------------ #
    # Convenience / status
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict:
        return {
            "docs_loaded": self.kb.doc_count(),
            "files": self.kb.file_names(),
            "llm_mode": self.llm.mode,
            "web_search": self.web.available,
            "web_backend": self.web.backend_name(),
            "embedding": self.kb.vector.embedding.info(),
            "vector_docs": self.kb.vector.count(),
            "threshold": self.threshold,
            "aerocalc": self.aerocalc.info(),
        }

