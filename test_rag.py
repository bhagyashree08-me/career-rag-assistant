"""
test_rag.py

Standalone diagnostic script for the Career RAG Assistant backend.

Run this directly with your venv active, from the project root
(same folder as app.py):

    python test_rag.py

It does NOT touch Streamlit at all -- it calls src.rag directly,
so if something is broken here, the problem is in rag.py /
prompts.py / your Chroma DB / your API key, not in app.py or the
UI. If everything here works but the Streamlit app still misbehaves,
the problem is more likely in app.py.

What it checks, in order:

1. Environment sanity (GOOGLE_API_KEY present, chroma_db exists)
2. Embeddings/vector-store load time (should be slow ONCE, then
   fast on every call after -- that's the caching fix)
3. A normal, in-scope question that should hit the local
   knowledge base and answer quickly without web search
4. An off-topic question that should get a short "out of scope"
   refusal instead of a forced fake analysis
5. A question with no supporting evidence at all, to check the
   assistant says so instead of guessing
6. A question unlikely to be covered by the local PDFs, to check
   that web search grounding actually kicks in as a fallback
7. Prints total timing for each call so you can see whether the
   speed fixes are actually taking effect
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_environment():
    section("1. ENVIRONMENT CHECK")

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("FAIL: GOOGLE_API_KEY is not set in your .env file.")
        print("      Nothing past this point will work without it.")
        return False
    else:
        print(f"OK: GOOGLE_API_KEY is set (starts with {api_key[:6]}...)")

    from src.rag import CHROMA_DIR

    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        print(f"INFO: Chroma directory not found at {CHROMA_DIR}")
        print("      get_vector_store() will auto-build it from the")
        print("      PDFs in data/ on first call (via document_processor.py).")
        print("      That first call will be slow (loading + embedding")
        print("      every PDF page) -- this is expected ONCE, not a bug.")
        print("      Make sure data/ actually contains your PDFs before")
        print("      running this script, or it will raise FileNotFoundError.")
    else:
        print(f"OK: Chroma directory found at {CHROMA_DIR}")

    return True


def time_call(label, fn):
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    print(f"\n[{elapsed:.2f}s] {label}")
    return result, elapsed


def check_caching():
    section("2. CACHING CHECK (embeddings/vector store/LLM client)")

    from src.rag import get_embeddings, retrieve_documents

    _, first_load = time_call(
        "First call to get_embeddings() -- loads the model, expect this to be the slow one",
        get_embeddings,
    )

    _, second_load = time_call(
        "Second call to get_embeddings() -- should be near-instant (cached)",
        get_embeddings,
    )

    if second_load < first_load / 3 or second_load < 0.05:
        print("OK: caching is working -- second call was much faster.")
    else:
        print(
            "WARN: second call wasn't meaningfully faster than the "
            "first. Caching may not be taking effect (e.g. if the "
            "process restarted between calls, or lru_cache isn't "
            "being hit for some reason)."
        )

    print(
        "\nNOTE: if chroma_db/ didn't exist yet, the call below "
        "triggers a one-time auto-build from data/*.pdf (loading + "
        "embedding every page). That first call can legitimately "
        "take a while depending on how many/how large your PDFs "
        "are -- that's expected, not a bug. Run this script a "
        "second time afterward to see the real steady-state speed."
    )

    _, retrieval_time = time_call(
        "retrieve_documents() with a sample query",
        lambda: retrieve_documents("Python developer with machine learning experience"),
    )


def run_question(label, question, documents=None):
    from src.rag import retrieve_documents, generate_answer

    section(label)

    if documents is None:
        documents, retrieval_time = time_call(
            "retrieve_documents()",
            lambda: retrieve_documents(question),
        )
        print(f"  -> {len(documents)} relevant local chunk(s) found")

    answer, gen_time = time_call(
        "generate_answer()",
        lambda: generate_answer(question, documents),
    )

    print(f"\n--- ANSWER ({len(answer)} chars) ---")
    print(answer[:1500])
    if len(answer) > 1500:
        print(f"... [truncated, {len(answer) - 1500} more characters]")
    print("--- END ANSWER ---")

    return answer, gen_time


def main():
    if not check_environment():
        return

    try:
        check_caching()
    except Exception as e:
        print(f"FAIL: caching check crashed: {e}")

    # ------------------------------------------------------------
    # TEST A: normal, in-scope, well-supported question.
    # Expect: fast-ish answer, grounded mainly in local docs,
    # no refusal, no obvious hallucination.
    # ------------------------------------------------------------
    try:
        run_question(
            "3. NORMAL IN-SCOPE QUESTION",
            (
                "CANDIDATE RESUME\n================\n"
                "MCA student with academic project experience in "
                "Computer Vision, Deep Learning (CNNs), and Full "
                "Stack Development.\n\n"
                "TARGET JOB DESCRIPTION\n=====================\n"
                "Software Engineer, 0-2 years experience, Python "
                "and machine learning skills required.\n\n"
                "Perform a complete career match analysis. "
                "At the beginning provide:\nOVERALL_MATCH: XX%"
            ),
        )
    except Exception as e:
        print(f"FAIL: normal question crashed: {e}")

    # ------------------------------------------------------------
    # TEST B: off-topic question -- should trigger the SCOPE CHECK
    # in prompts.py and return a short refusal, NOT a forced
    # career analysis.
    # ------------------------------------------------------------
    try:
        answer, _ = run_question(
            "4. OFF-TOPIC QUESTION (should be refused, not hallucinated)",
            (
                "CANDIDATE RESUME\n================\n(resume text)\n\n"
                "TARGET JOB DESCRIPTION\n=====================\n(job text)\n\n"
                "USER QUESTION:\nWhat's a good recipe for chocolate chip cookies?"
            ),
        )
        lowered = answer.lower()
        if "out of scope" in lowered or "outside the scope" in lowered or "not related" in lowered:
            print("\nOK: looks like the scope check correctly refused this.")
        else:
            print(
                "\nCHECK MANUALLY: the answer didn't contain an obvious "
                "refusal phrase -- read the printed answer above and "
                "confirm it didn't just answer the cookie question."
            )
    except Exception as e:
        print(f"FAIL: off-topic question crashed: {e}")

    # ------------------------------------------------------------
    # TEST C: question with essentially no supporting evidence.
    # Expect: an honest "insufficient evidence" style answer, not
    # a confident-sounding guess.
    # ------------------------------------------------------------
    try:
        run_question(
            "5. INSUFFICIENT-EVIDENCE QUESTION",
            (
                "CANDIDATE RESUME\n================\n"
                "MCA student, no professional experience listed.\n\n"
                "TARGET JOB DESCRIPTION\n=====================\n"
                "Senior Quantum Cryptography Architect, 15+ years "
                "experience required.\n\n"
                "USER QUESTION:\nWhat was the candidate's exact salary "
                "at their previous employer, and what specific "
                "quantum algorithms have they personally implemented "
                "in production?"
            ),
        )
        print(
            "\nCHECK MANUALLY: the answer above should say this "
            "information isn't available/demonstrated, not invent a "
            "salary figure or a list of algorithms."
        )
    except Exception as e:
        print(f"FAIL: insufficient-evidence question crashed: {e}")

    # ------------------------------------------------------------
    # TEST D: force the "no local documents" path directly, to
    # confirm the web-search grounding fallback actually fires
    # instead of just returning a canned refusal.
    # ------------------------------------------------------------
    try:
        run_question(
            "6. FORCED WEB-SEARCH FALLBACK (empty local documents)",
            (
                "CANDIDATE RESUME\n================\n(resume text)\n\n"
                "TARGET JOB DESCRIPTION\n=====================\n"
                "Job at a real, well-known company.\n\n"
                "USER QUESTION:\nWhat does this company do, and roughly "
                "how large is it?"
            ),
            documents=[],  # force empty local context
        )
        print(
            "\nCHECK MANUALLY: with zero local documents passed in, "
            "the answer should still say something concrete about the "
            "company (via web search) rather than just refusing "
            "outright. If it refuses instead, grounding may not be "
            "active -- check that `bind_tools` succeeded (no warning "
            "printed above) and that your langchain-google-genai "
            "version supports it."
        )
    except Exception as e:
        print(f"FAIL: forced web-search test crashed: {e}")

    section("DONE")
    print(
        "Review each section above. Anything marked FAIL is a hard "
        "break -- fix that first. Anything marked CHECK MANUALLY "
        "needs you to read the printed answer and judge it yourself, "
        "since correctness here depends on what the model actually "
        "said, not just whether it crashed."
    )


if __name__ == "__main__":
    main()