"""
Retrieval Accuracy Test Suite for SP 21:2005 RAG System
Tests 20 queries against known expected results.
"""
import sys
sys.path.insert(0, '.')
import query_engine
from intent_graph import detect_intent

query_engine.initialize()

test_cases = [
    {"q": "What is the compressive strength of 53 grade OPC?", "expect_section": 1},
    {"q": "Tell me about IS 269", "expect_is": "IS 269"},
    {"q": "grading requirements for fine aggregates", "expect_section": 1},
    {"q": "chemical composition of Portland slag cement", "expect_is": "IS 455"},
    {"q": "what timber species are used for structural purposes?", "expect_section": 6},
    {"q": "glass thickness standards", "expect_section": 21},
    {"q": "thermal insulation materials for buildings", "expect_section": 23},
    {"q": "dimensions of concrete masonry blocks", "expect_section": 1},
    {"q": "water absorption of burnt clay bricks", "expect_section": 1},
    {"q": "tensile strength of TMT bars", "expect_section": 14},
    {"q": "IS 8112 43 grade cement", "expect_is": "IS 8112"},
    {"q": "compressive strength of fly ash bricks", "expect_section": 1},
    {"q": "what are the types of plywood?", "expect_section": 4},
    {"q": "gypsum plaster board specifications", "expect_section": 5},
    {"q": "bitumen grade for road construction", "expect_section": 7},
    {"q": "sanitary fittings for bathroom", "expect_section": 10},
    {"q": "welding electrode for structural steel", "expect_section": 18},
    {"q": "wire rope specifications", "expect_section": 20},
    {"q": "aluminium alloy for windows", "expect_section": 16},
    {"q": "door and window frame standards", "expect_section": 13},
]

print("=" * 70)
print("  RETRIEVAL ACCURACY TEST -- 20 queries")
print("=" * 70)

correct = 0
total = len(test_cases)

for i, tc in enumerate(test_cases, 1):
    q = tc["q"]
    intent = detect_intent(q)
    qvec = query_engine.embed_query(q)

    cids = None
    if intent.get("is_code_ref"):
        is_num = intent["is_code_ref"]
        if is_num in query_engine._chunk_ids_by_is_number:
            did = query_engine._chunk_ids_by_is_number[is_num]
            dm = query_engine._metadata.get(str(did))
            if dm:
                cids = query_engine._chunk_ids_by_section.get(dm["section_id"], set())
    elif intent.get("section_id"):
        cids = query_engine._chunk_ids_by_section.get(intent["section_id"], set())

    results = query_engine.search_chunks(qvec, cids, top_k=5, score_threshold=0.15)
    if not results and cids:
        results = query_engine.search_chunks(qvec, None, top_k=5, score_threshold=0.15)

    hit = False
    if "expect_is" in tc:
        for r in results:
            if tc["expect_is"] in r["is_code"]:
                hit = True
                break
    elif "expect_section" in tc:
        for r in results:
            if r["section_id"] == tc["expect_section"]:
                hit = True
                break

    if hit:
        correct += 1

    status = "PASS" if hit else "MISS"
    top_code = results[0]["is_code"] if results else "none"
    top_score = f"{results[0]['score']:.3f}" if results else "0"
    top_section = results[0]["section_id"] if results else "-"
    intent_sec = intent.get("section_name", "ALL")

    print(f"  [{status}] Q{i}: {q[:55]}")
    print(f"         Intent: {intent_sec} | Top: {top_code} (s{top_section}, score={top_score})")

print()
pct = correct / total * 100
print(f"  ACCURACY: {correct}/{total} = {pct:.0f}%")
print("=" * 70)
