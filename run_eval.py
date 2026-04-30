"""
BIS Hackathon Evaluation Runner
Processes public_test_set.json through our RAG pipeline and generates
output in the exact format required by eval_script.py.

Enhancement: Extracts cross-referenced IS codes from retrieved chunk content
to find part-numbered standards (e.g., IS 2185 (Part 2): 1983) that appear
in the text but aren't separate index entries.
"""
import sys
import json
import time
import re

sys.path.insert(0, 'backend')
import query_engine
from intent_graph import detect_intent

# Initialize engine
query_engine.initialize()

# Load dataset for content access
with open("dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)
    standards_by_id = {s["chunk_id"]: s for s in dataset["standards"]}

# Load test set
with open("public_test_set.json", "r", encoding="utf-8") as f:
    test_set = json.load(f)


def extract_is_codes_from_text(text):
    """Extract all IS code references from chunk content, including part numbers."""
    # Pattern: IS 1489 (Part 2): 1991  or  IS 2185 (Part 1) : 1979  or  IS 383: 1970
    pattern = r'IS\s*(\d{2,5})\s*(?:\(Part\s*\d+\)\s*)?:\s*\d{4}'
    # Full pattern with part numbers
    full_pattern = r'IS\s*\d{2,5}\s*(?:\(Part\s*\d+\)\s*)?:\s*\d{4}'
    matches = re.findall(full_pattern, text, re.IGNORECASE)
    return matches


results = []

print("=" * 60)
print("  BIS HACKATHON -- Running Public Test Set")
print("=" * 60)

for item in test_set:
    qid = item["id"]
    query = item["query"]
    expected = item["expected_standards"]

    t_start = time.time()

    # Run intent detection
    intent = detect_intent(query)

    # Build candidate IDs
    candidate_ids = None
    if intent["is_code_ref"]:
        is_num = intent["is_code_ref"]
        if is_num in query_engine._chunk_ids_by_is_number:
            did = query_engine._chunk_ids_by_is_number[is_num]
            dm = query_engine._metadata.get(str(did))
            if dm:
                candidate_ids = query_engine._chunk_ids_by_section.get(dm["section_id"], set())
    elif intent["section_id"]:
        candidate_ids = query_engine._chunk_ids_by_section.get(intent["section_id"], set())

    # Embed and search
    qvec = query_engine.embed_query(query)
    retrieved = query_engine.search_chunks(qvec, candidate_ids, top_k=5, score_threshold=0.10)

    # Fallback to full search
    if not retrieved and candidate_ids is not None:
        retrieved = query_engine.search_chunks(qvec, None, top_k=5, score_threshold=0.10)

    t_end = time.time()
    latency = t_end - t_start

    # Extract primary IS codes from retrieved chunks
    retrieved_standards = [r["is_code"] for r in retrieved]

    # Enhancement: scan chunk content for cross-referenced IS codes
    # This catches part-numbered standards like IS 2185 (Part 2): 1983
    seen = set(s.replace(" ", "").lower() for s in retrieved_standards)
    for r in retrieved:
        chunk_id = r.get("chunk_id")
        if chunk_id is not None and chunk_id in standards_by_id:
            full_std = standards_by_id[chunk_id]
            content = json.dumps(full_std)  # Search entire JSON entry
            xrefs = extract_is_codes_from_text(content)
            for xref in xrefs:
                xref_norm = xref.replace(" ", "").lower()
                if xref_norm not in seen:
                    seen.add(xref_norm)
                    retrieved_standards.append(xref)

    # Trim to top 5
    retrieved_standards = retrieved_standards[:5]

    # Check hit
    expected_norm = set(s.replace(" ", "").lower() for s in expected)
    retrieved_norm = [s.replace(" ", "").lower() for s in retrieved_standards]
    hit = any(s in expected_norm for s in retrieved_norm[:3])

    status = "HIT" if hit else "MISS"
    print(f"  [{status}] {qid}: {query[:65]}...")
    print(f"         Expected: {expected[0]}")
    print(f"         Got top5: {retrieved_standards}")
    print(f"         Latency:  {latency:.3f}s")

    results.append({
        "id": qid,
        "query": query,
        "expected_standards": expected,
        "retrieved_standards": retrieved_standards,
        "latency_seconds": round(latency, 2),
    })

# Save results
output_path = "eval_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print()
print(f"  Results saved to: {output_path}")
print("=" * 60)
