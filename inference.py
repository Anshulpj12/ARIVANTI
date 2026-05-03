import sys
import json
import time
import argparse
import re
from pathlib import Path

# Insert backend to path for importing modules
sys.path.insert(0, str(Path(__file__).resolve().parent / 'backend'))
import query_engine
from intent_graph import detect_intent

def extract_is_codes_from_text(text):
    """Extract all IS code references from chunk content, including part numbers."""
    full_pattern = r'IS\s*\d{2,5}\s*(?:\(Part\s*\d+\)\s*)?:\s*\d{4}'
    matches = re.findall(full_pattern, text, re.IGNORECASE)
    return matches

def main():
    parser = argparse.ArgumentParser(description="Inference script for BIS Hackathon")
    parser.add_argument("--input", required=True, help="Path to the input JSON dataset")
    parser.add_argument("--output", required=True, help="Path to save the output JSON results")
    args = parser.parse_args()

    # Initialize query engine
    query_engine.initialize()

    # Load full dataset for content cross-referencing
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
            standards_by_id = {s["chunk_id"]: s for s in dataset["standards"]}
    except FileNotFoundError:
        standards_by_id = {}
        print("Warning: dataset.json not found. Cross-referencing will be disabled.")

    # Load the input queries
    with open(args.input, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    results = []
    
    for item in test_set:
        qid = item["id"]
        query = item["query"]

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
        retrieved = query_engine.search_chunks(qvec, candidate_ids, top_k=20, score_threshold=0.10)

        # Fallback to full search
        if not retrieved and candidate_ids is not None:
            retrieved = query_engine.search_chunks(qvec, None, top_k=20, score_threshold=0.10)

        t_end = time.time()
        latency = t_end - t_start

        # Extract primary IS codes from retrieved chunks
        retrieved_standards = [r["is_code"] for r in retrieved]

        # Scan chunk content for cross-referenced IS codes
        seen = set(s.replace(" ", "").lower() for s in retrieved_standards)
        for r in retrieved:
            chunk_id = r.get("chunk_id")
            if chunk_id is not None and chunk_id in standards_by_id:
                full_std = standards_by_id[chunk_id]
                content = json.dumps(full_std)
                xrefs = extract_is_codes_from_text(content)
                for xref in xrefs:
                    xref_norm = xref.replace(" ", "").lower()
                    if xref_norm not in seen:
                        seen.add(xref_norm)
                        retrieved_standards.append(xref)

        # Trim to top 5
        retrieved_standards = retrieved_standards[:5]

        # Append to results exactly matching the required schema
        results.append({
            "id": qid,
            "query": query,
            "expected_standards": item.get("expected_standards", []),
            "retrieved_standards": retrieved_standards,
            "latency_seconds": round(latency, 2),
        })

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Successfully processed {len(results)} queries. Results saved to {args.output}")

if __name__ == "__main__":
    main()
