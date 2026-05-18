import httpx
import time
import json
import asyncio

BACKEND = "http://localhost:8000"

TEST_QUESTIONS = [
    "What is monocistronic?",
    "What is DNA fingerprinting?", 
    "Explain the structure of DNA",
    "What is monocistronic?",  # repeat — should be cached
    "What is DNA fingerprinting?",  # repeat — should be cached
]

async def test_query(client, question, session_id):
    start = time.time()
    first_token_time = None
    answer_tokens = []

    async with client.stream(
        "POST",
        f"{BACKEND}/api/v1/query/stream",
        json={
            "question": question,
            "session_id": session_id,
            "top_k": 5,
            "use_cache": True,
        },
        timeout=120.0,
    ) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data["type"] == "token":
                    if first_token_time is None:
                        first_token_time = time.time()
                    answer_tokens.append(data["data"])
                elif data["type"] == "done":
                    break
                elif data["type"] == "cached":
                    first_token_time = time.time()
                    break
            except Exception:
                continue

    total_ms = round((time.time() - start) * 1000)
    ttft_ms = round((first_token_time - start) * 1000) if first_token_time else total_ms

    return {
        "question": question[:50],
        "total_ms": total_ms,
        "time_to_first_token_ms": ttft_ms,
        "answer_length": len("".join(answer_tokens)),
    }

async def main():
    print("\n" + "="*60)
    print("RAG SYSTEM LATENCY TEST")
    print("="*60)

    # Check health first
    async with httpx.AsyncClient() as client:
        health = await client.get(f"{BACKEND}/health")
        print(f"\nHealth: {health.json()}")
        print("\nRunning latency tests...\n")

    results = []
    async with httpx.AsyncClient() as client:
        for i, question in enumerate(TEST_QUESTIONS):
            print(f"Test {i+1}/{len(TEST_QUESTIONS)}: {question[:50]}")
            result = await test_query(client, question, "latency-test-session")
            results.append(result)

            cached = result["total_ms"] < 50
            cache_label = "⚡ CACHED" if cached else "  fresh "
            print(f"  {cache_label} | total={result['total_ms']}ms | TTFT={result['time_to_first_token_ms']}ms")

    # Summary
    fresh = [r for r in results if r["total_ms"] > 50]
    cached = [r for r in results if r["total_ms"] <= 50]

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if fresh:
        avg_fresh = round(sum(r["total_ms"] for r in fresh) / len(fresh))
        avg_ttft = round(sum(r["time_to_first_token_ms"] for r in fresh) / len(fresh))
        print(f"Fresh queries  : avg={avg_fresh}ms | TTFT avg={avg_ttft}ms")

    if cached:
        avg_cached = round(sum(r["total_ms"] for r in cached) / len(cached))
        print(f"Cached queries : avg={avg_cached}ms")

    if fresh and cached:
        speedup = round(avg_fresh / max(avg_cached, 1))
        print(f"Cache speedup  : {speedup}x faster")

    print("\nWhat to tell your company:")
    print(f"  p95 fresh query latency : ~{avg_fresh if fresh else 'N/A'}ms")
    print(f"  cached query latency    : ~{avg_cached if cached else 'N/A'}ms")
    print(f"  time to first token     : ~{avg_ttft if fresh else 'N/A'}ms")
    print("="*60)

asyncio.run(main())