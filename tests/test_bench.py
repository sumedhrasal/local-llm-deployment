import pytest

from scripts.bench import ReqResult, compare_table, summarize


def test_summarize_percentiles():
    results = [ReqResult(latency_s=float(i), ttft_s=i / 10, output_tokens=100, ok=True) for i in range(1, 11)]
    s = summarize(results, wall_s=5.0, concurrency=4)
    assert s["requests"] == 10
    assert s["concurrency"] == 4
    assert s["req_per_s"] == pytest.approx(2.0)
    assert s["output_tok_per_s"] == pytest.approx(200.0)
    assert s["p50_latency_s"] == pytest.approx(5.5)
    assert s["p95_latency_s"] == pytest.approx(9.55)
    assert s["p50_ttft_s"] == pytest.approx(0.55)


def test_summarize_ignores_failed():
    results = [ReqResult(1.0, 0.1, 50, True), ReqResult(99.0, 99.0, 0, False)]
    s = summarize(results, wall_s=1.0, concurrency=2)
    assert s["requests"] == 1
    assert s["failed"] == 1
    assert s["output_tok_per_s"] == pytest.approx(50.0)
    assert s["p95_latency_s"] == pytest.approx(1.0)


def test_summarize_all_failed():
    s = summarize([ReqResult(1.0, 1.0, 0, False)], wall_s=1.0, concurrency=1)
    assert s["requests"] == 0 and s["failed"] == 1 and s["output_tok_per_s"] == 0


def test_compare_table_shows_ratio():
    a = {"engine": "vllm", "model": "m1", "requests": 64, "failed": 0, "concurrency": 32, "req_per_s": 4.0,
         "output_tok_per_s": 1000.0, "p50_latency_s": 5.0, "p95_latency_s": 7.0, "p50_ttft_s": 0.2}
    b = dict(a, engine="ollama", model="m2", req_per_s=1.0, output_tok_per_s=250.0)
    table = compare_table([a, b])
    assert "vllm" in table and "ollama" in table
    assert "4.0x" in table


def test_ollama_on_gpu_reads_size_vram():
    from scripts.bench import ollama_on_gpu
    ps = {"models": [{"name": "qwen3:8b", "size": 6_000_000_000, "size_vram": 6_000_000_000}]}
    assert ollama_on_gpu(ps, "qwen3:8b") is True
    ps["models"][0]["size_vram"] = 0
    assert ollama_on_gpu(ps, "qwen3:8b") is False
    assert ollama_on_gpu({"models": []}, "qwen3:8b") is False
