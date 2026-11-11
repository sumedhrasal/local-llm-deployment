import json
from pathlib import Path

import pytest

from scripts import find_model as fm

FIX = json.loads((Path(__file__).parent / "fixtures" / "hub_models.json").read_text())
GiB = 1024**3


def cand(model_id):
    f = FIX[model_id]
    return fm.candidate_from_hub(f["listing"], f["config"], f["files"])


def gpu(free_mib=24100, total_mib=24564, cc=8.6, index=0):
    return fm.GPU(index=index, name="Test GPU", total_mib=total_mib, free_mib=free_mib, compute_cap=cc)


# --- hardware detection -------------------------------------------------------

def test_detect_gpus_parses_csv():
    text = "0, NVIDIA A10G, 23028, 22500, 8.6\n1, NVIDIA A10G, 23028, 1000, 8.6\n"
    gpus = fm.parse_nvidia_smi(text)
    assert gpus == [
        fm.GPU(0, "NVIDIA A10G", 23028, 22500, 8.6),
        fm.GPU(1, "NVIDIA A10G", 23028, 1000, 8.6),
    ]


def test_detect_gpus_none(monkeypatch):
    monkeypatch.setattr(fm.shutil, "which", lambda _: None)
    assert fm.detect_gpus() == []


def test_main_without_gpu_exits_with_ollama_hint(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(fm, "detect_gpus", lambda: [])
    with pytest.raises(SystemExit) as e:
        fm.main(["--out", str(tmp_path / "deployment.json")])
    assert e.value.code == 1
    assert "fallback/ollama.sh" in capsys.readouterr().out


# --- memory math --------------------------------------------------------------

def test_kv_bytes_per_token_qwen3_8b():
    # 2 (K and V) * 36 layers * 8 KV heads * 128 head_dim * 2 bytes (fp16)
    assert fm.kv_bytes_per_token(FIX["Qwen/Qwen3-8B-AWQ"]["config"]) == 147456


def test_kv_bytes_per_token_derives_head_dim():
    # Qwen2.5-7B has no head_dim: 3584 hidden / 28 heads = 128
    assert fm.kv_bytes_per_token(FIX["Qwen/Qwen2.5-7B-Instruct-AWQ"]["config"]) == 2 * 28 * 4 * 128 * 2


def test_candidate_from_hub_reads_sizes_and_quant():
    c = cand("Qwen/Qwen3-8B-AWQ")
    assert c.quant == "awq"
    assert c.params_b == 8
    assert c.arch == "Qwen3ForCausalLM"
    assert 5.5 * GiB < c.weights_bytes < 6 * GiB
    assert c.max_context == 40960


def test_gpu_util_respects_free_memory():
    assert fm.gpu_memory_utilization(gpu(free_mib=8000, total_mib=24564)) == 0.30
    assert fm.gpu_memory_utilization(gpu(free_mib=24100, total_mib=24564)) == 0.90


def test_fit_rejects_too_big():
    assert fm.fit(cand("Qwen/Qwen3-32B-AWQ"), [gpu(free_mib=8000)]) is None


def test_fit_rejects_32b_on_free_24gb():
    # 18 GiB of weights + 8k context + overhead does not fit in 90% of 24 GB
    assert fm.fit(cand("Qwen/Qwen3-32B-AWQ"), [gpu()]) is None


def test_fit_reduces_context():
    f = fm.fit(cand("Qwen/Qwen3-8B"), [gpu()])  # 15.3 GiB bf16 weights
    assert f is not None
    assert f.max_model_len == 16384
    assert f.tensor_parallel_size == 1


def test_fit_uses_tensor_parallel_when_one_gpu_is_not_enough():
    two = [gpu(free_mib=15000, total_mib=16000, index=0), gpu(free_mib=15000, total_mib=16000, index=1)]
    f = fm.fit(cand("Qwen/Qwen3-14B-AWQ"), two[:1])
    assert f is not None and f.tensor_parallel_size == 1
    f = fm.fit(cand("Qwen/Qwen3-32B-AWQ"), two)
    assert f is not None and f.tensor_parallel_size == 2


# --- filtering ----------------------------------------------------------------

def test_skip_fp8_on_ampere():
    listing = FIX["Qwen/Qwen3-4B-Instruct-2507-FP8"]["listing"]
    assert "FP8" in fm.skip_reason(listing, [gpu(cc=8.6)])
    assert fm.skip_reason(listing, [gpu(cc=8.9)]) is None


def test_skip_gated_and_unknown_arch():
    listing = json.loads(json.dumps(FIX["Qwen/Qwen3-8B-AWQ"]["listing"]))
    listing["gated"] = "manual"
    assert "gated" in fm.skip_reason(listing, [gpu()])
    listing["gated"] = False
    listing["config"]["architectures"] = ["GPT2LMHeadModel"]
    assert "architecture" in fm.skip_reason(listing, [gpu()])


def test_skip_model_without_tool_calling_template():
    listing = json.loads(json.dumps(FIX["Qwen/Qwen3-8B-AWQ"]["listing"]))
    listing["config"]["tokenizer_config"]["chat_template"] = "{{ messages }}"
    assert "tool" in fm.skip_reason(listing, [gpu()])


# --- selection ----------------------------------------------------------------

def test_select_prefers_larger_that_fits():
    cands = [cand(m) for m in ["Qwen/Qwen3-4B-AWQ", "Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ", "Qwen/Qwen3-32B-AWQ"]]
    chosen, f, reasons = fm.select(cands, [gpu()], max_params_b=14)
    assert chosen.model_id == "Qwen/Qwen3-14B-AWQ"
    assert any("Qwen3-32B-AWQ" in r and "skip" in r for r in reasons)


def test_select_prefers_quantized_bigger_model_over_small_full_precision():
    cands = [cand("Qwen/Qwen3-8B"), cand("Qwen/Qwen3-14B-AWQ")]
    chosen, _, _ = fm.select(cands, [gpu()], max_params_b=14)
    assert chosen.model_id == "Qwen/Qwen3-14B-AWQ"


def test_select_on_small_free_memory_picks_small_model():
    cands = [cand(m) for m in ["Qwen/Qwen3-4B-AWQ", "Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"]]
    chosen, f, _ = fm.select(cands, [gpu(free_mib=8000)])
    assert chosen.model_id == "Qwen/Qwen3-4B-AWQ"
    assert f.gpu_memory_utilization == 0.30


def test_select_nothing_fits_raises():
    with pytest.raises(fm.NoModelFits):
        fm.select([cand("Qwen/Qwen3-32B-AWQ")], [gpu(free_mib=4000)])


def test_select_falls_back_when_hub_fails(monkeypatch):
    def boom(*a, **k):
        raise OSError("network unreachable")

    monkeypatch.setattr(fm, "fetch_candidates", boom)
    cands, reasons = fm.load_candidates([gpu()], limit=10)
    assert {c.model_id for c in cands} >= {"Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-4B-AWQ"}
    assert any("fallback" in r.lower() for r in reasons)


def test_forced_model_that_does_not_fit_fails(monkeypatch):
    with pytest.raises(fm.NoModelFits):
        fm.select([cand("Qwen/Qwen3-32B-AWQ")], [gpu()], forced=True)


# --- deployment card ----------------------------------------------------------

def test_make_card_has_parsers_and_fit():
    c = cand("Qwen/Qwen3-8B-AWQ")
    f = fm.fit(c, [gpu()])
    card = fm.make_card(c, f, [gpu()], ["because"])
    assert card["model_id"] == "Qwen/Qwen3-8B-AWQ"
    assert card["quantization"] == "awq"
    assert card["tool_call_parser"] == "hermes"
    assert card["reasoning_parser"] == "qwen3"
    assert card["max_model_len"] == 32768
    assert card["gpu"]["count"] == 1
    assert card["reasons"] == ["because"]
    json.dumps(card)  # serialisable


def test_skip_untrusted_publisher():
    listing = json.loads(json.dumps(FIX["Qwen/Qwen3-8B-AWQ"]["listing"]))
    listing["id"] = "nm-testing/convert_awq_w4a16_asym-e2e"
    assert "publisher" in fm.skip_reason(listing, [gpu()])


def test_skip_qwen25_base_model_without_instruct_in_name():
    listing = json.loads(json.dumps(FIX["Qwen/Qwen2.5-7B-Instruct-AWQ"]["listing"]))
    assert fm.skip_reason(listing, [gpu()]) is None
    listing["id"] = "Qwen/Qwen2.5-0.5B"
    assert "base model" in fm.skip_reason(listing, [gpu()])


def test_select_reasons_list_biggest_fitting_first():
    cands = [cand(m) for m in ["Qwen/Qwen3-4B-AWQ", "Qwen/Qwen3-32B-AWQ", "Qwen/Qwen3-14B-AWQ"]]
    _, _, reasons = fm.select(cands, [gpu()])
    assert reasons[0].startswith("✓ Qwen/Qwen3-14B-AWQ")
    assert reasons[-1].startswith("✗")


def test_select_default_caps_at_8b():
    cands = [cand(m) for m in ["Qwen/Qwen3-4B-AWQ", "Qwen/Qwen3-8B-AWQ", "Qwen/Qwen3-14B-AWQ"]]
    chosen, _, reasons = fm.select(cands, [gpu()])
    assert fm.DEFAULT_MAX_PARAMS_B == 8
    assert chosen.model_id == "Qwen/Qwen3-8B-AWQ"
    assert any("Qwen3-14B-AWQ" in r and "cap" in r for r in reasons)
    assert any("Qwen3-8B-AWQ" in r and "chosen" in r for r in reasons)


def test_same_size_prefers_variant_with_longer_context():
    # bf16 8B only fits 16k context on 24 GB; the AWQ 8B fits 32k and leaves room for more users
    chosen, f, _ = fm.select([cand("Qwen/Qwen3-8B"), cand("Qwen/Qwen3-8B-AWQ")], [gpu()])
    assert chosen.model_id == "Qwen/Qwen3-8B-AWQ"
    assert f.max_model_len == 32768


def test_forced_model_ignores_cap():
    chosen, _, _ = fm.select([cand("Qwen/Qwen3-14B-AWQ")], [gpu()], forced=True)
    assert chosen.model_id == "Qwen/Qwen3-14B-AWQ"


def test_nothing_under_cap_fits_raises():
    with pytest.raises(fm.NoModelFits):
        fm.select([cand("Qwen/Qwen3-14B-AWQ")], [gpu()], max_params_b=8)


# --- review fixes ---------------------------------------------------------------

def test_load_candidates_falls_back_when_hub_details_all_fail(monkeypatch):
    # listing works but every per-model call is rate-limited → no candidates
    monkeypatch.setattr(fm, "fetch_candidates", lambda *a, **k: [])
    cands, reasons = fm.load_candidates([gpu()], limit=10)
    assert {c.model_id for c in cands} >= {"Qwen/Qwen3-8B-AWQ"}
    assert any("fallback" in r.lower() for r in reasons)


def test_main_writes_card_on_fallback(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(fm, "detect_gpus", lambda: [gpu()])
    monkeypatch.setattr(fm, "fetch_candidates", boom)
    monkeypatch.setattr(fm, "server_running", lambda: False)
    out = tmp_path / "deployment.json"
    fm.main(["--out", str(out)])
    assert json.loads(out.read_text())["model_id"] == "Qwen/Qwen3-8B-AWQ"


def test_main_forced_model_hub_error_is_friendly(monkeypatch, capsys, tmp_path):
    import httpx
    def boom(*a, **k):
        raise httpx.ConnectError("offline")
    monkeypatch.setattr(fm, "detect_gpus", lambda: [gpu()])
    monkeypatch.setattr(fm, "fetch_one", boom)
    monkeypatch.setattr(fm, "server_running", lambda: False)
    with pytest.raises(SystemExit) as e:
        fm.main(["--model", "Qwen/Qwen3-8B-AWQ", "--out", str(tmp_path / "d.json")])
    assert e.value.code == 1
    assert "Hugging Face" in capsys.readouterr().out


def test_main_refuses_while_our_server_holds_the_gpu(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(fm, "detect_gpus", lambda: [gpu(free_mib=2000)])
    monkeypatch.setattr(fm, "server_running", lambda: True)
    with pytest.raises(SystemExit) as e:
        fm.main(["--out", str(tmp_path / "d.json")])
    assert e.value.code == 1
    assert "make stop" in capsys.readouterr().out
