from flxbc.presentation import audit_status_label, related_experiment_run_ids, short_hash


def test_short_hash_keeps_empty_and_short_values_readable():
    assert short_hash(None) == "-"
    assert short_hash("mockchain:abc") == "mockchain:abc"


def test_short_hash_truncates_long_hashes():
    value = "mockchain:27198ecb85425780b4fa08fd65422f14"
    assert short_hash(value) == "mockchain:...422f14"


def test_audit_status_label_summarizes_valid_and_invalid_chains():
    assert audit_status_label({"valid": True, "blocks": 2}) == "有效 / 2 blocks"
    assert audit_status_label({"valid": False, "broken_at": 3}) == "异常 / block 3"


def test_related_experiment_run_ids_finds_child_runs():
    runs = [
        {"run_id": "compare-quick-fedavg"},
        {"run_id": "compare-quick-fedprox"},
        {"run_id": "other"},
    ]

    assert related_experiment_run_ids("compare-quick", runs) == [
        "compare-quick-fedavg",
        "compare-quick-fedprox",
    ]
