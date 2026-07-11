import json

from src.clean import normalize_profile
from src.config import load


def test_sentinel_gets_identity_missing_and_null_age(spark, tmp_path):
    records = [
        {"age": 118, "registered_on": "20180101", "gender": None, "id": "sentinel", "credit_card_limit": None},
        {"age": 35, "registered_on": "20180101", "gender": "F", "id": "normal", "credit_card_limit": 5000},
    ]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(records))
    cfg = load(raw_dir=tmp_path)

    df = spark.read.option("multiLine", True).json(str(cfg.profile_path))
    result = {row["account_id"]: row for row in normalize_profile(df, cfg).collect()}

    assert result["sentinel"]["identity_missing"] == 1
    assert result["sentinel"]["age"] is None
    assert result["normal"]["identity_missing"] == 0
    assert result["normal"]["age"] == 35


def test_missing_gender_becomes_unknown_and_tenure_is_computed(spark, tmp_path):
    records = [
        {"age": 40, "registered_on": "20180101", "gender": None, "id": "acc1", "credit_card_limit": 1000},
    ]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(records))
    cfg = load(raw_dir=tmp_path, test_start_date="20180201")

    df = spark.read.option("multiLine", True).json(str(cfg.profile_path))
    row = normalize_profile(df, cfg).collect()[0]

    assert row["gender"] == "unknown"
    assert row["tenure_days"] == 31
