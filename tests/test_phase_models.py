import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from app.ml.markov_model import MarkovChainModel
from app.ml.random_forest_model import RandomForestPhaseModel

client = TestClient(app)

VALID_PHASES = ["1.0", "2.0", "3.1", "3.2", "3.3", "4.0", "4.5", "5.0", "6.0"]

PHASE_PAYLOAD = {
    "segment_id": "327801004",
    "subsegment": "A1",
    "current_phase": "3.1",
    "previous_phase": "2.0",
    "district_code": "3278010",
    "month": 6,
    "year": 2026,
}


def fresh_markov_model() -> MarkovChainModel:
    return MarkovChainModel(get_settings().ml_artifacts_dir)


def fresh_random_forest_model() -> RandomForestPhaseModel:
    return RandomForestPhaseModel(get_settings().ml_artifacts_dir)


def test_markov_model_loads_valid_phases_artifact():
    model = fresh_markov_model()

    assert model._valid_phases == VALID_PHASES


def test_markov_predict_returns_all_three_horizons():
    model = fresh_markov_model()

    results = model.predict("3.1")

    assert sorted(results) == [1, 2, 3]
    for predicted_phase, probability in results.values():
        assert predicted_phase in VALID_PHASES
        assert 0.0 <= probability <= 1.0


def test_markov_predict_falls_back_when_phase_missing_from_matrix():
    model = fresh_markov_model()
    model._transition_matrices[1] = model._transition_matrices[1].drop(index="3.1")

    predicted_phase, probability = model.predict("3.1")[1]

    assert predicted_phase == "3.1"
    assert probability == 0.0


def test_markov_predict_rejects_unknown_phase():
    model = fresh_markov_model()

    with pytest.raises(ValueError, match="Invalid phase"):
        model.predict("9.9")


def test_random_forest_validate_rejects_unknown_inputs():
    model = fresh_random_forest_model()

    with pytest.raises(ValueError, match="District code"):
        model.validate("3.1", "2.0", "9999999", "A1")
    with pytest.raises(ValueError, match="Current phase"):
        model.validate("9.9", "2.0", "3278010", "A1")


def test_random_forest_predict_returns_all_three_horizons():
    model = fresh_random_forest_model()

    results = model.predict(
        current_phase="3.1",
        previous_phase="2.0",
        district_code="3278010",
        subsegment="A1",
        month=6,
    )

    assert sorted(results) == [1, 2, 3]
    for predicted_phase, confidence in results.values():
        assert predicted_phase in VALID_PHASES
        assert 0.0 <= confidence <= 1.0


def test_markov_endpoint_returns_three_horizons():
    response = client.post("/api/v1/markov/predict", json=PHASE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["current_phase"] == "3.1"
    assert [p["horizon_months"] for p in body["predictions"]] == [1, 2, 3]
    assert all(0.0 <= p["transition_probability"] <= 1.0 for p in body["predictions"])


def test_markov_endpoint_rejects_unknown_phase_with_422():
    payload = {**PHASE_PAYLOAD, "current_phase": "9.9"}

    response = client.post("/api/v1/markov/predict", json=payload)

    assert response.status_code == 422
    assert "invalid" in response.json()["detail"].lower()


def test_markov_batch_endpoint_returns_predictions_per_item():
    payload = {"items": [PHASE_PAYLOAD, {**PHASE_PAYLOAD, "current_phase": "4.0"}]}

    response = client.post("/api/v1/markov/predict/batch", json=payload)

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert all(len(item["predictions"]) == 3 for item in results)


def test_markov_batch_endpoint_rejects_unknown_phase():
    payload = {"items": [{**PHASE_PAYLOAD, "current_phase": "7.7"}]}

    response = client.post("/api/v1/markov/predict/batch", json=payload)

    assert response.status_code == 422


def test_random_forest_endpoint_returns_three_horizons():
    response = client.post("/api/v1/random-forest/predict", json=PHASE_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert [p["horizon_months"] for p in body["predictions"]] == [1, 2, 3]
    assert body["predictions"][0]["target_year"] == 2026
    assert body["predictions"][0]["target_month"] == 7


def test_random_forest_endpoint_rejects_unknown_district_with_422():
    payload = {**PHASE_PAYLOAD, "district_code": "9999999"}

    response = client.post("/api/v1/random-forest/predict", json=payload)

    assert response.status_code == 422
    assert "district" in response.json()["detail"].lower()
