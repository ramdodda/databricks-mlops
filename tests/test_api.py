from fastapi.testclient import TestClient

import app as churn_app


def test_home_endpoint():
    client = TestClient(churn_app.app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_endpoint_when_model_is_loaded(monkeypatch):
    monkeypatch.setattr(churn_app, "model", object())

    client = TestClient(churn_app.app, raise_server_exceptions=False)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_predict_rejects_negative_tenure():
    client = TestClient(churn_app.app)

    response = client.post(
        "/predict",
        json={
            "tenure": -1,
            "MonthlyCharges": 80,
            "TotalCharges": 960,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
        },
    )

    assert response.status_code == 422