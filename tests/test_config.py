from customer_churn.train import load_config


def test_config_has_required_sections():
    config = load_config()

    assert "data" in config
    assert "training" in config
    assert "mlflow" in config


def test_config_has_valid_training_settings():
    config = load_config()

    assert config["training"]["test_size"] == 0.2
    assert config["training"]["random_state"] == 42
    assert config["training"]["max_iter"] == 1000


def test_config_has_churn_feature_and_target():
    config = load_config()

    assert config["data"]["feature_columns"] == ["salary"]
    assert config["data"]["target_column"] == "churn"
