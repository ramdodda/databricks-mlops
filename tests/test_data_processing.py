from customer_churn.data_processing import validate_customer_records


def test_valid_records_are_preserved():
    records = [
        {"customer_id": 101, "name": "Ravi", "salary": 50000, "churn": "Yes"},
        {"customer_id": 102, "name": "Priya", "salary": 80000, "churn": "No"},
    ]

    assert len(validate_customer_records(records)) == 2


def test_duplicate_customer_ids_are_removed():
    records = [
        {"customer_id": 101, "name": "Ravi", "salary": 50000, "churn": "Yes"},
        {"customer_id": 101, "name": "Ravi", "salary": 55000, "churn": "No"},
    ]

    result = validate_customer_records(records)

    assert len(result) == 1
    assert result[0]["salary"] == 50000


def test_invalid_records_are_removed():
    records = [
        {"customer_id": 101, "name": "Ravi", "salary": 50000, "churn": "Yes"},
        {"customer_id": 102, "name": "Anil", "salary": -100, "churn": "No"},
        {"customer_id": 103, "name": "Priya", "salary": 60000, "churn": "Unknown"},
    ]

    assert len(validate_customer_records(records)) == 1
