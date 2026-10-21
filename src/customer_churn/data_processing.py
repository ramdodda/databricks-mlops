VALID_CHURN_VALUES = {"Yes", "No"}


def validate_customer_records(records):
    """Validate customer records and remove invalid or duplicate entries."""
    seen_ids = set()
    valid_records = []

    for record in records:
        customer_id = record.get("customer_id")
        name = record.get("name")
        salary = record.get("salary")
        churn = record.get("churn")

        if customer_id is None or not name or salary is None:
            continue

        if not isinstance(salary, (int, float)) or salary < 0:
            continue

        if churn not in VALID_CHURN_VALUES:
            continue

        if customer_id in seen_ids:
            continue

        seen_ids.add(customer_id)
        valid_records.append(record)

    return valid_records
