def check_drift(
    metrics: dict,
    metric_name: str = "accuracy",
    threshold: float = 0.75,
) -> bool:
    value = metrics.get(metric_name)
    if value is None:
        raise ValueError(f"Metric not found: {metric_name}")
    return value < threshold

