from src.pipelines.pipeline_runner import STEP_REGISTRY


def test_pipeline_registry_contains_core_steps():
    assert set(STEP_REGISTRY) >= {
        "ingestion",
        "processing",
        "feature_engineering",
        "training",
        "evaluation",
        "deployment",
    }

