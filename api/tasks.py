from api.celery_app import celery_app
from ingest.pipeline import run_ingestion_pipeline


@celery_app.task(name="api.tasks.run_ingestion_task")
def run_ingestion_task() -> None:
    run_ingestion_pipeline()
