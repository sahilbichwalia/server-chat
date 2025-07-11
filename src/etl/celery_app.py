# src/etl/celery_app.py
from celery import Celery

celery_app = Celery(
    "etl_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.beat_schedule = {
    "run-etl-every-15-minutes": {
        "task": "src.etl.etl_job.run_etl_task",
        "schedule": 90.0
    }
}

celery_app.conf.timezone = 'Asia/Kolkata'

# ✅ This line forces task registration
import src.etl.etl_job
