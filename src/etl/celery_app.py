# src/etl/celery_app.py
from celery import Celery
import os
from dotenv import load_dotenv

load_dotenv()

BROKER=os.getenv("BROKER")
BACKEND=os.getenv("BACKEND")

celery_app = Celery(
    "etl_tasks",
    broker=BROKER,
    backend=BACKEND
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
