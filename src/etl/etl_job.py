from src.data_processing.data_preprocessing import process_server_data, get_fresh_server_data
from pymongo import MongoClient
from datetime import datetime
from src.etl.celery_app import celery_app
from src.common.constant import MONGO_URI, DB_NAME 

@celery_app.task(name="src.etl.etl_job.run_etl_task")
def run_etl_task():
    print("🔄 [ETL] Running periodic ETL job...")
    client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
    client.server_info()  
    db = client[DB_NAME]  

    raw_data = get_fresh_server_data()
    processed_data = process_server_data(raw_data)

    db.processed_servers.delete_many({})
    db.server_rankings.delete_many({})

    for serial, data in processed_data.items():
        db.processed_servers.insert_one({
            "serial_number": serial,
            **data,
            "etl_timestamp": datetime.utcnow()
        })

    rankings = {
        "_id": "latest",
        "etl_timestamp": datetime.utcnow(),
        "top_cpu": sorted([(k, v["peak_cpu_util"]) for k, v in processed_data.items() if v.get("peak_cpu_util")],
                          key=lambda x: x[1], reverse=True),
        "bottom_cpu": sorted([(k, v["lowest_cpu_util"]) for k, v in processed_data.items() if v.get("lowest_cpu_util")],
                             key=lambda x: x[1])
    }

    db.server_rankings.insert_one(rankings)
    print("✅ [ETL] Job completed and saved to MongoDB.")
