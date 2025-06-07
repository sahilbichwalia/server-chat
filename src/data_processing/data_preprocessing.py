from src.data_processing.loader import server_data_raw
from src.config.logging_config import setup_logging
from src.common.common import DEFAULT_CARBON_INTENSITY,BASE_POWER, MAX_POWER
from datetime import datetime
from typing import Dict

logger = setup_logging()

def estimate_power(cpu_util: float) -> float:
    base_power = BASE_POWER
    max_power = MAX_POWER
    return base_power + ((max_power - base_power) * cpu_util / 100.0)

# -----------------------------
# Data Processing Functions
# -----------------------------
def process_server_data() -> dict[str, dict]: # Replaced typing.Dict with dict
    processed_data = {}
    if not server_data_raw:
        logger.warning("No raw server data to process.")
        return processed_data

    logger.info(f"Processing data for {len(server_data_raw)} servers")

    for server_item in server_data_raw:
        serial_number = server_item.get("serial_number")
        if not serial_number:
            continue

        power_data_entries = server_item.get("power", [])
        if not power_data_entries:
            continue

        cpu_utils = []
        temps = []
        peaks = []
        timestamps = []
        records = []
        estimated_energy_kwh = 0.0
        estimated_powers = []

        valid_entries_for_sorting = []
        for entry in power_data_entries:
            timestamp_str = entry.get("time", "")
            if timestamp_str:
                try:
                    timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M:%S")
                    valid_entries_for_sorting.append((timestamp, entry))
                except ValueError:
                    logger.warning(f"Invalid timestamp format for server {serial_number}, entry skipped: {timestamp_str}")
                    continue

        valid_entries_for_sorting.sort(key=lambda x: x[0])
        sorted_power_data_entries = [item[1] for item in valid_entries_for_sorting]


        for idx, entry in enumerate(sorted_power_data_entries):
            try:
                timestamp_str = entry.get("time", "")
                timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y, %H:%M:%S")

                cpu_util = entry.get("cpu_util")
                amb_temp = entry.get("amb_temp")
                peak = entry.get("peak")

                est_power_val = None
                if isinstance(cpu_util, (int, float)):
                    est_power_val = estimate_power(float(cpu_util))
                    estimated_powers.append(est_power_val)

                if idx > 0 and est_power_val is not None and timestamps: # Check est_power_val is not None
                    prev_time = timestamps[-1]
                    delta_hours = (timestamp - prev_time).total_seconds() / 3600.0
                    if delta_hours > 0:
                         estimated_energy_kwh += (est_power_val * delta_hours) / 1000.0 # est_power_val is in Watts, so this is Wh. Division by 1000 makes it kWh.

                if isinstance(cpu_util, (int, float)):
                    cpu_utils.append(float(cpu_util))
                if isinstance(amb_temp, (int, float)):
                    temps.append(float(amb_temp))
                if isinstance(peak, (int, float)):
                    peaks.append(float(peak))

                timestamps.append(timestamp)
                records.append({
                    "time": timestamp,
                    "time_str": timestamp_str,
                    "cpu_util": cpu_util,
                    "amb_temp": amb_temp,
                    "peak": peak,
                    "power_consumption": entry.get("power_consumption"),
                    "temperature": entry.get("temperature"),
                    "fan_speed": entry.get("fan_speed"),
                    "cpu_watts": entry.get("cpu_watts"),
                    "dimm_watts": entry.get("dimm_watts"),
                    "estimated_power": est_power_val
                })
            except (ValueError, TypeError) as e:
                logger.warning(f"Error processing entry for server {serial_number}: {e}")
                continue

        if not records:
            continue

        avg_cpu = round(sum(cpu_utils) / len(cpu_utils), 2) if cpu_utils else None
        peak_cpu_util_val = max(cpu_utils) if cpu_utils else None
        peak_cpu_record = next((r for r in records if r["cpu_util"] == peak_cpu_util_val), None) if peak_cpu_util_val is not None else None
        lowest_cpu_util_val = min(cpu_utils) if cpu_utils else None
        lowest_cpu_record = next((r for r in records if r["cpu_util"] == lowest_cpu_util_val), None) if lowest_cpu_util_val is not None else None

        avg_est_power = round(sum(estimated_powers) / len(estimated_powers), 2) if estimated_powers else None


        max_amb_temp = max(temps) if temps else None
        max_temp_record = next((r for r in records if r["amb_temp"] == max_amb_temp), None) if max_amb_temp is not None else None
        min_amb_temp = min(temps) if temps else None
        min_temp_record = next((r for r in records if r["amb_temp"] == min_amb_temp), None) if min_amb_temp is not None else None

        max_peak = max(peaks) if peaks else None
        max_peak_record = next((r for r in records if r["peak"] == max_peak), None) if max_peak is not None else None
        min_peak = min(peaks) if peaks else None
        min_peak_record = next((r for r in records if r["peak"] == min_peak), None) if min_peak is not None else None

        latest_record = max(records, key=lambda x: x["time"]) if records else None

        co2_emissions = {
            grid_type: round(estimated_energy_kwh * intensity, 3)
            for grid_type, intensity in DEFAULT_CARBON_INTENSITY.items()
        }

        processed_data[serial_number] = {
            "avg_cpu_util": avg_cpu,
            "peak_cpu_util": peak_cpu_util_val,
            "peak_cpu_record": peak_cpu_record,
            "lowest_cpu_util": lowest_cpu_util_val,
            "lowest_cpu_record": lowest_cpu_record,
            "avg_est_power": avg_est_power,
            "max_amb_temp": max_amb_temp,
            "max_temp_record": max_temp_record,
            "min_amb_temp": min_amb_temp,
            "min_temp_record": min_temp_record,
            "max_peak": max_peak,
            "max_peak_record": max_peak_record,
            "min_peak": min_peak,
            "min_peak_record": min_peak_record,
            "latest_record": latest_record,
            "estimated_energy_kwh": round(estimated_energy_kwh, 3),
            "co2_emissions": co2_emissions,
            "all_records": records
        }

    logger.info(f"Successfully processed data for {len(processed_data)} servers")
    return processed_data

processed_server_data = process_server_data()

server_rankings = {
    "top_cpu": sorted(
        [(k, v["peak_cpu_util"]) for k, v in processed_server_data.items() if v.get("peak_cpu_util") is not None],
        key=lambda x: x[1],
        reverse=True
    ),
    "bottom_cpu": sorted(
        [(k, v["lowest_cpu_util"]) for k, v in processed_server_data.items() if v.get("lowest_cpu_util") is not None],
        key=lambda x: x[1]
    ),
    "top_amb_temp": sorted(
        [(k, v["max_amb_temp"]) for k, v in processed_server_data.items() if v.get("max_amb_temp") is not None],
        key=lambda x: x[1],
        reverse=True
    ),
    "bottom_amb_temp": sorted(
        [(k, v["min_amb_temp"]) for k, v in processed_server_data.items() if v.get("min_amb_temp") is not None],
        key=lambda x: x[1]
    ),
    "top_peak": sorted(
        [(k, v["max_peak"]) for k, v in processed_server_data.items() if v.get("max_peak") is not None],
        key=lambda x: x[1],
        reverse=True
    ),
    "bottom_peak": sorted(
        [(k, v["min_peak"]) for k, v in processed_server_data.items() if v.get("min_peak") is not None],
        key=lambda x: x[1]
    )
}

logger.info("Server rankings initialized.")