from src.data_processing.data_preprocessing import processed_server_data,server_rankings,estimate_power
from src.common.common import EFFICIENCY_THRESHOLDS, DEFAULT_CARBON_INTENSITY
from src.config.logging_config import setup_logging
import re
from typing import  List, Optional # Replaced with modern builtins
from word2number import w2n
from collections import Counter
import json


logger=setup_logging()


# -----------------------------
# Tool Functions
# -----------------------------
def list_servers(_:str) -> str:
    if not processed_server_data:
        return "Error: No server data available."
    serials = list(processed_server_data.keys())
    if not serials:
        return "No servers found in the processed data."
    result = f"Available Servers ({len(serials)} total):\n" + "=" * 40 + "\n\n"
    for i, serial in enumerate(serials, 1):
        server_info = processed_server_data.get(serial)
        result += f"{i:2d}. Server {serial}:\n"
        if not server_info:
            result += "   - Data incomplete.\n\n"
            continue
        latest_rec = server_info.get("latest_record")
        time_str = latest_rec['time_str'] if latest_rec else "N/A"
        avg_cpu = server_info.get("avg_cpu_util")
        all_temps = [rec['amb_temp'] for rec in server_info.get("all_records", []) if rec.get('amb_temp') is not None]
        avg_temp = round(sum(all_temps) / len(all_temps), 1) if all_temps else None
        record_count = len(server_info.get("all_records", []))
        
        result += f"   - Last Seen: {time_str}\n"
        result += f"   - Total Records: {record_count}\n"
        
        avg_cpu_str = f"{avg_cpu:.1f}%" if avg_cpu is not None else "N/A"
        result += f"   - Average CPU: {avg_cpu_str}\n"
        
        avg_temp_str = f"{avg_temp:.1f}°C" if avg_temp is not None else "N/A"
        result += f"   - Average Ambient Temp: {avg_temp_str}\n"
        
        result += "\n" # Single blank line separator between servers
    return result


def extract_server_count(text: str, default: int = 10) -> float:
    if not text or not isinstance(text, str):
        return float(default)
    text_clean = text.lower().strip()
    if any(word in text_clean for word in ['all', 'every', 'everything']):
        return float('inf')
    
    digits = re.findall(r'\d+', text.replace(',', ''))
    if digits:
        try:
            return float(digits[0]) # Use the first number found
        except ValueError: 
            pass 

    try:
        # Keep this word removal conservative to avoid removing parts of numbers
        cleaned_words = [word for word in text_clean.split() if word not in {'top', 'show', 'give', 'me', 'the', 'first', 'last', 'highest', 'lowest', 'servers', 'server'}]
        if cleaned_words:
            try:
                return float(w2n.word_to_num(" ".join(cleaned_words)))
            except ValueError:
                for word in cleaned_words: 
                    try:
                        return float(w2n.word_to_num(word))
                    except ValueError:
                        continue
    except Exception as e: 
        logger.debug(f"Word to number conversion failed for '{text}': {e}")
    return float(default)

def get_top_servers_by_cpu_util(query: str = "") -> str:
    if not server_rankings.get("top_cpu"):
        return "No CPU utilization data available for ranking top servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["top_cpu"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for top CPU utilization."
    result_header = "Server with highest peak CPU utilization:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by highest peak CPU utilization:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by highest peak CPU utilization:\n\n"
    result = result_header
    for i, (serial, peak_cpu) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("peak_cpu_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        peak_record = server_info["peak_cpu_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Peak CPU: {peak_cpu}%\n"
        result += f"   - Timestamp: {peak_record['time_str']}\n"
        result += f"   - Power: {peak_record.get('power_consumption', 'N/A')}W\n"
        result += f"   - Temperature: {peak_record.get('temperature', 'N/A')}°C\n"
        result += f"   - Fan Speed: {peak_record.get('fan_speed', 'N/A')} RPM\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result
def get_specific_server_cpu_utilization(query: str) -> str:
    """
    Get CPU utilization data for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - CPU utilization request
    
    Returns:
        Formatted string with CPU utilization data for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    found_servers = []
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'CPU utilization for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} CPU utilization'")
    
    # Get CPU utilization data for all found servers
    results = []
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        
        # Check if CPU utilization data exists
        if not server_data.get("peak_cpu_record"):
            results.append({
                "serial": server_serial,
                "error": "No CPU utilization data available"
            })
            continue
        
        peak_cpu_record = server_data["peak_cpu_record"]
        
        # Get the peak CPU utilization from server rankings if available
        peak_cpu_util = None
        if server_rankings.get("top_cpu"):
            for serial, cpu_util in server_rankings["top_cpu"]:
                if serial == server_serial:
                    peak_cpu_util = cpu_util
                    break
        
        # Fallback to peak_cpu_record if not found in rankings
        if peak_cpu_util is None:
            peak_cpu_util = peak_cpu_record.get("cpu_util", "N/A")
        
        # Calculate efficiency rating
        avg_cpu = server_data.get("avg_cpu_util", 0)
        if avg_cpu == 0:
            efficiency = "idle"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
        
        # Get server ranking position if available
        ranking_position = "N/A"
        if server_rankings.get("top_cpu"):
            for i, (serial, _) in enumerate(server_rankings["top_cpu"]):
                if serial == server_serial:
                    ranking_position = f"#{i+1}"
                    break
        
        servers_with_data += 1
        
        results.append({
            "serial": server_serial,
            "peak_cpu_util": peak_cpu_util,
            "avg_cpu_util": avg_cpu,
            "timestamp": peak_cpu_record.get("time_str", "N/A"),
            "power_consumption": peak_cpu_record.get("power_consumption", "N/A"),
            "temperature": peak_cpu_record.get("temperature", "N/A"),
            "fan_speed": peak_cpu_record.get("fan_speed", "N/A"),
            "efficiency": efficiency,
            "ranking_position": ranking_position
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"CPU utilization data for server {result['serial']}:\n\n"
        output += f"   - Peak CPU Utilization: {result['peak_cpu_util']}%\n"
        output += f"   - Average CPU Utilization: {result['avg_cpu_util']}%\n"
        output += f"   - Timestamp of Peak: {result['timestamp']}\n"
        output += f"   - Power Consumption at Peak: {result['power_consumption']}W\n"
        output += f"   - Temperature at Peak: {result['temperature']}°C\n"
        output += f"   - Fan Speed at Peak: {result['fan_speed']} RPM\n"
        output += f"   - CPU Efficiency Rating: {result['efficiency'].capitalize()}\n"
        output += f"   - Fleet Ranking: {result['ranking_position']}\n"
        
    else:
        # Multiple servers output
        output = f"CPU utilization data for {len(found_servers)} specified servers:\n\n"
        
        if servers_with_data > 0:
            # Calculate summary statistics
            valid_peak_cpus = [r["peak_cpu_util"] for r in results if "error" not in r and r["peak_cpu_util"] is not None]
            valid_avg_cpus = [r["avg_cpu_util"] for r in results if "error" not in r and r["avg_cpu_util"] is not None]
            
            if valid_peak_cpus:
                highest_peak = max(valid_peak_cpus)
                lowest_peak = min(valid_peak_cpus)
                avg_peak = sum(valid_peak_cpus) / len(valid_peak_cpus)
                
                output += f"   - Highest Peak CPU Among Servers: {highest_peak}%\n"
                output += f"   - Lowest Peak CPU Among Servers: {lowest_peak}%\n"
                output += f"   - Average Peak CPU: {round(avg_peak, 2)}%\n"
            
            if valid_avg_cpus:
                overall_avg = sum(valid_avg_cpus) / len(valid_avg_cpus)
                output += f"   - Overall Average CPU Utilization: {round(overall_avg, 2)}%\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']} (Rank: {result['ranking_position']}):\n"
                output += f"   - Peak CPU: {result['peak_cpu_util']}%\n"
                output += f"   - Average CPU: {result['avg_cpu_util']}%\n"
                output += f"   - Peak Timestamp: {result['timestamp']}\n"
                output += f"   - Peak Power: {result['power_consumption']}W\n"
                output += f"   - Peak Temperature: {result['temperature']}°C\n"
                output += f"   - Peak Fan Speed: {result['fan_speed']} RPM\n"
                output += f"   - Efficiency: {result['efficiency'].capitalize()}\n\n"
    
    return output


def get_lowest_servers_by_cpu_util(query: str = "") -> str:
    if not server_rankings.get("bottom_cpu"):
        return "No CPU utilization data available for ranking lowest servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["bottom_cpu"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for lowest CPU utilization."
    result_header = "Server with lowest peak CPU utilization:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by lowest peak CPU utilization:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by lowest peak CPU utilization:\n\n"
    result = result_header
    for i, (serial, lowest_cpu) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("lowest_cpu_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        lowest_record = server_info["lowest_cpu_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Lowest CPU: {lowest_cpu}%\n"
        result += f"   - Timestamp: {lowest_record['time_str']}\n"
        result += f"   - Power: {lowest_record.get('power_consumption', 'N/A')}W\n"
        result += f"   - Temperature: {lowest_record.get('temperature', 'N/A')}°C\n"
        result += f"   - Fan Speed: {lowest_record.get('fan_speed', 'N/A')} RPM\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_top_servers_by_ambient_temp(query: str = "") -> str:
    if not server_rankings.get("top_amb_temp"):
        return "No ambient temperature data available for ranking top servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["top_amb_temp"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for top ambient temperature."
    result_header = "Server with highest ambient temperature:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by highest ambient temperature:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by highest ambient temperature:\n\n"
    result = result_header
    for i, (serial, max_temp) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("max_temp_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        temp_record = server_info["max_temp_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Highest Ambient Temperature: {max_temp}°C\n"
        result += f"   - Timestamp: {temp_record['time_str']}\n"
        result += f"   - CPU Utilization: {temp_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - CPU Power: {temp_record.get('cpu_watts', 'N/A')}W\n"
        result += f"   - DIMM Power: {temp_record.get('dimm_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_specific_server_ambient_temp(query: str) -> str:
    """
    Get ambient temperature data for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - Ambient temperature request
    
    Returns:
        Formatted string with ambient temperature data for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    found_servers = []
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'ambient temperature for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} ambient temperature'")
    
    # Get ambient temperature data for all found servers
    results = []
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        
        # Check if ambient temperature data exists
        if not server_data.get("max_temp_record"):
            results.append({
                "serial": server_serial,
                "error": "No ambient temperature data available"
            })
            continue
        
        temp_record = server_data["max_temp_record"]
        # Get the max ambient temperature from server rankings if available
        max_ambient_temp = None
        if server_rankings.get("top_amb_temp"):
            for serial, temp in server_rankings["top_amb_temp"]:
                if serial == server_serial:
                    max_ambient_temp = temp
                    break
        
        # Fallback to temp_record if not found in rankings
        if max_ambient_temp is None:
            max_ambient_temp = temp_record.get("amb_temp", "N/A")
        
        servers_with_data += 1
        
        results.append({
            "serial": server_serial,
            "max_ambient_temp": max_ambient_temp,
            "timestamp": temp_record.get("time_str", "N/A"),
            "cpu_util": temp_record.get("cpu_util", "N/A"),
            "cpu_watts": temp_record.get("cpu_watts", "N/A"),
            "dimm_watts": temp_record.get("dimm_watts", "N/A"),
            "avg_ambient_temp": server_data.get("avg_ambient_temp", "N/A")
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"Ambient temperature data for server {result['serial']}:\n\n"
        output += f"   - Highest Ambient Temperature: {result['max_ambient_temp']}°C\n"
        output += f"   - Timestamp of Peak: {result['timestamp']}\n"
        output += f"   - Average Ambient Temperature: {result['avg_ambient_temp']}°C\n"
        output += f"   - CPU Utilization at Peak: {result['cpu_util']}%\n"
        output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n"
        output += f"   - DIMM Power at Peak: {result['dimm_watts']}W\n"
        
    else:
        # Multiple servers output
        output = f"Ambient temperature data for {len(found_servers)} specified servers:\n\n"
        
        if servers_with_data > 0:
            # Calculate summary statistics
            valid_max_temps = [r["max_ambient_temp"] for r in results if "error" not in r and r["max_ambient_temp"] is not None]
            if valid_max_temps:
                avg_max_temp = sum(valid_max_temps) / len(valid_max_temps)
                highest_temp = max(valid_max_temps)
                lowest_temp = min(valid_max_temps)
                
                output += f"   - Highest Temperature Among Servers: {highest_temp}°C\n"
                output += f"   - Lowest Temperature Among Servers: {lowest_temp}°C\n"
                output += f"   - Average Maximum Temperature: {round(avg_max_temp, 2)}°C\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']}:\n"
                output += f"   - Highest Ambient Temp: {result['max_ambient_temp']}°C\n"
                output += f"   - Average Ambient Temp: {result['avg_ambient_temp']}°C\n"
                output += f"   - Peak Timestamp: {result['timestamp']}\n"
                output += f"   - CPU Util at Peak: {result['cpu_util']}%\n"
                output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n"
                output += f"   - DIMM Power at Peak: {result['dimm_watts']}W\n\n"
    
    return output

def get_lowest_servers_by_ambient_temp(query: str = "") -> str:
    if not server_rankings.get("bottom_amb_temp"):
        return "No ambient temperature data available for ranking lowest servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["bottom_amb_temp"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for lowest ambient temperature."
    result_header = "Server with lowest ambient temperature:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by lowest ambient temperature:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by lowest ambient temperature:\n\n"
    result = result_header
    for i, (serial, min_temp) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("min_temp_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        temp_record = server_info["min_temp_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Lowest Ambient Temperature: {min_temp}°C\n"
        result += f"   - Timestamp: {temp_record['time_str']}\n"
        result += f"   - CPU Utilization: {temp_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - CPU Power: {temp_record.get('cpu_watts', 'N/A')}W\n"
        result += f"   - DIMM Power: {temp_record.get('dimm_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_top_servers_by_peak(query: str = "") -> str:
    if not server_rankings.get("top_peak"):
        return "No peak data available for ranking top servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["top_peak"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for top peak values."
    result_header = "Server with highest peak value:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by highest peak value:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by highest peak value:\n\n"
    result = result_header
    for i, (serial, max_peak_val) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("max_peak_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        peak_record = server_info["max_peak_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Highest Peak Value: {max_peak_val}\n"
        result += f"   - Timestamp: {peak_record['time_str']}\n"
        result += f"   - CPU Utilization: {peak_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - Ambient Temperature: {peak_record.get('amb_temp', 'N/A')}°C\n"
        result += f"   - CPU Power: {peak_record.get('cpu_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def get_specific_server_peak_data(query: str) -> str:
    """
    Get peak data for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - Peak data request
    
    Returns:
        Formatted string with peak data for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    found_servers = []
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'peak data for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} peak values'")
    
    # Get peak data for all found servers
    results = []
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        
        # Check if peak data exists
        if not server_data.get("max_peak_record"):
            results.append({
                "serial": server_serial,
                "error": "No peak data available"
            })
            continue
        
        peak_record = server_data["max_peak_record"]
        # Get the max peak value from server rankings if available
        max_peak_value = None
        if server_rankings.get("top_peak"):
            for serial, peak_val in server_rankings["top_peak"]:
                if serial == server_serial:
                    max_peak_value = peak_val
                    break
        
        # Fallback to peak_record if not found in rankings
        if max_peak_value is None:
            max_peak_value = peak_record.get("peak_value", "N/A")
        
        servers_with_data += 1
        
        results.append({
            "serial": server_serial,
            "max_peak_value": max_peak_value,
            "timestamp": peak_record.get("time_str", "N/A"),
            "cpu_util": peak_record.get("cpu_util", "N/A"),
            "amb_temp": peak_record.get("amb_temp", "N/A"),
            "cpu_watts": peak_record.get("cpu_watts", "N/A"),
            "avg_peak_value": server_data.get("avg_peak_value", "N/A")
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"Peak data for server {result['serial']}:\n\n"
        output += f"   - Highest Peak Value: {result['max_peak_value']}\n"
        output += f"   - Timestamp of Peak: {result['timestamp']}\n"
        output += f"   - Average Peak Value: {result['avg_peak_value']}\n"
        output += f"   - CPU Utilization at Peak: {result['cpu_util']}%\n"
        output += f"   - Ambient Temperature at Peak: {result['amb_temp']}°C\n"
        output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n"
        
    else:
        # Multiple servers output
        output = f"Peak data for {len(found_servers)} specified servers:\n\n"
        
        if servers_with_data > 0:
            # Calculate summary statistics
            valid_peak_values = [r["max_peak_value"] for r in results if "error" not in r and r["max_peak_value"] is not None]
            if valid_peak_values:
                avg_peak = sum(valid_peak_values) / len(valid_peak_values)
                highest_peak = max(valid_peak_values)
                lowest_peak = min(valid_peak_values)
                
                output += f"   - Highest Peak Value Among Servers: {highest_peak}\n"
                output += f"   - Lowest Peak Value Among Servers: {lowest_peak}\n"
                output += f"   - Average Peak Value: {round(avg_peak, 2)}\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']}:\n"
                output += f"   - Highest Peak Value: {result['max_peak_value']}\n"
                output += f"   - Average Peak Value: {result['avg_peak_value']}\n"
                output += f"   - Peak Timestamp: {result['timestamp']}\n"
                output += f"   - CPU Util at Peak: {result['cpu_util']}%\n"
                output += f"   - Ambient Temp at Peak: {result['amb_temp']}°C\n"
                output += f"   - CPU Power at Peak: {result['cpu_watts']}W\n\n"
    
    return output


def get_lowest_servers_by_peak(query: str = "") -> str:
    if not server_rankings.get("bottom_peak"):
        return "No peak data available for ranking lowest servers."
    num_servers_float = extract_server_count(query)
    ranked_servers = server_rankings["bottom_peak"]
    available_servers = len(ranked_servers)
    if num_servers_float == float('inf'):
        actual_count = available_servers
        num_to_show = available_servers
    else:
        num_to_show = int(num_servers_float)
        actual_count = min(num_to_show, available_servers)
    if actual_count == 0: return "No servers match the criteria for lowest peak values."
    result_header = "Server with lowest peak value:\n\n" if num_to_show == 1 and actual_count == 1 else \
                    f"All {actual_count} servers by lowest peak value:\n\n" if num_servers_float == float('inf') else \
                    f"Top {actual_count} of {num_to_show} requested servers by lowest peak value:\n\n"
    result = result_header
    for i, (serial, min_peak_val) in enumerate(ranked_servers[:actual_count]):
        server_info = processed_server_data.get(serial)
        if not server_info or not server_info.get("min_peak_record"):
            result += f"{i+1}. Server {serial}: Data incomplete.\n\n"
            continue
        peak_record = server_info["min_peak_record"]
        result += f"{i+1}. Server {serial}:\n"
        result += f"   - Lowest Peak Value: {min_peak_val}\n"
        result += f"   - Timestamp: {peak_record['time_str']}\n"
        result += f"   - CPU Utilization: {peak_record.get('cpu_util', 'N/A')}%\n"
        result += f"   - Ambient Temperature: {peak_record.get('amb_temp', 'N/A')}°C\n"
        result += f"   - CPU Power: {peak_record.get('cpu_watts', 'N/A')}W\n\n"
    if num_to_show > available_servers and num_servers_float != float('inf'):
        result += f"Note: Requested {num_to_show}, but only {available_servers} servers have relevant data.\n"
    return result

def calculate_carbon_footprint(query: str) -> str:
    """
    Calculate carbon footprint for multiple servers or fleet-wide analysis.
    Handles requests for all servers or top N servers with different grid types.
    
    Args:
        query: Natural language query that may contain:
               - Carbon intensity type (e.g. "low carbon", "average", "high carbon")
               - Number of servers to show (e.g. "top 5", "all servers")
    
    Returns:
        Formatted string with carbon footprint results for multiple servers
    """
    if not processed_server_data:
        return "No server data available."
    
    # Default values
    carbon_intensity = 'average_grid'
    num_servers_to_show = extract_server_count(query, default=10)
    
    # Parse carbon intensity preference
    query_lower = query.lower()
    if "low carbon" in query_lower or "renewable" in query_lower:
        carbon_intensity = 'low_carbon_grid'
    elif "high carbon" in query_lower or "coal" in query_lower:
        carbon_intensity = 'high_carbon_grid'
    
    # Validate carbon intensity input
    if carbon_intensity not in DEFAULT_CARBON_INTENSITY:
        return f"Invalid carbon intensity. Choose from: {', '.join(DEFAULT_CARBON_INTENSITY.keys())}"
    
    intensity_factor = DEFAULT_CARBON_INTENSITY[carbon_intensity]
    total_co2 = 0.0
    results = []
    
    # Process all servers
    for serial in processed_server_data.keys():
        server_data = processed_server_data[serial]
        energy_kwh = server_data.get("estimated_energy_kwh", 0)
        
        # Skip servers with insufficient data
        if energy_kwh == 0:
            continue
        
        # Calculate CO2 emissions
        co2_kg = energy_kwh * intensity_factor
        total_co2 += co2_kg
        
        # Get efficiency rating
        avg_cpu = server_data["avg_cpu_util"]
        
        # Special case for 0% utilization
        if avg_cpu == 0:
            efficiency = "idle"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
        
        results.append({
            "serial": serial,
            "energy_kwh": round(energy_kwh, 2),
            "co2_kg": round(co2_kg, 2),
            "avg_cpu": avg_cpu,
            "efficiency": efficiency,
            "carbon_intensity": carbon_intensity
        })
    
    if not results:
        return "No valid server data available for carbon footprint calculation."
    
    # Multiple servers summary output
    grid_type_display = carbon_intensity.replace('_', ' ').title()
    available_servers = len(results)
    
    output = f"Carbon footprint summary for all {available_servers} servers ({grid_type_display}):\n\n"
    output += f"   - Total CO2 Emissions: {round(total_co2, 2)} kg\n"
    output += f"   - Average per Server: {round(total_co2/available_servers, 2)} kg\n\n"
    
    if num_servers_to_show == float('inf'):
        # Show all servers
        top_count = available_servers
        output += f"All {available_servers} servers:\n\n"
    else:
        # Show requested number of servers
        top_count = min(int(num_servers_to_show), available_servers)
        output += f"Top {top_count} highest emitting servers:\n\n"
    
    # Add top N highest emitters
    sorted_results = sorted(results, key=lambda x: x["co2_kg"], reverse=True)
    
    for i, res in enumerate(sorted_results[:top_count]):
        output += f"{i+1}. Server {res['serial']}:\n"
        output += f"   - CO2 Emissions: {res['co2_kg']} kg\n"
        output += f"   - Energy Consumed: {res['energy_kwh']} kWh\n"
        output += f"   - CPU Utilization: {res['avg_cpu']}%\n"
        output += f"   - Efficiency Rating: {res['efficiency'].capitalize()}\n\n"
            
    # Add efficiency distribution
    eff_dist = {}
    for res in results:
        eff_dist[res["efficiency"]] = eff_dist.get(res["efficiency"], 0) + 1
    
    output += "Energy efficiency distribution:\n\n"
    for eff, count in sorted(eff_dist.items()):
        percentage = round((count / available_servers) * 100, 1)
        output += f"   - {eff.capitalize()}: {count} servers ({percentage}%)\n"

    return output


def co2_emission_server(query: str) -> str:
    """
    Calculate carbon footprint for specific server(s) with robust server identification.
    Can handle single or multiple servers in one query.
    
    Args:
        query: Natural language query containing:
               - One or more server serial numbers (e.g. "server SGH227WTNK and ABC123", "SGH227WTNK, DEF456")
               - Optional carbon intensity type (e.g. "low carbon", "average", "high carbon")
    
    Returns:
        Formatted string with carbon footprint results for the specified server(s)
        or error message if server(s) not found
    """
    if not processed_server_data:
        return "No server data available."
    
    # Default carbon intensity
    carbon_intensity = 'average_grid'
    found_servers = []
    
    # Parse carbon intensity preference
    query_lower = query.lower()
    if "low carbon" in query_lower or "renewable" in query_lower:
        carbon_intensity = 'low_carbon_grid'
    elif "high carbon" in query_lower or "coal" in query_lower:
        carbon_intensity = 'high_carbon_grid'
    
    # Validate carbon intensity input
    if carbon_intensity not in DEFAULT_CARBON_INTENSITY:
        return f"Invalid carbon intensity. Choose from: {', '.join(DEFAULT_CARBON_INTENSITY.keys())}"
    
    # Robust server serial extraction with multiple patterns
    server_patterns = [
        r'server\s+([A-Za-z0-9_-]+)',  # "server ABC123"
        r'([A-Za-z0-9]{6,})',          # Direct alphanumeric codes like "SGH227WTNK"
        r'([A-Za-z0-9_-]{5,})'         # Alphanumeric with underscores/hyphens
    ]
    
    # Collect all potential server matches
    potential_servers = set()
    
    # Try each pattern to find all server serials
    for pattern in server_patterns:
        matches = re.findall(pattern, query, re.IGNORECASE)
        for match in matches:
            potential_serial = match.upper().strip()
            # Check if this serial exists in our data
            if potential_serial in processed_server_data:
                potential_servers.add(potential_serial)
    
    # Fallback: check if any known server serial appears directly in query
    query_upper = query.upper()
    for serial_key in processed_server_data.keys():
        # Check for exact match or serial as substring with word boundaries
        if re.search(r'\b' + re.escape(serial_key) + r'\b', query_upper):
            potential_servers.add(serial_key)
    
    # Additional fallback: fuzzy matching for common typos
    if not potential_servers:
        # Split query into potential server tokens
        tokens = re.findall(r'[A-Za-z0-9]{5,}', query)
        for token in tokens:
            token_clean = re.sub(r'[^A-Za-z0-9]', '', token.upper())
            if len(token_clean) >= 5:
                for serial_key in processed_server_data.keys():
                    serial_clean = re.sub(r'[^A-Za-z0-9]', '', serial_key)
                    # Check if cleaned token matches cleaned serial
                    if token_clean == serial_clean:
                        potential_servers.add(serial_key)
                        break
    
    found_servers = list(potential_servers)
    
    # If no servers found, return helpful error message
    if not found_servers:
        available_servers = list(processed_server_data.keys())
        sample_servers = ', '.join(available_servers[:5])
        if len(available_servers) > 5:
            sample_servers += f" (and {len(available_servers) - 5} more)"
        
        return (f"No servers found in query. Please check the server serial number(s).\n\n"
                f"Available servers include: {sample_servers}\n"
                f"Example usage: 'co2 emission for server {available_servers[0]}' or "
                f"'{available_servers[0]} and {available_servers[1]} carbon footprint'")
    
    # Calculate CO2 emissions for all found servers
    intensity_factor = DEFAULT_CARBON_INTENSITY[carbon_intensity]
    results = []
    total_co2 = 0.0
    servers_with_data = 0
    
    for server_serial in sorted(found_servers):
        server_data = processed_server_data[server_serial]
        energy_kwh = server_data.get("estimated_energy_kwh", 0)
        
        if energy_kwh == 0:
            results.append({
                "serial": server_serial,
                "error": "No energy consumption data available"
            })
            continue
        
        # Calculate CO2 emissions
        co2_kg = energy_kwh * intensity_factor
        total_co2 += co2_kg
        servers_with_data += 1
        
        # Get efficiency rating
        avg_cpu = server_data["avg_cpu_util"]
        
        # Special case for 0% utilization
        if avg_cpu == 0:
            efficiency = "idle"
            efficiency_note = "Server is idle (0% CPU utilization)"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
            efficiency_note = f"Energy Efficiency Rating: {efficiency.capitalize()}"
        
        results.append({
            "serial": server_serial,
            "energy_kwh": round(energy_kwh, 2),
            "co2_kg": round(co2_kg, 2),
            "avg_cpu": avg_cpu,
            "efficiency": efficiency,
            "efficiency_note": efficiency_note
        })
    
    # Format output based on number of servers
    if len(found_servers) == 1:
        # Single server detailed output
        result = results[0]
        if "error" in result:
            return f"Server {result['serial']}: {result['error']}"
        
        output = f"Carbon footprint analysis for server {result['serial']}:\n\n"
        output += f"   - Energy Consumed: {result['energy_kwh']} kWh\n"
        output += f"   - CO2 Emissions: {result['co2_kg']} kg\n"
        output += f"   - Carbon Intensity: {carbon_intensity.replace('_', ' ').title()} ({intensity_factor} kg CO2/kWh)\n"
        output += f"   - Average CPU Utilization: {result['avg_cpu']}%\n"
        output += f"   - {result['efficiency_note']}\n"
        
    else:
        # Multiple servers output
        grid_type_display = carbon_intensity.replace('_', ' ').title()
        output = f"Carbon footprint analysis for {len(found_servers)} specified servers ({grid_type_display}):\n\n"
        
        if servers_with_data > 0:
            output += f"   - Total CO2 Emissions: {round(total_co2, 2)} kg\n"
            output += f"   - Average per Server: {round(total_co2/servers_with_data, 2)} kg\n\n"
        
        output += "Individual server details:\n\n"
        
        for i, result in enumerate(results, 1):
            if "error" in result:
                output += f"{i}. Server {result['serial']}: {result['error']}\n\n"
            else:
                output += f"{i}. Server {result['serial']}:\n"
                output += f"   - CO2 Emissions: {result['co2_kg']} kg\n"
                output += f"   - Energy Consumed: {result['energy_kwh']} kWh\n"
                output += f"   - CPU Utilization: {result['avg_cpu']}%\n"
                output += f"   - Efficiency: {result['efficiency'].capitalize()}\n\n"
    
    return output

def calculate_carbon_footprint_lowest(query: str) -> str:
    """
    Calculate carbon footprint for servers based on natural language query.
    Can handle requests for specific servers or all servers with different grid types.
    Shows LOWEST CO2 emitting servers by default.
    
    Args:
        query: Natural language query that may contain:
               - Server serial number (e.g. "server ABC123")
               - Carbon intensity type (e.g. "low carbon", "average", "high carbon")
               - Number of servers to show (e.g. "top 5", "show 15")
    
    Returns:
        Formatted string with carbon footprint results (lowest emitters first)
    """
    # Default values
    server_serial = None
    carbon_intensity = 'average_grid'
    num_servers_to_show = extract_server_count(query, default=10)
    
    # Parse query for server serial
    query_lower = query.lower()
    if "server" in query_lower:
        # Extract potential serial number after "server"
        parts = query_lower.split("server")
        if len(parts) > 1:
            potential_serial = parts[1].strip().upper()
            if potential_serial in processed_server_data:
                server_serial = potential_serial
    
    # Parse carbon intensity preference
    if "low carbon" in query_lower or "renewable" in query_lower:
        carbon_intensity = 'low_carbon_grid'
    elif "high carbon" in query_lower or "coal" in query_lower:
        carbon_intensity = 'high_carbon_grid'
    
    # Validate carbon intensity input
    if carbon_intensity not in DEFAULT_CARBON_INTENSITY:
        return f"Invalid carbon intensity. Choose from: {', '.join(DEFAULT_CARBON_INTENSITY.keys())}"
    
    intensity_factor = DEFAULT_CARBON_INTENSITY[carbon_intensity]
    total_co2 = 0.0
    results = []
    
    # Determine which servers to process
    servers_to_process = [server_serial] if server_serial else processed_server_data.keys()
    
    for serial in servers_to_process:
        if serial not in processed_server_data:
            continue
            
        server_data = processed_server_data[serial]
        energy_kwh = server_data.get("estimated_energy_kwh", 0)
        
        # Skip servers with insufficient data
        if energy_kwh == 0:
            continue
        
        # Calculate CO2 emissions
        co2_kg = energy_kwh * intensity_factor
        total_co2 += co2_kg
        
        # Get efficiency rating
        avg_cpu = server_data["avg_cpu_util"]
        
        # Special case for 0% utilization
        if avg_cpu == 0:
            efficiency = "idle"
        else:
            # Calculate power ratio only for active servers
            power_ratio = estimate_power(avg_cpu) / (50 + (300 - 50) * (avg_cpu/100))
            
            efficiency = "poor"  # Default to poor
            for rating, threshold in EFFICIENCY_THRESHOLDS["cpu_power_ratio"].items():
                if power_ratio <= threshold:
                    efficiency = rating
                    break
        
        results.append({
            "serial": serial,
            "energy_kwh": round(energy_kwh, 2),
            "co2_kg": round(co2_kg, 2),
            "avg_cpu": avg_cpu,
            "efficiency": efficiency,
            "carbon_intensity": carbon_intensity
        })
    
    if not results:
        return "No valid server data available for carbon footprint calculation."
    
    # Format the results
    if server_serial:
        # Single server detailed output
        result = next((r for r in results if r["serial"] == server_serial), None)
        if not result:
            return f"No data available for server {server_serial}."
        
        output = f"Carbon footprint analysis for server {server_serial}:\n\n"
        output += f"   - Energy Consumed: {result['energy_kwh']} kWh\n"
        output += f"   - CO2 Emissions: {result['co2_kg']} kg\n"
        output += f"   - Carbon Intensity: {carbon_intensity.replace('_', ' ').title()} ({intensity_factor} kg CO2/kWh)\n"
        output += f"   - Average CPU Utilization: {result['avg_cpu']}%\n"
        
        if result['efficiency'] == 'idle':
            output += "   - Energy Efficiency: Server is idle (0% CPU utilization)\n"
        else:
            output += f"   - Energy Efficiency Rating: {result['efficiency'].capitalize()}\n"
            
        return output
    
    else:
        # Multiple servers summary output - LOWEST emitters
        grid_type_display = carbon_intensity.replace('_', ' ').title()
        available_servers = len(results)
        
        output = f"Carbon footprint summary for all {available_servers} servers ({grid_type_display}):\n\n"
        output += f"   - Total CO2 Emissions: {round(total_co2, 2)} kg\n"
        output += f"   - Average per Server: {round(total_co2/available_servers, 2)} kg\n\n"
        
        if num_servers_to_show == float('inf'):
            # Show all servers
            top_count = available_servers
            output += f"All {available_servers} servers (lowest to highest emissions):\n\n"
        else:
            # Show requested number of servers
            top_count = min(int(num_servers_to_show), available_servers)
            output += f"Top {top_count} LOWEST emitting servers:\n\n"
        
        # Sort by LOWEST emissions (ascending order)
        sorted_results = sorted(results, key=lambda x: x["co2_kg"], reverse=False)
        
        for i, res in enumerate(sorted_results[:top_count]):
            output += f"{i+1}. Server {res['serial']}:\n"
            output += f"   - CO2 Emissions: {res['co2_kg']} kg\n"
            output += f"   - Energy Consumed: {res['energy_kwh']} kWh\n"
            output += f"   - CPU Utilization: {res['avg_cpu']}%\n"
            output += f"   - Efficiency Rating: {res['efficiency'].capitalize()}\n\n"
                
        # Add efficiency distribution
        eff_dist = {}
        for res in results:
            eff_dist[res["efficiency"]] = eff_dist.get(res["efficiency"], 0) + 1
        
        output += "Energy efficiency distribution:\n\n"
        for eff, count in sorted(eff_dist.items()):
            percentage = round((count / available_servers) * 100, 1)
            output += f"   - {eff.capitalize()}: {count} servers ({percentage}%)\n"
    
    return output

def get_server_stats(query: str) -> str:
    specific_server_serial = None
    match = re.search(r"server\s+([A-Z0-9-]+)", query, re.IGNORECASE)
    if match:
        potential_serial = match.group(1).upper()
        if potential_serial in processed_server_data:
            specific_server_serial = potential_serial
        else:
            # Invalid serial number - server doesn't exist
            return f"Server {potential_serial} does not exist."
    
    if specific_server_serial:
        data = processed_server_data[specific_server_serial]
        result = f"Server {specific_server_serial} Statistics:\n" + "=" * (len(specific_server_serial) + 20) + "\n\n"
        latest = data.get("latest_record")
        if latest:
            result += f"Latest Observation ({latest.get('time_str', 'N/A')}):\n"
            if latest.get('cpu_util') is not None: result += f"  CPU Utilization: {latest['cpu_util']}%\n"
            if latest.get('amb_temp') is not None: result += f"  Ambient Temperature: {latest['amb_temp']}°C\n"
            if latest.get('peak') is not None: result += f"  Peak Value: {latest['peak']}\n"
            if latest.get('estimated_power') is not None: result += f"  Estimated Power: {latest['estimated_power']}W\n"
        else:
            result += "No latest observation data available.\n"
        result += "\nSummary Metrics:\n"
        if data.get('avg_cpu_util') is not None: result += f"  Average CPU Utilization: {data['avg_cpu_util']}%\n"
        peak_cpu_rec = data.get("peak_cpu_record")
        if peak_cpu_rec and peak_cpu_rec.get('cpu_util') is not None:
            result += f"  Peak CPU: {peak_cpu_rec['cpu_util']}% at {peak_cpu_rec.get('time_str', 'N/A')}\n"
        lowest_cpu_rec = data.get("lowest_cpu_record")
        if lowest_cpu_rec and lowest_cpu_rec.get('cpu_util') is not None:
            result += f"  Lowest CPU: {lowest_cpu_rec['cpu_util']}% at {lowest_cpu_rec.get('time_str', 'N/A')}\n"
        if data.get('max_amb_temp') is not None:
            max_temp_rec = data.get("max_temp_record")
            result += f"  Max Ambient Temp: {data['max_amb_temp']}°C at {max_temp_rec.get('time_str', 'N/A') if max_temp_rec else 'N/A'}\n"
        if data.get('min_amb_temp') is not None:
            min_temp_rec = data.get("min_temp_record")
            result += f"  Min Ambient Temp: {data['min_amb_temp']}°C at {min_temp_rec.get('time_str', 'N/A') if min_temp_rec else 'N/A'}\n"
        if data.get('estimated_energy_kwh') is not None:
            result += f"  Total Estimated Energy: {data['estimated_energy_kwh']} kWh\n"
        if data.get('co2_emissions'):
            result += f"  Est. CO2 (avg grid): {data['co2_emissions'].get('average_grid', 'N/A')} kg\n"
        return result.strip()
    
    # Default response when no server specified
    if not processed_server_data:
        return "No server data available."
    
    result = "Server Fleet Statistics Summary (Top 10 by Peak CPU shown):\n" + "=" * 50 + "\n\n"
    servers_to_show = server_rankings.get("top_cpu", [])[:10]
    
    if not servers_to_show:
        result += "No ranked servers to display in summary.\n"
        count = 0
        for serial, data_item in processed_server_data.items():
            if count >= 5: 
                break
            latest = data_item.get("latest_record", {})
            result += f"Server {serial} (Last Seen: {latest.get('time_str', 'N/A')}):\n"
            result += f"  Latest CPU: {latest.get('cpu_util', 'N/A')}%, Peak CPU: {data_item.get('peak_cpu_util', 'N/A')}%\n"
            result += f"  Latest Amb Temp: {latest.get('amb_temp', 'N/A')}°C, Max Amb Temp: {data_item.get('max_amb_temp', 'N/A')}°C\n\n"
            count += 1
        if count == 0: 
            result += "No server data processed to display.\n"
    else:
        for serial, peak_cpu in servers_to_show:
            data = processed_server_data[serial]
            latest = data.get("latest_record", {})
            result += f"Server {serial} (Peak CPU: {peak_cpu}%):\n"
            result += f"  Last Observed: {latest.get('time_str', 'N/A')}\n"
            result += f"  Current CPU: {latest.get('cpu_util', 'N/A')}%\n"
            result += f"  Current Ambient: {latest.get('amb_temp', 'N/A')}°C\n\n"
    
    result += "For specific server details, query 'stats for server [SERIAL_NUMBER]'.\n"
    result += f"Total processed servers: {len(processed_server_data)}.\n"
    return result

def get_server_timestamps(query: str) -> str:
    if not processed_server_data: return "No server data available."
    server_patterns = [r'server\s+([A-Za-z0-9_-]+)', r'([A-Za-z0-9_-]{5,})']
    server_serial = None
    for pattern in server_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            potential_serial = match.group(1).upper()
            if potential_serial in processed_server_data:
                server_serial = potential_serial
                break
    if not server_serial:
        query_upper = query.upper()
        for serial_key in processed_server_data.keys():
            if serial_key in query_upper:
                server_serial = serial_key
                break
    if not server_serial:
        available_servers = list(processed_server_data.keys())
        return f"Could not identify server. Examples: {', '.join(available_servers[:3])}{'...' if len(available_servers) > 3 else ''}"
    server_info = processed_server_data[server_serial]
    if 'all_records' not in server_info or not server_info['all_records']:
        return f"No timestamp data for server {server_serial}"
    timestamps = [record['time_str'] for record in server_info['all_records'] if 'time_str' in record]
    if not timestamps: return f"No timestamps found for server {server_serial}"
    result = f"Timestamps for server {server_serial}:\nTotal records: {len(timestamps)}\n\n"
    display_count = min(20, len(timestamps))
    for i, timestamp in enumerate(timestamps[:display_count], 1):
        result += f"{i:2d}. {timestamp}\n"
    if len(timestamps) > display_count:
        result += f"\n... and {len(timestamps) - display_count} more timestamps"
    return result

def identify_high_cpu_servers(query: str) -> str:
    if not processed_server_data: return "No server data available."
    match = re.search(r'(\d+(\.\d+)?)', query)
    if not match: return "Please specify a CPU threshold (e.g., 'CPU above 80%')"
    try: threshold = float(match.group(1))
    except ValueError: return "Invalid number for CPU threshold."
    if not (0 <= threshold <= 100): return f"Invalid threshold: {threshold}%. Must be 0-100."
    high_cpu_servers_details = []
    for serial, server_info in processed_server_data.items():
        if 'all_records' not in server_info or not server_info['all_records']: continue
        high_cpu_count = 0
        max_cpu_this_server = 0.0
        for record in server_info['all_records']:
            cpu_util = record.get('cpu_util')
            if cpu_util is None: continue
            try: cpu_util_float = float(cpu_util)
            except (ValueError, TypeError): continue
            if cpu_util_float > threshold: high_cpu_count += 1
            if cpu_util_float > max_cpu_this_server: max_cpu_this_server = cpu_util_float
        if high_cpu_count > 0:
            total_records = len(server_info['all_records'])
            percentage_high_cpu_time = (high_cpu_count / total_records) * 100 if total_records > 0 else 0
            high_cpu_servers_details.append({'serial': serial, 'high_cpu_count': high_cpu_count,
                                             'total_records': total_records, 'percentage': percentage_high_cpu_time,
                                             'max_cpu_observed': max_cpu_this_server})
    if not high_cpu_servers_details: return f"No servers found with CPU above {threshold}%"
    high_cpu_servers_details.sort(key=lambda x: (x['percentage'], x['max_cpu_observed']), reverse=True)
    result = f"Servers with CPU records above {threshold}% (sorted by prevalence & max CPU):\n"
    result += f"Found {len(high_cpu_servers_details)} server(s) out of {len(processed_server_data)} total.\n\n"
    for i, stats in enumerate(high_cpu_servers_details[:20], 1):
        result += f"{i:2d}. Server: {stats['serial']}\n"
        result += f"    Instances >{threshold}%: {stats['high_cpu_count']}/{stats['total_records']} ({stats['percentage']:.1f}% of records)\n"
        result += f"    Highest CPU recorded: {stats['max_cpu_observed']:.1f}%\n\n"
    if len(high_cpu_servers_details) > 20: result += f"... and {len(high_cpu_servers_details) - 20} more."
    return result

def get_ambient_temp_stats(query: str) -> str:
    if not processed_server_data: return "Error: No server data for ambient temp stats."
    query_lower = query.lower()
    serial_match = re.search(r"server\s+([A-Z0-9-]+)", query_lower, re.IGNORECASE)
    specific_serial = None
    if serial_match:
        potential_serial = serial_match.group(1).upper()
        if potential_serial in processed_server_data: specific_serial = potential_serial
        else: return f"Server {potential_serial} not found for temp stats."
    if specific_serial:
        data = processed_server_data[specific_serial]
        result = f"Ambient Temp Stats for Server {specific_serial}:\n" + "=" * (len(specific_serial) + 30) + "\n\n"
        latest = data.get("latest_record", {})
        max_rec = data.get("max_temp_record", {})
        min_rec = data.get("min_temp_record", {})
        if latest.get('amb_temp') is not None: result += f"  Current: {latest['amb_temp']}°C (at {latest.get('time_str', 'N/A')})\n"
        else: result += "  Current: N/A\n"
        if data.get('max_amb_temp') is not None: result += f"  Maximum: {data['max_amb_temp']}°C (at {max_rec.get('time_str', 'N/A') if max_rec else 'N/A'})\n"
        else: result += "  Maximum: N/A\n"
        if data.get('min_amb_temp') is not None: result += f"  Minimum: {data['min_amb_temp']}°C (at {min_rec.get('time_str', 'N/A') if min_rec else 'N/A'})\n"
        else: result += "  Minimum: N/A\n"
        if data.get('max_amb_temp') is not None and data.get('min_amb_temp') is not None:
            result += f"  Range: {data['max_amb_temp'] - data['min_amb_temp']:.1f}°C\n"
        else: result += "  Range: N/A\n"
        all_server_temps = [rec['amb_temp'] for rec in data.get("all_records", []) if rec.get('amb_temp') is not None]
        if all_server_temps:
            avg_server_temp = sum(all_server_temps) / len(all_server_temps)
            result += f"  Average: {avg_server_temp:.1f}°C (over all its records)\n"
        else: result += "  Average: N/A\n"
        return result.strip()
    result = "Overall Ambient Temperature Statistics (Fleet):\n" + "=" * 45 + "\n\n"
    if server_rankings["top_amb_temp"]:
        result += "🏆 Top 5 Highest Max Ambient Temperatures:\n"
        for i, (serial, temp) in enumerate(server_rankings["top_amb_temp"][:5], 1):
            result += f"  {i}. Server {serial}: {temp}°C (at {processed_server_data[serial]['max_temp_record']['time_str']})\n"
        result += "\n"
    if server_rankings["bottom_amb_temp"]:
        result += "❄️ Top 5 Lowest Min Ambient Temperatures:\n"
        for i, (serial, temp) in enumerate(server_rankings["bottom_amb_temp"][:5], 1):
            result += f"  {i}. Server {serial}: {temp}°C (at {processed_server_data[serial]['min_temp_record']['time_str']})\n"
        result += "\n"
    all_latest_temps = [s_data['latest_record']['amb_temp'] for s_data in processed_server_data.values()
                        if s_data.get('latest_record') and s_data['latest_record'].get('amb_temp') is not None]
    if all_latest_temps:
        avg_fleet_temp = sum(all_latest_temps) / len(all_latest_temps)
        max_fleet_latest_temp = max(all_latest_temps)
        min_fleet_latest_temp = min(all_latest_temps)
        result += f"🌡️ Current Fleet Ambient Temperatures (latest records):\n"
        result += f"   - Average: {avg_fleet_temp:.1f}°C\n"
        result += f"   - Highest Current: {max_fleet_latest_temp:.1f}°C\n"
        result += f"   - Lowest Current: {min_fleet_latest_temp:.1f}°C\n"
    return result.strip()


def get_filtered_server_records(query_params_str: str) -> str:
    try:
        params = json.loads(query_params_str)
        server_serial, metric_key, operator, value = params.get("server_serial"), params.get("metric"), params.get("operator"), params.get("value")
        if not all([server_serial, metric_key, operator, value is not None]):
            return "Error: Missing one or more required JSON fields: 'server_serial', 'metric', 'operator', 'value'."
        server_serial = server_serial.upper()
        if server_serial not in processed_server_data: return f"Error: Server {server_serial} not found."
        server_info = processed_server_data[server_serial]
        if 'all_records' not in server_info or not server_info['all_records']:
            return f"No records for server {server_serial}."
        if metric_key not in ['cpu_util', 'amb_temp', 'peak']:
            return f"Error: Unsupported metric '{metric_key}'. Use: cpu_util, amb_temp, peak."
        if operator not in ['greater_than', 'less_than', 'equals']:
            return f"Error: Unsupported operator '{operator}'. Use: greater_than, less_than, equals."
        try: filter_value = float(value)
        except ValueError: return f"Error: Filter value '{value}' must be numeric."
        matching_records_info = []
        for record in server_info['all_records']:
            record_value = record.get(metric_key)
            if record_value is None: continue
            try: record_value_float = float(record_value)
            except (ValueError, TypeError): continue
            match = False
            if operator == 'greater_than' and record_value_float > filter_value: match = True
            elif operator == 'less_than' and record_value_float < filter_value: match = True
            elif operator == 'equals' and record_value_float == filter_value: match = True
            if match:
                matching_records_info.append(f"- Timestamp: {record['time_str']}, {metric_key.replace('_', ' ').title()}: {record_value}")
        if not matching_records_info:
            return f"No records for {server_serial} where {metric_key} {operator.replace('_',' ')} {filter_value}."
        result = f"Filtered records for {server_serial} ({metric_key} {operator.replace('_',' ')} {filter_value}):\n"
        result += f"Found {len(matching_records_info)} record(s).\n\n"
        display_count = min(20, len(matching_records_info))
        for i, rec_info in enumerate(matching_records_info[:display_count], 1):
            result += f"{i:2d}. {rec_info}\n"
        if len(matching_records_info) > display_count:
            result += f"\n... and {len(matching_records_info) - display_count} more."
        return result
    except json.JSONDecodeError:
        return "Error: Invalid JSON for Action Input. E.g., '{\"server_serial\": \"XYZ\", \"metric\": \"cpu_util\", \"operator\": \"greater_than\", \"value\": 10}'. Double quotes essential."
    except Exception as e:
        logger.error(f"Error in get_filtered_server_records: {e}", exc_info=True)
        return f"Unexpected error filtering records: {str(e)}"

def extract_server_name(query: str, all_servers: set) -> Optional[str]:
    """Extracts a likely server name from the query based on known server IDs."""
    upper_query = query.upper()

    # Match all uppercase-alphanumeric-underscore-hyphen patterns
    candidates = re.findall(r'\b[A-Z0-9_\\-]{3,}\b', upper_query)
    for candidate in candidates:
        if candidate in all_servers:
            return candidate
    return None

def detect_anomalies(query: str) -> str:
    METRIC_KEYWORDS = {
    "cpu_util": ["cpu utilization", "cpu util", "cpu usage", "strange behavior in cpu", "cpu load"],
    "amb_temp": ["ambient temperature", "amb temp", "temperature", "temperature spikes"],
    "cpu_watts": ["cpu power", "cpu watts", "power consumption", "power usage"],
    "dimm_watts": ["memory power", "dimm watts", "dimm memory power"],
}
    
    def extract_metrics(query: str) -> List[str]:
        query = query.lower()
        matched = []
        for metric, aliases in METRIC_KEYWORDS.items():
            for phrase in aliases:
                if phrase in query:
                    matched.append(metric)
                    break
        return matched if matched else ["cpu_util", "amb_temp", "cpu_watts", "dimm_watts"]


    # Parse query to determine scope
    analyze_all = True
    specific_server = None
    specific_metric = None

    query_lower = query.lower()
    
    potential_serial = extract_server_name(query, set(processed_server_data.keys()))
    if potential_serial:
        analyze_all = False
        specific_server = potential_serial
    elif re.search(r"\bserver\b", query.lower()):
        return f"⚠️ Server mentioned but not found in dataset. Please check the name."
    # otherwise, proceed with analyze_all = True



    
    # Determine which metric to analyze
    metrics_to_check = extract_metrics(query)
    
    # Check for multi-word metrics first
    if "cpu watts" in query_lower or "cpu_watts" in query_lower:
        metrics_to_check = ["cpu_watts"]
    elif "dimm watts" in query_lower or "dimm_watts" in query_lower:
        metrics_to_check = ["dimm_watts"]
    elif "cpu util" in query_lower or "cpu_util" in query_lower:
        metrics_to_check = ["cpu_util"]
    elif "amb temp" in query_lower or "amb_temp" in query_lower:
        metrics_to_check = ["amb_temp"]
    # Fallback to single-word matches
    elif "watts" in query_lower:
        metrics_to_check = ["cpu_watts", "dimm_watts"]
    elif "temp" in query_lower:
        metrics_to_check = ["amb_temp"]
    elif "cpu" in query_lower or "util" in query_lower:
        metrics_to_check = ["cpu_util"]
    
    if not metrics_to_check:
        metrics_to_check = ["cpu_util", "amb_temp", "cpu_watts", "dimm_watts"]

    
    # Enhanced statistical anomaly detection
    def find_anomalies(values, timestamps, metric_name, server_serial):
        if not values or len(values) < 3:
            return [], None
        
        values = [float(v) for v in values]
        median = sorted(values)[len(values)//2]
        deviations = [abs(x - median) for x in values]
        mad = sorted(deviations)[len(deviations)//2]
        
        if mad == 0:
            return [], median
        
        base_threshold = 3.5  # Lower base threshold
        threshold = base_threshold
        
        # More sensitive scaling
        if len(values) < 10:
            threshold = 3.0  # Very sensitive for tiny datasets
        elif len(values) > 100:
            threshold = base_threshold + (len(values) / 500)  # Faster scaling
        
        # Debug output (optional)
        # print(f"Debug - {metric_name}: Median={median}, MAD={mad}, Threshold={threshold}")
        # print(f"Values: {values}, Z-scores: {modified_z_scores}")
        
        # Find anomalies with deduplication
        anomalies = []
        seen = set()
        modified_z_scores = [0.6745 * (x - median) / mad for x in values]
        
        for i, score in enumerate(modified_z_scores):
            if abs(score) > threshold:
                anomaly_key = f"{values[i]}-{timestamps[i]}"
                if anomaly_key not in seen:
                    seen.add(anomaly_key)
                    anomalies.append({
                        "value": values[i],
                        "z_score": round(score, 2),
                        "metric": metric_name,
                        "server": server_serial,
                        "timestamp": timestamps[i]
                    })
        
        return anomalies, median
    
    # Process requested servers
    servers_to_check = list(processed_server_data.keys()) if analyze_all else [specific_server]
    all_anomalies = []
    median_baselines = {}
    
    for serial in servers_to_check:
        server_data = processed_server_data.get(serial)
        if not server_data:
            continue
            
        records = server_data.get("all_records", [])
        if not records:
            continue
            
        # Check each requested metric
        for metric in metrics_to_check:
            values = []
            timestamps = []
            
            for record in records:
                val = record.get(metric)
                if val is not None:
                    values.append(val)
                    timestamps.append(record["time_str"])
            
            if values:
                metric_anomalies, median = find_anomalies(values, timestamps, metric, serial)
                if metric not in median_baselines:
                    median_baselines[metric] = median
                all_anomalies.extend(metric_anomalies)
    
    # Format results with enhanced information
    if not all_anomalies:
        if analyze_all:
            return "No significant anomalies detected across all servers and metrics."
        return f"No significant anomalies detected for server {specific_server}."
    
    # Group by severity
    critical = [a for a in all_anomalies if abs(a["z_score"]) > 5]
    major = [a for a in all_anomalies if 3.5 < abs(a["z_score"]) <= 5]
    
    # Temporal analysis
    anomaly_hours = [a['timestamp'].split(', ')[1][:2] for a in all_anomalies]
    hour_dist = Counter(anomaly_hours).most_common(3)
    
    # Build enhanced output
    output = []
    if analyze_all:
        output.append(f"📊 Enhanced Anomaly Report for {len(servers_to_check)} servers")
    else:
        output.append(f"📊 Enhanced Anomaly Report for server {specific_server}")
    
    # Baseline context
    output.append("\n🔍 Normal Ranges (median values):")
    for metric, median in median_baselines.items():
        output.append(f"- {metric}: {median}")
    
    # Critical anomalies
    if critical:
        output.append("\n🚨 CRITICAL ANOMALIES (z-score > 5):")
        for a in critical[:5]:  # Top 5 only
            output.append(
                f"- {a['server']} | {a['metric']} = {a['value']} "
                f"(z-score: {a['z_score']}) at {a['timestamp']}"
            )
    
    # Major anomalies
    if major:
        output.append("\n⚠️ MAJOR ANOMALIES (3.5 < z-score ≤ 5):")
        for a in major[:5]:  # Top 5 only
            output.append(
                f"- {a['server']} | {a['metric']} = {a['value']} "
                f"(z-score: {a['z_score']}) at {a['timestamp']}"
            )
    
    # Temporal patterns
    if hour_dist:
        output.append("\n⏰ Frequent Anomaly Times:")
        for hour, count in hour_dist:
            output.append(f"- {hour}:00 - {count} anomalies")
    
    # Root cause suggestions
    output.append("\n🔧 Potential Investigation Paths:")
    if 'cpu_watts' in median_baselines:
        output.append("- CPU Power Spikes: Check workload scheduler and cooling")
    if 'amb_temp' in median_baselines:
        output.append("- Temp Fluctuations: Verify HVAC and rack airflow")
    if 'dimm_watts' in median_baselines:
        output.append("- Memory Power: Run DIMM diagnostics")
    
    # Summary
    total_anomalies = len(critical) + len(major)
    output.append(f"\n📈 Found {total_anomalies} significant anomalies (showing top 5 each)")
    
    return "\n".join(output)

def identify_low_cpu_servers(query: str) -> str:
    """
    Identifies servers with CPU usage below a specified threshold.
    
    Args:
        query: String containing CPU threshold (e.g., 'CPU below 20%')
    
    Returns:
        Formatted string with servers having CPU below threshold
    """
    if not processed_server_data: 
        return "No server data available."
    
    match = re.search(r'(\d+(\.\d+)?)', query)
    if not match: 
        return "Please specify a CPU threshold (e.g., 'CPU below 20%')"
    
    try: 
        threshold = float(match.group(1))
    except ValueError: 
        return "Invalid number for CPU threshold."
    
    if not (0 <= threshold <= 100): 
        return f"Invalid threshold: {threshold}%. Must be 0-100."
    
    low_cpu_servers_details = []
    
    for serial, server_info in processed_server_data.items():
        if 'all_records' not in server_info or not server_info['all_records']: 
            continue
            
        low_cpu_count = 0
        min_cpu_this_server = 100.0  # Start with max possible CPU
        total_valid_records = 0
        
        for record in server_info['all_records']:
            cpu_util = record.get('cpu_util')
            if cpu_util is None: 
                continue
                
            try: 
                cpu_util_float = float(cpu_util)
            except (ValueError, TypeError): 
                continue
                
            total_valid_records += 1
            
            if cpu_util_float < threshold: 
                low_cpu_count += 1
            if cpu_util_float < min_cpu_this_server: 
                min_cpu_this_server = cpu_util_float
        
        if low_cpu_count > 0 and total_valid_records > 0:
            percentage_low_cpu_time = (low_cpu_count / total_valid_records) * 100
            low_cpu_servers_details.append({
                'serial': serial, 
                'low_cpu_count': low_cpu_count,
                'total_records': total_valid_records, 
                'percentage': percentage_low_cpu_time,
                'min_cpu_observed': min_cpu_this_server
            })
    
    if not low_cpu_servers_details: 
        return f"No servers found with CPU below {threshold}%"
    
    # Sort by percentage of time below threshold (descending), then by minimum CPU observed (ascending)
    low_cpu_servers_details.sort(key=lambda x: (x['percentage'], -x['min_cpu_observed']), reverse=True)
    
    result = f"Servers with CPU records below {threshold}% (sorted by prevalence & min CPU):\n"
    result += f"Found {len(low_cpu_servers_details)} server(s) out of {len(processed_server_data)} total.\n\n"
    
    for i, stats in enumerate(low_cpu_servers_details[:20], 1):
        result += f"{i:2d}. Server: {stats['serial']}\n"
        result += f"    Instances <{threshold}%: {stats['low_cpu_count']}/{stats['total_records']} ({stats['percentage']:.1f}% of records)\n"
        result += f"    Lowest CPU recorded: {stats['min_cpu_observed']:.1f}%\n\n"
    
    if len(low_cpu_servers_details) > 20: 
        result += f"... and {len(low_cpu_servers_details) - 20} more."
    
    return result
