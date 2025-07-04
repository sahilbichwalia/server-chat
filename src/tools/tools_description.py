from langchain.tools import Tool
from src.tools.tool_function import (
    list_servers,
    get_top_servers_by_cpu_util,
    get_specific_server_cpu_utilization,
    get_lowest_servers_by_cpu_util,
    get_top_servers_by_ambient_temp,
    get_specific_server_ambient_temp,
    get_lowest_servers_by_ambient_temp,
    get_top_servers_by_peak,
    get_specific_server_peak_data,
    get_lowest_servers_by_peak,
    get_server_stats,
    calculate_carbon_footprint,
    co2_emission_server,
    calculate_carbon_footprint_lowest,
    identify_high_cpu_servers,
    get_server_timestamps,
    get_filtered_server_records,
    detect_anomalies,
    generate_csv_report 
)
from src.tools.rag import query_documents, list_available_documents

tools = [
    Tool(
        name="ListServers",
        func=list_servers,
        description=(
            "**Purpose:** Use this tool to view, list, or summarize all available servers being monitored.\n\n"
            "**Triggers:** 'List all servers', 'Show servers being monitored', 'What servers are active?'.\n\n"
            "**Input:** An empty string.\n"
            "**Example:** `Action: ListServers[]`\n\n"
            "**Returns:** A human-readable summary of all monitored servers, including:\n"
            "- Serial number\n"
            "- Last seen timestamp\n"
            "- Total records\n"
            "- Average CPU usage\n"
            "- Average ambient temperature\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the summarized server list as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetTopServersByCPUUtil",
        func=get_top_servers_by_cpu_util,
        description=(
            "**Purpose:** Retrieve servers with the highest CPU utilization.\n\n"
            "**Input:** A natural language query specifying how many top servers to show (e.g., '5', 'ten', 'all'). "
            "Defaults to 10 servers if no number is specified. 'All' returns all servers with relevant data.\n\n"
            "**Example Queries:**\n"
            "- 'Top 5 CPU servers'\n"
            "- 'Which 3 servers have highest CPU utilization?'\n"
            "- 'Show all high CPU servers'\n\n"
            "**Returns:** For each server in the requested top list:\n"
            "- Serial number\n"
            "- Peak CPU (%)\n"
            "- Timestamp of peak CPU\n"
            "- Power (Watts) at peak\n"
            "- Temperature (°C) at peak\n"
            "- Fan speed (RPM) at peak\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the detailed server list as your final answer without further tool calls for this request. If the requested count exceeds available data, the tool will inform you. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetServerCPUUtilization",
        func=get_specific_server_cpu_utilization,
        description=(
            "**Purpose:** Get detailed CPU utilization information for SPECIFIC server(s).\n\n"
            "**Input:** A natural language query identifying one or more server serial numbers. "
            "The tool uses robust server identification to handle various formats including full server names (e.g., 'server SGH227WTNK'), "
            "direct serial numbers (e.g., 'SGH227WTNK'), multiple servers (e.g., 'SGH227WTNK and ABC123'), "
            "case-insensitive matching, and common typos.\n\n"
            "**Example Queries:**\n"
            "- 'What is the CPU utilization of server SGH227WTNK?'\n"
            "- 'Show CPU stats for SGH227WTNK and ABC123'\n"
            "- 'CPU usage for servers SGH227WTNK, DEF456'\n"
            "- 'SGH227WTNK CPU utilization details'\n"
            "- 'Compare CPU usage SGH227WTNK ABC123'\n\n"
            "**Returns:** Detailed analysis:\n"
            "For a single server:\n"
            "- Average CPU Utilization (%)\n"
            "- Peak CPU Utilization (%) with timestamp\n"
            "- Power consumption (W) at peak\n"
            "- Temperature (°C) and fan speed (RPM) at peak\n"
            "- CPU efficiency rating\n"
            "- Fleet ranking position\n"
            "For multiple servers:\n"
            "- Summary statistics across servers\n"
            "- Individual server breakdowns\n"
            "- Comparative analysis with rankings\n\n"
            "**Error Handling:** If server(s) are not found, returns a helpful error message with examples of available servers.\n\n"
            "**Important Notes for Agent:** Upon receiving this data, you have sufficient information to answer the user's query directly. Present the detailed analysis as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetLowestServersByCPUUtil",
        func=get_lowest_servers_by_cpu_util,
        description=(
            "**Purpose:** Find servers with the lowest CPU utilization.\n\n"
            "**Input:** A natural language query specifying how many lowest servers to show (e.g., '5', 'three', 'all'). "
            "Defaults to 10 servers if no number is specified. 'All' returns all servers with relevant data.\n\n"
            "**Example Queries:**\n"
            "- 'Top 5 servers with lowest CPU utilization'\n"
            "- 'Which three servers have the lowest CPU usage?'\n"
            "- 'Show all low CPU servers'\n"
            "- 'List one server with lowest CPU utilization'\n\n"
            "**Returns:** For each server in the requested list:\n"
            "- Serial number\n"
            "- Lowest CPU (%)\n"
            "- Timestamp of lowest CPU\n"
            "- Power (Watts) at lowest CPU\n"
            "- Temperature (°C) at lowest CPU\n"
            "- Fan speed (RPM) at lowest CPU\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the detailed server list as your final answer without further tool calls for this request. If the requested count exceeds available data, the tool will inform you. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetTopServersByAmbientTemp",
        func=get_top_servers_by_ambient_temp,
        description=(
            "**Purpose:** Find servers ranked by their highest ambient temperature records.\n\n"
            "**Input:** A natural language query specifying how many top servers to show (e.g., '5', 'two', 'all'). "
            "Defaults to showing all servers if no number is specified. 'All' returns all servers with data.\n\n"
            "**Example Queries:**\n"
            "- 'Top 5 servers with highest ambient temperature'\n"
            "- 'Which three servers have the highest ambient temperature?'\n"
            "- 'Show all servers by ambient temperature'\n"
            "- 'List one server with highest ambient temperature'\n\n"
            "**Returns:** For each server in the requested list:\n"
            "- Server serial number\n"
            "- Highest ambient temperature (°C)\n"
            "- Timestamp of the highest temperature record\n"
            "- CPU utilization (%) at that time\n"
            "- CPU power consumption (Watts) at that time\n"
            "- DIMM power consumption (Watts) at that time\n\n"
            "**Error Handling:** Handles incomplete data gracefully and informs if the requested count exceeds available data.\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the detailed server list as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetSpecificServerAmbientTemp",
        func=get_specific_server_ambient_temp,
        description=(
            "**Purpose:** Get ambient temperature data for specific server(s) identified by their serial numbers.\n\n"
            "**Input:** A natural language query identifying one or more server serial numbers. "
            "It can handle single or multiple servers with robust server identification.\n\n"
            "**Example Queries:**\n"
            "- 'What is the ambient temperature for server SGH227WTNK?'\n"
            "- 'Show ambient temperature data for SGH227WTNK and ABC123'\n"
            "- 'Get temperature info for server XYZ456'\n"
            "- 'SGH227WTNK, DEF456 ambient temperature'\n\n"
            "**Returns:** For each server:\n"
            "- Server serial number\n"
            "- Highest ambient temperature recorded (°C)\n"
            "- Average ambient temperature (°C)\n"
            "- Timestamp of the highest temperature record\n"
            "- CPU utilization (%) at peak temperature\n"
            "- CPU power consumption (Watts) at peak\n"
            "- DIMM power consumption (Watts) at peak\n\n"
            "**Error Handling:** Handles multiple servers with summary statistics and gracefully handles missing data. "
            "Uses the same robust server identification patterns as the CO2 emission function.\n\n"
            "**Important Notes for Agent:** Upon receiving this data, you have sufficient information to answer the user's query directly. Present the detailed analysis as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetLowestServersByAmbientTemp",
        func=get_lowest_servers_by_ambient_temp,
        description=(
            "**Purpose:** Find servers ranked by their lowest ambient temperature records.\n\n"
            "**Input:** A natural language query specifying how many lowest servers to show (e.g., 'one', 'two', 'all'). "
            "Defaults to showing all servers if no number is specified. 'All' returns all servers with data.\n\n"
            "**Example Queries:**\n"
            "- 'Bottom 5 servers with lowest ambient temperature'\n"
            "- 'Which three servers have the lowest ambient temperature?'\n"
            "- 'Show all servers by lowest ambient temperature'\n"
            "- 'List one server with lowest ambient temperature'\n\n"
            "**Returns:** For each server in the requested list:\n"
            "- Server serial number\n"
            "- Lowest ambient temperature (°C)\n"
            "- Timestamp of the lowest temperature record\n"
            "- CPU utilization (%) at that time\n"
            "- CPU power consumption (Watts) at that time\n"
            "- DIMM power consumption (Watts) at that time\n\n"
            "**Error Handling:** Handles incomplete data gracefully and informs if the requested count exceeds available data.\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the detailed server list as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetTopServersByPeak",
        func=get_top_servers_by_peak,
        description=(
            "**Purpose:** Retrieve servers with the highest peak values across all metrics (CPU, temperature, power, etc.).\n\n"
            "**Input:** A natural language query specifying the number of top servers to show (e.g., 'one server', 'top 3 servers', 'all peak servers'). "
            "Defaults to 10 if unspecified. 'All' returns the full list.\n\n"
            "**Example Queries:**\n"
            "- 'Which 5 servers have the highest peak values?'\n"
            "- 'Top 3 servers by peak usage'\n"
            "- 'Show all servers with highest peak value'\n\n"
            "**Returns:** For each server in the requested top list:\n"
            "- Serial number\n"
            "- Highest peak value observed\n"
            "- Timestamp of peak\n"
            "- CPU Utilization (%) at peak\n"
            "- Ambient Temperature (°C) at peak\n"
            "- CPU Power (Watts) at peak\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the detailed server list as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetSpecificServerPeakData",
        func=get_specific_server_peak_data,
        description=(
            "**Purpose:** Get peak data for specific server(s) identified by their serial numbers.\n\n"
            "**Input:** A natural language query identifying one or more server serial numbers. "
            "It can handle single or multiple servers in one query with robust server identification.\n\n"
            "**Example Queries:**\n"
            "- 'What is the peak data for server SGH227WTNK?'\n"
            "- 'Show peak values for SGH227WTNK and ABC123'\n"
            "- 'Get peak performance for server XYZ456'\n"
            "- 'Lowest peak servers abc123'\n"
            "- 'SGH227WTNK, DEF456 peak data'\n\n"
            "**Returns:** For each server:\n"
            "- Server serial number\n"
            "- Highest peak value recorded\n"
            "- Average peak value\n"
            "- Timestamp of the highest peak record\n"
            "- CPU utilization (%) at peak\n"
            "- Ambient temperature (°C) at peak\f"
            "- CPU power consumption (Watts) at peak\n\n"
            "**Error Handling:** Handles multiple servers with summary statistics and gracefully handles missing data. "
            "Uses the same robust server identification patterns as other specific server functions.\n\n"
            "**Important Notes for Agent:** Upon receiving this data, you have sufficient information to answer the user's query directly. Present the detailed analysis as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetLowestServersByPeak",
        func=get_lowest_servers_by_peak,
        description=(
            "**Purpose:** Retrieve servers with the lowest peak values across all metrics.\n\n"
            "**Input:** A natural language query specifying the number of servers to show (e.g., 'one server', 'bottom 3 servers', 'all low peak servers'). "
            "Defaults to 10 if not specified. 'All' returns the full list.\n\n"
            "**Example Queries:**\n"
            "- 'Show 3 servers with the lowest peak usage'\n"
            "- 'Bottom 5 peak value servers'\n"
            "- 'All servers with the lowest peak values'\n\n"
            "**Returns:** For each server in the requested list:\n"
            "- Serial number\n"
            "- Lowest peak value observed\n"
            "- Timestamp of that value\n"
            "- CPU Utilization (%) at lowest peak\n"
            "- Ambient Temperature (°C) at lowest peak\n"
            "- CPU Power (Watts) at lowest peak\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the detailed server list as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetServerStats",
        func=get_server_stats,
        description=(
            "**Purpose:** Retrieve statistics for a specific server or a summary of the entire fleet.\n\n"
            "**Triggers:**\n"
            "- 'Stats for server ABC123'\n"
            "- 'Give me server ST-998 details'\n"
            "- 'Show latest observation for server Y56-22'\n"
            "- 'Show latest observation for all the server'\n"
            "- 'Show latest observation for each server'\n"
            "- 'What’s the summary of all servers?'\n\n"
            "**Returns:**\n"
            "If a specific server serial number is mentioned:\n"
            "- Latest record timestamp\n"
            "- Current CPU Utilization, Peak Value, Power (W), Ambient Temperature\n"
            "- Peak and lowest CPU with timestamps\n"
            "- Max/min ambient temperatures with timestamps\n"
            "- Estimated total energy used and CO₂ emissions\n"
            "If no server is specified (fleet-wide summary):\n"
            "- Top 5 servers by peak CPU usage\n"
            "- Latest CPU and temperature readings across the fleet\n"
            "- General fleet statistics and record availability\n\n"
            "**Example Inputs:**\n"
            "- 'Show stats for server TDX-901'\n"
            "- 'Fleet summary'\n"
            "- 'Give observation for server XP100'\n\n"
            "**Important Notes for Agent:** Upon receiving this data, you have sufficient information to answer the user's query directly. Present the detailed statistics or summary as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="CalculateCarbonFootprint",
        func=calculate_carbon_footprint,
        description=(
            "**Purpose:** Calculate the carbon footprint for multiple servers or perform a fleet-wide analysis.\n\n"
            "**Input:** A natural language query that may include keywords for grid intensity ('average', 'low-carbon', 'high-carbon', 'renewable', 'coal') "
            "and server counts (e.g., 'top 5', 'ten servers', 'all'). This tool is specifically for MULTIPLE server analysis.\n\n"
            "**Important:** For individual server carbon footprint calculation, use `CO2EmissionServer` instead.\n\n"
            "**Example Queries:**\n"
            "- 'Show CO2 emissions for all servers using renewable energy.'\n"
            "- 'Calculate carbon footprint for top 3 servers using high carbon grid.'\n"
            "- 'List servers with highest emissions based on coal grid.'\n"
            "- 'Top 10 servers by carbon emissions'\n"
            "- 'Fleet-wide carbon footprint analysis'\n\n"
            "**Returns:** A fleet summary including:\n"
            "- Total CO₂ Emissions across all selected servers\n"
            "- Average emissions per server\n"
            "- Top N highest emitting servers with details\n"
            "- Energy efficiency distribution\n\n"
            "**Important Notes for Agent:** Upon receiving this report, you have sufficient information to answer the user's query directly. Present the detailed carbon footprint analysis as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="CO2EmissionServer",
        func=co2_emission_server,
        description=(
            "**Purpose:** Calculate the carbon footprint of SPECIFIC server(s) – single or multiple.\n\n"
            "**Input:** A natural language query identifying one or more server serial numbers (e.g., 'server SGH227WTNK', 'SGH227WTNK and ABC123'). "
            "Also include keywords for carbon intensity if desired (e.g., 'renewable', 'coal', 'low carbon'). "
            "The tool uses robust server identification (case-insensitive, handles typos) to find servers.\n\n"
            "**Example Queries:**\n"
            "- 'What is the carbon footprint of server SGH227WTNK?'\n"
            "- 'CO2 emissions for SGH227WTNK and ABC123 using renewable energy'\n"
            "- 'Show carbon footprint of servers SGH227WTNK, DEF456 with high carbon grid'\n"
            "- 'SGH227WTNK ABC123 carbon emissions'\n"
            "- 'Compare CO2 for SGH227WTNK and DEF456'\n\n"
            "**Returns:** Detailed analysis:\n"
            "For a single server:\n"
            "- Energy Consumed (kWh)\n"
            "- CO₂ Emissions (kg)\n"
            "- Carbon Intensity used in calculation\n"
            "- Average CPU Utilization (%)\n"
            "- Energy Efficiency Rating\n"
            "For multiple servers:\n"
            "- Total and average CO₂ emissions\n"
            "- Individual server breakdowns\n"
            "- Comparative analysis\n\n"
            "**Error Handling:** If server(s) are not found, returns a helpful error message with examples of available servers.\n\n"
            "**Important Notes for Agent:** Upon receiving this data, you have sufficient information to answer the user's query directly. Present the detailed carbon footprint analysis as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="CalculateCarbonFootprintLowest",
        func=calculate_carbon_footprint_lowest,
        description=(
            "**Purpose:** Calculate the carbon footprint of servers, specifically identifying those with the LOWEST CO₂ emissions (most energy-efficient).\n\n"
            "**Input:** A natural language query that may include keywords for grid intensity ('average', 'low-carbon', 'high-carbon', 'renewable', 'coal'), "
            "and server counts (e.g., 'top 5', 'ten servers', 'all'). If a specific server is mentioned, it returns data for that server only.\n\n"
            "**Example Queries:**\n"
            "- 'Show me the 10 most energy-efficient servers'\n"
            "- 'Which servers have the lowest carbon footprint?'\n"
            "- 'Top 5 cleanest servers using renewable energy'\n"
            "- 'List servers with minimum CO2 emissions'\n"
            "- 'Show least polluting servers'\n"
            "- 'Most efficient servers by carbon footprint'\n\n"
            "**Returns:** For each server in the requested list:\n"
            "- Serial number\n"
            "- Energy Consumed (kWh)\n"
            "- CO₂ Emissions (kg)\n"
            "- Average CPU Utilization (%)\n"
            "- Efficiency Rating (based on CPU-to-power ratio)\n"
            "Also returns a fleet summary when multiple servers are included, with efficiency distribution.\n\n"
            "**Important Notes for Agent:** Upon receiving this report, you have sufficient information to answer the user's query directly. Present the detailed carbon footprint analysis as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="IdentifyHighCPUServers",
        func=identify_high_cpu_servers,
        description=(
            "**Purpose:** Identify servers that have CPU utilization above a specified threshold.\n\n"
            "**Input:** A natural language query including a numeric threshold (e.g., 'above 80%' or 'more than 70%').\n\n"
            "**Action:** Analyzes all server records and returns those with at least one instance of CPU utilization above the given threshold.\n\n"
            "**Example Inputs:**\n"
            "- 'Show servers with CPU above 90%'\n"
            "- 'List all servers crossing 75% CPU utilization'\n"
            "- 'Which servers hit CPU over 85%?'\n\n"
            "**Returns:** For each matching server:\n"
            "- Serial number\n"
            "- Count and percentage of records where CPU > threshold\n"
            "- Maximum CPU utilization observed\n\n"
            "**Notes:**\n"
            "- Maximum 20 servers are shown in detail; remaining are summarized.\n"
            "- Results are sorted by percentage of high CPU records and peak CPU observed.\n\n"
            "**Important Notes for Agent:** Upon receiving this report, you have sufficient information to answer the user's query directly. Present the detailed server list as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="GetServerTimestamps",
        func=get_server_timestamps,
        description=(
            "**Purpose:** Retrieve the list of timestamped monitoring records for a specific server.\n\n"
            "**Input:** A natural language query containing the server serial number (e.g., 'server A123B', 'timestamps for server XYZ001').\n\n"
            "**Example Inputs:**\n"
            "- 'Show timestamps for server A12B9'\n"
            "- 'Get monitoring history of server 7GHT9'\n"
            "- 'When was server TEST-SRVR last active?'\n\n"
            "**Returns:**\n"
            "- The server’s total number of records\n"
            "- Up to 20 of the earliest timestamps (in order of appearance in data)\n"
            "- A note about any additional timestamps\n\n"
            "**Notes:**\n"
            "- The tool attempts to match server serials even with partial or imprecise queries.\n"
            "- If no match is found, it will suggest a few available server serials.\n\n"
            "**Important Notes for Agent:** Upon receiving this data, you have sufficient information to answer the user's query directly. Present the detailed timestamp information as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="FilterServerRecords",
        func=get_filtered_server_records,
        description=(
            "**Purpose:** Filter and retrieve monitoring records for a specific server based on a metric condition.\n\n"
            "**Input:** A JSON string specifying the `server_serial`, `metric`, `operator`, and `value`.\n\n"
            "**Supported Metrics:**\n"
            "- 'cpu_util': CPU utilization (%)\n"
            "- 'amb_temp': Ambient temperature (°C)\n"
            "- 'peak': Peak performance value\n\n"
            "**Supported Operators:**\n"
            "- 'greater_than': Metric is greater than the given value\n"
            "- 'less_than': Metric is less than the given value\n"
            "- 'equals': Metric is equal to the given value\n\n"
            "**Example Inputs:**\n"
            "- `{'server_serial': 'SRV123', 'metric': 'cpu_util', 'operator': 'greater_than', 'value': 80}`\n"
            "- `{'server_serial': 'ABC456', 'metric': 'amb_temp', 'operator': 'less_than', 'value': 25}`\n\n"
            "**Returns:**\n"
            "- A list of up to 20 matching records (timestamp and metric value)\n"
            "- Total count of matching entries\n"
            "- A message if no records matched\n\n"
            "**Notes:**\n"
            "- Ensure all fields are correctly specified in double-quoted JSON.\n"
            "- Server serial is case-insensitive.\n"
            "- Returns an error message if fields are missing or invalid.\n\n"
            "**Important Notes for Agent:** Upon receiving this data, you have sufficient information to answer the user's query directly. Present the filtered records as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool."
        )
    ),
    Tool(
        name="DetectAnomalies",
        func=detect_anomalies,
        description=(
            "**Purpose:** Analyze server monitoring data to detect significant anomalies across key performance metrics. "
            "This tool employs statistical analysis (modified Z-score with median and MAD) to pinpoint abnormal spikes "
            "in CPU utilization, temperature, and power usage.\n\n"
            "**Usage:** You can run this tool for all servers or specify a particular server or metric using natural language.\n\n"
            "**Supported Metrics (automatically inferred from query):**\n"
            "- 'cpu_util': CPU utilization (%)\n"
            "- 'amb_temp': Ambient temperature (°C)\n"
            "- 'cpu_watts': CPU power consumption (watts)\n"
            "- 'dimm_watts': DIMM memory power usage (watts)\n\n"
            "**Example Queries:**\n"
            "- 'Check anomalies for server SRV123'\n"
            "- 'Analyze CPU utilization across all servers'\n"
            "- 'Find temperature spikes in ABC789'\n"
            "- 'Show anomalies in power consumption'\n\n"
            "**Returns:** A comprehensive text-based anomaly report, which directly addresses the user's request. This report includes:\n"
            "  • Median baseline values for each metric\n"
            "  • Critical anomalies (Z-score > 5)\n"
            "  • Major anomalies (3.5 < Z-score ≤ 5)\n"
            "  • Frequent times of anomalies (hour-level)\n"
            "  • Suggested root causes based on patterns\n\n"
            "**Important Notes for Agent:**\n"
            "- The function works with historical records stored for each server.\n"
            "- If no significant anomalies are detected, a clean report is returned.\n"
            "- Supports natural language input only.\n"
            "**- Upon receiving the anomaly report from this tool, you have sufficient information to answer the user's query directly. Present the detailed report as your final answer without further tool calls for this request. If the user asks for a 'full report' or 'PDF/Word' of this data, recommend using the `GenerateReport` tool.**"
        )
    ),
    Tool(
        name="QueryDocuments",
        func=query_documents,
        description=(
            "**Purpose:** Search and query the HPE Energy Efficiency and Sustainability knowledge base. "
            "This tool provides comprehensive guidance on HPE server energy optimization, carbon emission reduction, "
            "and sustainable data center operations.\n\n"
            "**Triggers:** 'How to reduce server power consumption', 'HPE energy efficiency recommendations', "
            "'PUE optimization strategies', 'Carbon footprint reduction', 'HPE iLO power management', "
            "'Thermal efficiency issues', 'Server consolidation advice', 'HPE OneView energy features', etc.\n\n"
            "**Input:** A specific question about:\n"
            "- HPE server energy efficiency (Power Usage Effectiveness, Energy Efficiency Rating)\n"
            "- HPE-specific power management technologies (iLO, Dynamic Power Capping, Power Regulator)\n"
            "- Carbon Usage Effectiveness (CUE) and emissions tracking\n"
            "- Renewable energy integration strategies\n"
            "- Thermal management and cooling optimization\n"
            "- HPE server hardware efficiency recommendations (Gen10/Gen11 ProLiant)\n"
            "- Data center sustainability compliance (ASHRAE 90.4, ISO 14001/50001, Energy Star)\n"
            "- HPE infrastructure optimization (OneView, InfoSight, Synergy)\n\n"
            "**Example:** 'How can I optimize power consumption on HPE ProLiant Gen11 servers with high idle power?'\n\n"
            "**Important Notes for Agent:** Upon receiving search results from this tool, you have sufficient information to answer the user's query directly. Synthesize the relevant information from the documents and present it as your final answer without further tool calls for this request."
        )
    ),
    Tool(
        name="ListAvailableDocuments",
        func=list_available_documents,
        description=(
            "**Purpose:** List all available HPE energy efficiency and sustainability documents in the knowledge base. "
            "This tool helps users understand what specific HPE energy optimization resources are available for querying.\n\n"
            "**Triggers:** 'What HPE energy documents do you have?', 'List available sustainability guides', "
            "'Show HPE efficiency documentation', 'What energy resources are loaded?', 'Available HPE knowledge base files'.\n\n"
            "**Input:** An empty string or 'list'.\n"
            "**Example:** `Action: ListAvailableDocuments[]`\n\n"
            "**Returns:** Information about available document categories such as:\n"
            "- HPE Energy Efficiency Standards and Guidelines\n"
            "- HPE Server Power Management Documentation\n"
            "- Carbon Emission Reduction Frameworks\n"
            "- HPE Data Center Sustainability Best Practices\n"
            "- HPE-Specific Tool Integration Guides (iLO, OneView, InfoSight)\n"
            "- Compliance and Certification Documentation\n"
            "- HPE Hardware Optimization Manuals\n\n"
            "**Important Notes for Agent:** Upon receiving this list, you have sufficient information to answer the user's query directly. Present the list of available documents as your final answer without further tool calls for this request."
        )
    ),

    Tool(
    name="GenerateReport",
    func=generate_csv_report,
    description=(
        "**Purpose:** Generate a downloadable CSV report for server data based on a natural language query.\n\n"
        "**Triggers:** Use this tool when the user asks for a 'CSV report', 'export to Excel', 'generate report', 'download data', etc.\n\n"
        "**Input:** Natural language query such as:\n"
        "- 'all servers'\n"
        "- 'top 5 cpu servers'\n"
        "- 'servers with cpu above 75%'\n\n"
        "**Returns:** File path of the generated `.csv` file (e.g., `temp_reports/server_report_abc.csv`).\n"
        "Final answer should include: 'Your detailed CSV report is ready! You can download it here: [file_path]'"
    )
)
]
