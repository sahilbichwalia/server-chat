from langchain.tools import Tool
from src.tools.tool_function import list_servers, get_top_servers_by_cpu_util, get_specific_server_cpu_utilization, \
    get_lowest_servers_by_cpu_util, get_top_servers_by_ambient_temp, get_specific_server_ambient_temp,get_lowest_servers_by_ambient_temp, get_top_servers_by_peak, get_specific_server_peak_data, \
    get_lowest_servers_by_peak, get_server_stats, calculate_carbon_footprint, co2_emission_server, calculate_carbon_footprint_lowest, identify_high_cpu_servers, get_server_timestamps, get_filtered_server_records, detect_anomalies
from src.tools.rag import query_documents, list_available_documents
from src.tools.prediction import predict_server_metrics_tool
from src.tools.tool_function import identify_low_cpu_servers

tools = [
    Tool(
        name="ListServers",
        func=list_servers,
        description=(
            "Use this tool when the user asks to view, list, or summarize all available servers. "
            "Triggers include: 'List all servers', 'Show servers being monitored', 'What servers are active?'. "
            "Returns a human-readable summary of all monitored servers, including:\n"
            "Use this tool to list all monitored servers. "
            "Input should be an empty string. "
            "Example: Action: ListAllServers[]"
            "- Serial number\n"
            "- Last seen timestamp\n"
            "- Total records\n"
            "- Average CPU usage\n"
            "- Average ambient temperature"
        ),
        return_direct=True
    ),
    Tool(
    name="GetTopServersByCPUUtil",
    func=get_top_servers_by_cpu_util,
    description=(
        "Use this tool to retrieve servers with the highest CPU utilization. "
        "Extracts how many top servers to show from the query (default: 10; 'all' returns all). "
        "including numeric words such as 'one server', 'two servers', 'three servers', etc. (default: 10; 'all' returns all). "
        "tell me the top 100 server which have highest cpu utlization\n "
         "tell me the top 100 server which have highest cpu utlization if the count of server is greater tahn list server then return default\n "
        "Example inputs:\n"
        "- 'Top 5 CPU servers'\n"
        "- 'Which 3 servers have highest CPU utilization?'\n"
        "- 'Show all high CPU servers'\n\n"
        "Returns for each server:\n"
        "- Serial number\n"
        "- Peak CPU (%)\n"
        "- Timestamp\n"
        "- Power (Watts)\n"
        "- Temperature (°C)\n"
        "- Fan speed (RPM)\n\n"
        "Format: Action: GetTopServersByCPUUtil[\"<query>\"]"
    ),
    return_direct=True),
    
Tool(
    name="GetServerCPUUtilization",
    func=get_specific_server_cpu_utilization,
    description=(
        "Use this tool to get detailed CPU utilization information for SPECIFIC server(s).\n\n"
        "The tool uses robust server identification to handle various formats including:\n"
        "- Full server names (e.g., 'server SGH227WTNK')\n"
        "- Direct serial numbers (e.g., 'SGH227WTNK')\n"
        "- Multiple servers in one query (e.g., 'SGH227WTNK and ABC123', 'SGH227WTNK, DEF456')\n"
        "- Case-insensitive matching\n"
        "- Handles common typos and formatting variations\n\n"
        "Example queries:\n"
        "- 'What is the CPU utilization of server SGH227WTNK?'\n"
        "- 'Show CPU stats for SGH227WTNK and ABC123'\n"
        "- 'CPU usage for servers SGH227WTNK, DEF456'\n"
        "- 'SGH227WTNK CPU utilization details'\n"
        "- 'Compare CPU usage SGH227WTNK ABC123'\n\n"
        "Returns detailed analysis:\n"
        "For single server:\n"
        "- Average CPU Utilization (%)\n"
        "- Peak CPU Utilization (%) with timestamp\n"
        "- Power consumption at peak\n"
        "- Temperature and fan speed at peak\n"
        "- CPU efficiency rating\n"
        "- Fleet ranking position\n\n"
        "For multiple servers:\n"
        "- Summary statistics across servers\n"
        "- Individual server breakdowns\n"
        "- Comparative analysis with rankings\n\n"
        "If server(s) not found, returns helpful error message with available server examples.\n\n"
        "Format: Action: GetServerCPUUtilization[\"<query>\"]"
    ),
    return_direct=True
),
    
    Tool(
    name="GetLowestServersByCPUUtil",
    func=get_lowest_servers_by_cpu_util,
    description=(
        "Use this tool to find servers with the lowest CPU utilization. It extracts how many servers to show from the user's query, "
        "including numeric words such as 'one server', 'two servers', 'three servers', etc. (default: 10; 'all' returns all). "
         "tell me the top 100 server which have Lowest cpu utlization\n "
         "tell me the top 100 server which have Lowest cpu utlization if the count of server is greater tahn list server then return default\n "
        "Example queries: "
        "'Top 5 servers with lowest CPU utilization', "
        "'Which three servers have the lowest CPU usage?', "
        "'Show all low CPU servers', "
        "'List one server with lowest CPU utilization'. "
        "Returns for each server: "
        "- Serial number "
        "- Lowest CPU (%) "
        "- Timestamp "
        "- Power (Watts) "
        "- Temperature (°C) "
        "- Fan speed (RPM)"
    ),
    return_direct=True
),
Tool(
    name="GetTopServersByAmbientTemp",
    func=get_top_servers_by_ambient_temp,
    description=(
        "Use this tool to find servers ranked by their highest ambient temperature records. "
        "It interprets the user's query to extract how many top servers to show, including numeric words such as "
         "tell me the top 100 server which have highest Ambient Temprature\n "
         "tell me the top 100 server which have highest Ambient Temprature if the count of server is greater tahn list server then return default\n "
        "'one', 'two', 'three', etc., or 'all' for all servers with data. If no number is specified, it shows all servers. "
        "Example queries include: "
        "'Top 5 servers with highest ambient temperature', "
        "'Which three servers have the highest ambient temperature?', "
        "'Show all servers by ambient temperature', "
        "'List one server with highest ambient temperature'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Highest ambient temperature (°C) "
        "- Timestamp of the highest temperature record "
        "- CPU utilization (%) at that time "
        "- CPU power consumption (Watts) "
        "- DIMM power consumption (Watts) "
        "Handles incomplete data gracefully and informs if requested count exceeds available data."
    ),
    return_direct=True
),
Tool(
    name="GetSpecificServerAmbientTemp",
    func=get_specific_server_ambient_temp,
    description=(
        "Use this tool to get ambient temperature data for specific server(s) identified by their serial numbers. "
        "It can handle single or multiple servers in one query with robust server identification. "
        "The function accepts natural language queries containing server serial numbers and returns detailed "
        "ambient temperature information. "
        "Example queries include: "
        "'What is the ambient temperature for server SGH227WTNK?', "
        "'Show ambient temperature data for SGH227WTNK and ABC123', "
        "'Get temperature info for server XYZ456', "
        "'SGH227WTNK, DEF456 ambient temperature'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Highest ambient temperature recorded (°C) "
        "- Average ambient temperature (°C) "
        "- Timestamp of the highest temperature record "
        "- CPU utilization (%) at peak temperature "
        "- CPU power consumption (Watts) at peak "
        "- DIMM power consumption (Watts) at peak "
        "Handles multiple servers with summary statistics and gracefully handles missing data. "
        "Uses the same robust server identification patterns as the CO2 emission function."
    ),
    return_direct=True
),
Tool(
    name="GetLowestServersByAmbientTemp",
    func=get_lowest_servers_by_ambient_temp,
    description=(
        "Use this tool to find servers ranked by their lowest ambient temperature records. "
         "tell me the top 100 server which have lowest ambient temperature\n "
         "tell me the top 100 server which have lowest ambient temperature if the count of server is greater tahn list server then return default\n "
        "It interprets the user's query to extract how many bottom servers to show, including numeric words such as "
        "'one', 'two', 'three', etc., or 'all' for all servers with data. If no number is specified, it shows all servers. "
        "Example queries include: "
        "'Bottom 5 servers with lowest ambient temperature', "
        "'Which three servers have the lowest ambient temperature?', "
        "'Show all servers by lowest ambient temperature', "
        "'List one server with lowest ambient temperature'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Lowest ambient temperature (°C) "
        "- Timestamp of the lowest temperature record "
        "- CPU utilization (%) at that time "
        "- CPU power consumption (Watts) "
        "- DIMM power consumption (Watts) "
        "Handles incomplete data gracefully and informs if requested count exceeds available data."
    ),
    return_direct=True
),
Tool(
    name="GetTopServersByPeak",
    func=get_top_servers_by_peak,
    description=(
        "Use this tool to retrieve servers with the highest peak values across all metrics. "
        "The number of top servers to show is extracted from the user query. "
        "tell me the top 100 server which have highest peak values\n "
         "tell me the top 100 server which have highest peak values if the count of server is greater tahn list server then return default\n "
        "Handles numeric expressions like 'one server', 'top 3 servers', 'all peak servers', etc. "
        "Defaults to 10 if unspecified. 'All' returns the full list.\n\n"
        "Example queries:\n"
        "- 'Which 5 servers have the highest peak values?'\n"
        "- 'Top 3 servers by peak usage'\n"
        "- 'Show all servers with highest peak value'\n\n"
        "For each server, returns:\n"
        "- Serial number\n"
        "- Highest peak value\n"
        "- Timestamp of peak\n"
        "- CPU Utilization (%)\n"
        "- Ambient Temperature (°C)\n"
        "- CPU Power (Watts)\n\n"
        "Format: Action: GetTopServersByPeak[\"<query>\"]"
    ),
    return_direct=True
),
Tool(
    name="GetSpecificServerPeakData",
    func=get_specific_server_peak_data,
    description=(
        "Use this tool to get peak data for specific server(s) identified by their serial numbers. "
        "It can handle single or multiple servers in one query with robust server identification. "
        "The function accepts natural language queries containing server serial numbers and returns detailed "
        "peak performance information. "
        "Example queries include: "
        "'What is the peak data for server SGH227WTNK?', "
        "'Show peak values for SGH227WTNK and ABC123', "
        "'Get peak performance for server XYZ456', "
        "'Lowest peak servers abc123',"
        "'SGH227WTNK, DEF456 peak data'. "
        "For each server, it returns: "
        "- Server serial number "
        "- Highest peak value recorded "
        "- Average peak value "
        "- Timestamp of the highest peak record "
        "- CPU utilization (%) at peak "
        "- Ambient temperature (°C) at peak "
        "- CPU power consumption (Watts) at peak "
        "Handles multiple servers with summary statistics and gracefully handles missing data. "
        "Uses the same robust server identification patterns as other specific server functions."
    ),
    return_direct=True
),
Tool(
    name="GetLowestServersByPeak",
    func=get_lowest_servers_by_peak,
    description=(
        "Use this tool to retrieve servers with the lowest peak values across all metrics. "
        "The number of servers to show is extracted from the user query. "
         "tell me the top 100 server which have  Lowest peak values\n "
         "tell me the top 100 server which have lowest peak values if the count of server is greater tahn list server then return default\n "
        "Handles phrases like 'one server', 'bottom 3 servers', 'all low peak servers', etc. "
        "Defaults to 10 if not specified. 'All' returns the full list.\n\n"
        "Example queries:\n"
        "- 'Show 3 servers with the lowest peak usage'\n"
        "- 'Bottom 5 peak value servers'\n"
        "- 'All servers with the lowest peak values'\n\n"
        "For each server, returns:\n"
        "- Serial number\n"
        "- Lowest peak value\n"
        "- Timestamp of that value\n"
        "- CPU Utilization (%)\n"
        "- Ambient Temperature (°C)\n"
        "- CPU Power (Watts)\n\n"
        "Format: Action: GetLowestServersByPeak[\"<query>\"]"
    ),
    return_direct=True
),
Tool(
    name="GetServerStats",
    func=get_server_stats,
    description=(
        "Use this tool to retrieve statistics for a specific server or a summary of the fleet.\n\n"
        "Triggers include:\n"
        "- 'Stats for server ABC123'\n"
        "- 'Give me server ST-998 details'\n"
        "- 'Show latest observation for server Y56-22'\n"
        "- 'Show latest observation for all the server '\n"
        "- 'Show latest observation for each  server'\n"
        "- 'What’s the summary of all servers?'\n\n"
        "If a specific server serial number is mentioned in the query, this tool returns:\n"
        "- Latest record timestamp\n"
        "- CPU Utilization, Peak Value, Power (W), Ambient Temperature\n"
        "- Peak and lowest CPU with timestamps\n"
        "- Max/min ambient temperatures with timestamps\n"
        "- Estimated total energy used and CO₂ emissions\n\n"
        "If no server is specified, it returns a fleet-wide summary:\n"
        "- Top 5 servers by peak CPU usage\n"
        "- Latest CPU and temperature readings\n"
        "- General fleet statistics and record availability\n\n"
        "Example inputs:\n"
        "- 'Show stats for server TDX-901'\n"
        "- 'Fleet summary'\n"
        "- 'Give observation for server XP100'\n\n"
        "Format: Action: GetServerStats[\"<query>\"]"
    ),
    return_direct=True
),

# Updated Tool Definitions
Tool(
    name="CalculateCarbonFootprint",
    func=calculate_carbon_footprint,
    description=(
        "Use this tool to calculate the carbon footprint of multiple servers or fleet-wide analysis.\n\n"
        "The tool determines whether to use 'average', 'low-carbon', or 'high-carbon' grid intensity based on keywords in the query "
        "(e.g., 'renewable', 'coal', 'low carbon', etc.). It extracts server counts (like 'top 5', 'ten servers', or 'all') "
        "and optionally filters by grid type.\n\n"
        "This tool is for MULTIPLE server analysis only. For individual servers, use CO2EmissionServer instead.\n\n"
        "Example queries:\n"
        "- 'Show CO2 emissions for all servers using renewable energy.'\n"
        "- 'Calculate carbon footprint for top 3 servers using high carbon grid.'\n"
        "- 'List servers with highest emissions based on coal grid.'\n"
        "- 'Top 10 servers by carbon emissions'\n"
        "- 'Fleet-wide carbon footprint analysis'\n\n"
        "Returns fleet summary including:\n"
        "- Total CO₂ Emissions across all servers\n"
        "- Average emissions per server\n"
        "- Top N highest emitting servers with details\n"
        "- Energy efficiency distribution\n\n"
        "Format: Action: CalculateCarbonFootprint[\"<query>\"]"
    ),
    return_direct=True
),

Tool(
    name="CO2EmissionServer",
    func=co2_emission_server,
    description=(
        "Use this tool to calculate the carbon footprint of SPECIFIC server(s) - single or multiple.\n\n"
        "The tool uses robust server identification to handle various formats including:\n"
        "- Full server names (e.g., 'server SGH227WTNK')\n"
        "- Direct serial numbers (e.g., 'SGH227WTNK')\n"
        "- Multiple servers in one query (e.g., 'SGH227WTNK and ABC123', 'SGH227WTNK, DEF456')\n"
        "- Case-insensitive matching\n"
        "- Handles common typos and formatting variations\n\n"
        "Also determines carbon intensity based on keywords in the query "
        "(e.g., 'renewable', 'coal', 'low carbon', etc.).\n\n"
        "Example queries:\n"
        "- 'What is the carbon footprint of server SGH227WTNK?'\n"
        "- 'CO2 emissions for SGH227WTNK and ABC123 using renewable energy'\n"
        "- 'Show carbon footprint of servers SGH227WTNK, DEF456 with high carbon grid'\n"
        "- 'SGH227WTNK ABC123 carbon emissions'\n"
        "- 'Compare CO2 for SGH227WTNK and DEF456'\n\n"
        "Returns detailed analysis:\n"
        "For single server:\n"
        "- Energy Consumed (kWh)\n"
        "- CO₂ Emissions (kg)\n"
        "- Carbon Intensity used in calculation\n"
        "- Average CPU Utilization (%)\n"
        "- Energy Efficiency Rating\n\n"
        "For multiple servers:\n"
        "- Total and average CO₂ emissions\n"
        "- Individual server breakdowns\n"
        "- Comparative analysis\n\n"
        "If server(s) not found, returns helpful error message with available server examples.\n\n"
        "Format: Action: CO2EmissionServer[\"<query>\"]"
    ),
    return_direct=True
),

Tool(
    name="CalculateCarbonFootprintLowest",
    func=calculate_carbon_footprint_lowest,
    description=(
        "Use this tool to calculate the carbon footprint of one or more servers based on estimated energy consumption, "
        "specifically showing servers with the LOWEST CO₂ emissions.\n\n"
        "The tool determines whether to use 'average', 'low-carbon', or 'high-carbon' grid intensity based on keywords in the query "
        "(e.g., 'renewable', 'coal', 'low carbon', etc.). It extracts server serial numbers, counts (like 'top 5', 'ten servers', or 'all'), "
        "and optionally filters by grid type.\n\n"
        "If a specific server is mentioned (e.g., 'Server ABC123'), it will return the carbon footprint details for that server only.\n"
        "If a number of top servers are requested, it returns those with the LOWEST CO₂ emissions (most energy-efficient).\n\n"
        "Example queries:\n"
        "- 'Show me the 10 most energy-efficient servers'\n"
        "- 'Which servers have the lowest carbon footprint?'\n"
        "- 'Top 5 cleanest servers using renewable energy'\n"
        "- 'List servers with minimum CO2 emissions'\n"
        "- 'Show least polluting servers'\n"
        "- 'Most efficient servers by carbon footprint'\n\n"
        "Returns for each server:\n"
        "- Serial number\n"
        "- Energy Consumed (kWh)\n"
        "- CO₂ Emissions (kg)\n"
        "- Average CPU Utilization (%)\n"
        "- Efficiency Rating (based on CPU-to-power ratio)\n\n"
        "Also returns a fleet summary when multiple servers are included, with efficiency distribution.\n\n"
        "Format: Action: CalculateCarbonFootprintLowest[\"<query>\"]"
    ),
    return_direct=True
),

Tool(
    name="IdentifyHighCPUServers",
    func=identify_high_cpu_servers,
    description=(
        "Use this tool to identify servers that have CPU utilization above a specified threshold. "
        "The query should include a numeric threshold (e.g., 'above 80%' or 'more than 70%'). "
        "This tool analyzes all server records and returns those with at least one instance of CPU utilization above the given threshold.\n\n"
        "Example inputs:\n"
        "- 'Show servers with CPU above 90%'\n"
        "- 'List all servers crossing 75% CPU utilization'\n"
        "- 'Which servers hit CPU over 85%?'\n\n"
        "Returns for each matching server:\n"
        "- Serial number\n"
        "- Count and percentage of records where CPU > threshold\n"
        "- Maximum CPU utilization observed\n\n"
        "Notes:\n"
        "- Maximum 20 servers are shown in detail; remaining are summarized.\n"
        "- Results are sorted by percentage of high CPU records and peak CPU observed.\n\n"
        "Format: Action: IdentifyHighCPUServers[\"<query>\"]"
    ),
    return_direct=True,
),
Tool(
    name="GetServerTimestamps",
    func=get_server_timestamps,
    description=(
        "Use this tool to retrieve the list of timestamped monitoring records for a specific server. "
        "The query must contain the server serial number (e.g., 'server A123B', 'timestamps for server XYZ001'). "
        "This helps in auditing the monitoring intervals or understanding activity history.\n\n"
        "Example inputs:\n"
        "- 'Show timestamps for server A12B9'\n"
        "- 'Get monitoring history of server 7GHT9'\n"
        "- 'When was server TEST-SRVR last active?'\n\n"
        "Returns:\n"
        "- The server’s total number of records\n"
        "- Up to 20 of the earliest timestamps (in order of appearance in data)\n"
        "- A note about any additional timestamps\n\n"
        "Notes:\n"
        "- The tool tries to match server serials even with partial or imprecise queries.\n"
        "- If no match is found, it will suggest a few available server serials.\n\n"
        "Format: Action: GetServerTimestamps[\"<query>\"]"
    ),
    return_direct=True,
),
Tool(
    name="FilterServerRecords",
    func=get_filtered_server_records,
    description=(
        "Use this tool to filter and retrieve monitoring records for a specific server based on a metric condition. "
        "The input must be a JSON string specifying the server serial, metric, comparison operator, and value.\n\n"
        "Supported metrics:\n"
        "- 'cpu_util': CPU utilization (%)\n"
        "- 'amb_temp': Ambient temperature (°C)\n"
        "- 'peak': Peak performance value\n\n"
        "Supported operators:\n"
        "- 'greater_than': Metric is greater than the given value\n"
        "- 'less_than': Metric is less than the given value\n"
        "- 'equals': Metric is equal to the given value\n\n"
        "Example inputs:\n"
        "- '{\"server_serial\": \"SRV123\", \"metric\": \"cpu_util\", \"operator\": \"greater_than\", \"value\": 80}'\n"
        "- '{\"server_serial\": \"ABC456\", \"metric\": \"amb_temp\", \"operator\": \"less_than\", \"value\": 25}'\n\n"
        "Returns:\n"
        "- A list of up to 20 matching records (timestamp and metric value)\n"
        "- Total count of matching entries\n"
        "- A message if no records matched\n\n"
        "Notes:\n"
        "- Ensure all fields are correctly specified in double-quoted JSON.\n"
        "- Server serial is case-insensitive.\n"
        "- Returns an error message if fields are missing or invalid.\n\n"
        "Format: Action: FilterServerRecords[\"<JSON-formatted string>\"]"
    ),
    return_direct=True,
),
Tool(
    name="DetectAnomalies",
    func=detect_anomalies,
    description=(
        "Use this tool to analyze server monitoring data and detect significant anomalies across key performance metrics. "
        "The tool uses statistical analysis (modified Z-score with median and MAD) to identify abnormal spikes in CPU utilization, temperature, and power usage.\n\n"
        
        "You can run this tool for all servers or specify a particular server or metric in natural language.\n\n"
        
        "Supported metrics (automatically inferred from query):\n"
        "- 'cpu_util': CPU utilization (%)\n"
        "- 'amb_temp': Ambient temperature (°C)\n"
        "- 'cpu_watts': CPU power consumption (watts)\n"
        "- 'dimm_watts': DIMM memory power usage (watts)\n\n"
        
        "Example queries:\n"
        "- 'Check anomalies for server SRV123'\n"
        "- 'Analyze CPU utilization across all servers'\n"
        "- 'Find temperature spikes in ABC789'\n"
        "- 'Show anomalies in power consumption'\n\n"
        
        "Returns:\n"
        "- Enhanced anomaly report (text) including:\n"
        "  • Median baseline values for each metric\n"
        "  • Critical anomalies (Z-score > 5)\n"
        "  • Major anomalies (3.5 < Z-score ≤ 5)\n"
        "  • Frequent times of anomalies (hour-level)\n"
        "  • Suggested root causes based on patterns\n\n"
        
        "Notes:\n"
        "- The function works with historical records stored for each server.\n"
        "- If no significant anomalies are detected, a clean report is returned.\n"
        "- Supports natural language input only.\n\n"
        
        "Format: Action: DetectAnomalies[\"<natural language query>\"]"
    ),
    return_direct=True,
),
 Tool(
        name="QueryDocuments",
        func=query_documents,
        description=(
            "Use this tool to search and query the HPE Energy Efficiency and Sustainability knowledge base. "
            "This tool provides comprehensive guidance on HPE server energy optimization, carbon emission reduction, "
            "and sustainable data center operations. "
            "Triggers include: 'How to reduce server power consumption', 'HPE energy efficiency recommendations', "
            "'PUE optimization strategies', 'Carbon footprint reduction', 'HPE iLO power management', "
            "'Thermal efficiency issues', 'Server consolidation advice', 'HPE OneView energy features'. "
            "Input should be a specific question about:\n"
            "- HPE server energy efficiency (Power Usage Effectiveness, Energy Efficiency Rating)\n"
            "- HPE-specific power management technologies (iLO, Dynamic Power Capping, Power Regulator)\n"
            "- Carbon Usage Effectiveness (CUE) and emissions tracking\n"
            "- Renewable energy integration strategies\n"
            "- Thermal management and cooling optimization\n"
            "- HPE server hardware efficiency recommendations (Gen10/Gen11 ProLiant)\n"
            "- Data center sustainability compliance (ASHRAE 90.4, ISO 14001/50001, Energy Star)\n"
            "- HPE infrastructure optimization (OneView, InfoSight, Synergy)\n"
            "Example: 'How can I optimize power consumption on HPE ProLiant Gen11 servers with high idle power?'"
        ),
        return_direct=True
    ),
    
    Tool(
        name="ListAvailableDocuments",
        func=list_available_documents,
        description=(
            "Use this tool to list all available HPE energy efficiency and sustainability documents in the knowledge base. "
            "This tool helps users understand what specific HPE energy optimization resources are available for querying. "
            "Triggers include: 'What HPE energy documents do you have?', 'List available sustainability guides', "
            "'Show HPE efficiency documentation', 'What energy resources are loaded?', 'Available HPE knowledge base files'. "
            "Returns information about:\n"
            "- HPE Energy Efficiency Standards and Guidelines\n"
            "- HPE Server Power Management Documentation\n"
            "- Carbon Emission Reduction Frameworks\n"
            "- HPE Data Center Sustainability Best Practices\n"
            "- HPE-Specific Tool Integration Guides (iLO, OneView, InfoSight)\n"
            "- Compliance and Certification Documentation\n"
            "- HPE Hardware Optimization Manuals\n"
            "Input should be an empty string or 'list'. "
            "Example usage: Action: ListAvailableDocuments[]"
        ),
        return_direct=True
    ),
     Tool(
    name="ServerMetricsPredictor",
    func=predict_server_metrics_tool,
    description=(
        "Predict server metrics (CPU utilization, ambient temperature, power consumption) for future dates. "
        "Uses Prophet machine learning models trained on historical server data. "
        "Input should be a natural language query containing:\n"
        "- Server serial number (e.g., '2M270600W3')\n"
        "- Target date (e.g., 'March 27, 2028', 'tomorrow', '2028-03-27')\n"
        "- Optional: specific metrics (CPU utilization, ambient temperature, peak power, etc.)\n\n"
        "Examples:\n"
        "- 'CPU utilization for server 2M270600W3 on March 27, 2028'\n"
        "- 'Ambient temperature for server ABC123 tomorrow'\n"
        "- 'What will be the peak power for server XYZ789 next week?'\n"
        "- 'Predict all metrics for server 2M270600W3 on 2028-12-25'"
    ),
    return_direct=True
),
Tool(
    name="IdentifyLowCpuServers", 
    func=identify_low_cpu_servers,
    description=(
        "Use this tool to find servers with CPU usage below a specified threshold. "
        "The query must contain a numeric threshold value (e.g., 'CPU below 20%', 'underutilized servers under 10').\n\n"
        "Example inputs:\n"
        "- 'Show servers with CPU below 15%'\n"
        "- 'Find underutilized servers under 25%'\n"
        "- 'Which servers have low CPU usage below 30%?'\n\n"
        "Returns:\n"
        "- List of servers sorted by prevalence of low CPU usage and minimum CPU observed\n"
        "- For each server: number of low CPU instances, total records, percentage of time below threshold\n"
        "- Lowest CPU value recorded for each server\n"
        "- Shows up to 20 servers with additional count if more exist\n\n"
        "Notes:\n"
        "- Threshold must be between 0-100%\n"
        "- Only processes servers with valid CPU utilization data\n"
        "- Results help identify underutilized or idle servers\n"
        "- Useful for capacity planning and resource optimization\n\n"
        "Format: Action: IdentifyLowCpuServers[\"<query>\"]"
    ),
    return_direct=True,
)

]