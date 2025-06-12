from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from typing import Any
from src.tools.tools_description import tools
from src.config.logging_config import setup_logging
import re
import os # Import os
from langchain_google_genai import ChatGoogleGenerativeAI # Ensure this is imported for LLM_INSTANCE
from dotenv import load_dotenv # Ensure this is imported

logger = setup_logging()

# Load environment variables if not already loaded
load_dotenv()
api_key = os.getenv("api_key")
os.environ["GOOGLE_API_KEY"] = api_key

# Global LLM_INSTANCE for consistency, assuming BASE_MODEL is defined elsewhere (e.g., config.py)
# For this example, I'll define a placeholder for BASE_MODEL if it's not in this snippet
try:
    from src.common.constant import BASE_MODEL, EMBEDDING_MODEL
except ImportError:
    BASE_MODEL = "gemini-1.5-pro" # Placeholder if not in constant.py
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # Placeholder if not in constant.py


LLM_INSTANCE= ChatGoogleGenerativeAI(
    model=BASE_MODEL,
    temperature=0.5,
    # convert_system_message_to_human=True, # Deprecated warning, consider removing if issues arise
    top_k=30,
    top_p=0.85,
    # request_timeout=120
)

# Global instances
vectorstore = None

def setup_agent(current_llm):
    # The current_llm parameter already provides the LLM instance, so global llm_instance might not be strictly needed here,
    # but I'll keep it for clarity if other parts of your system rely on it.
    global LLM_INSTANCE # Reference the global LLM_INSTANCE defined above

    system_template = """You are a Server Monitoring Assistant.
    If any type of greeting query comes, respond to it directly without using tools.
    Your primary task is to respond to user queries by selecting the appropriate tool and providing the correct input for it.
    You MUST ALWAYS use the ReAct format when a tool is needed.

    **IMPORTANT: All tools return their output as formatted strings. You are responsible for synthesizing these outputs into a clear, concise, and helpful final answer to the user's original query.** Do NOT simply repeat the tool's output verbatim unless it is already a complete, user-friendly answer.

    **MANDATORY ReAct FORMAT (Thought, Action, Action Input):**
    When you need to use a tool, you MUST output your reasoning and the tool call in this EXACT, multi-line format. Each part MUST be on its own line.

    Thought: [Your concise reasoning for choosing the tool and formulating the input. Always include this line.]
    Action: [The EXACT name of ONE tool from the available tools list. Always include this line. Do NOT invent tool names.]
    Action Input: [The specific input string or JSON for the chosen tool. Always include this line. If the tool description says input is an empty string, provide an empty string. For JSON, provide a single-line, correctly formatted JSON string with double quotes for keys and string values.]

    After outputting Action Input, STOP. The system will provide an Observation with the tool's string output.
    DO NOT add any other text, commentary, or formatting around or before the Thought, Action, or Action Input lines.
    DO NOT write "Observation:" yourself.

    Example 1 (Tool expecting simple text, possibly empty):
    Thought: The user wants to list all servers. I should use the ListServers tool. The tool description indicates an empty input is appropriate.
    Action: ListServers
    Action Input:

    Example 2 (Tool expecting specific text):
    Thought: The user wants the top 3 CPU servers. I should use the GetTopServersByCPUUtil tool and specify '3'.
    Action: GetTopServersByCPUUtil
    Action Input: 3

    Example 3 (Tool expecting JSON):
    Thought: The user wants records for server SGH123 where CPU utilization is greater than 50. I should use FilterServerRecords with the specified JSON structure.
    Action: FilterServerRecords
    Action Input: {{"server_serial": "SGH123", "metric": "cpu_util", "operator": "greater_than", "value": 50}}

    If the question is a simple greeting, a follow-up to a previous tool's direct output, or does not require a tool (e.g., "hello", "thank you", "that's all"), respond directly using the Final Answer format.
    **When you have gathered all necessary information from tool outputs and formed a complete, user-friendly response, present it as your final answer.**

    Thought: The user asked for a greeting. I should respond directly.
    Final Answer: Hello! How can I help you today?

    Thought: The user asked for high CPU servers, and I have obtained the list from the IdentifyHighCPUServers tool. This information directly answers the user's query.
    Final Answer: Here are the servers with CPU records above 35.0%:\n[Tool output from IdentifyHighCPUServers]

    **Available Tools Information:**
    {tools}

    **Tool Names for the 'Action:' line (Case Sensitive):**
    {tool_names}

    **Key Reminders:**
    - Your response, when using a tool, MUST strictly follow the Thought, Action, Action Input format. Each part MUST be on its own line.
    - The 'Action:' line MUST contain ONLY the exact tool name from the 'Tool Names for the Action line' list. Do NOT add any other characters, markdown (like '**' or '_'), quotes, or extra newlines to the tool name itself on this line. It must be plain text.
    - The 'Action Input:' line MUST follow immediately after the 'Action:' line.
    - For `FilterServerRecords`, the Action Input MUST be the JSON string. If the user says "server X cpu > 10", you must translate that to the JSON.
    - You are expected to process tool outputs and synthesize them into a final answer. DO NOT simply relay the raw tool output unless it is explicitly designed to be a complete answer (like detailed reports).
    - All tool outputs are already formatted as complete, readable strings - DO NOT attempt to parse, modify, or add formatting to them, but rather integrate them directly or summarize them into your Final Answer.
    - If you are providing a final answer directly without a tool, use the 'Final Answer:' format. Do NOT mix 'Final Answer:' with 'Action:' blocks in the same response. Once you output 'Final Answer:', your response for that turn is complete.
    - Do NOT use any ANSI escape codes (like color codes such as `[0m`) in your Thought, Action, Action Input, or Final Answer sections. All these sections must be plain text.

    Conversation History (for context):
    {chat_history}

    User's current question:
    {input}

    Agent scratchpad (your previous thoughts, actions, and observations for this current question):
    {agent_scratchpad}"""

    # Using ChatPromptTemplate.from_template to ensure proper handling by create_react_agent
    prompt = ChatPromptTemplate.from_template(system_template)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        max_token_limit=1500
    )

    def enhanced_parsing_error_handler(error: Any) -> str:
        error_str = str(error)
        logger.error(f"Agent parsing/execution error: {error_str}", exc_info=True)

        final_answer_match = re.search(r"Final Answer:\s*(.*)", error_str, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            extracted_answer = final_answer_match.group(1).strip()
            if len(extracted_answer) > 5:
                logger.info(f"Extracted a potential Final Answer from error: {extracted_answer[:100]}...")
                return extracted_answer

        guidance = " To help me process your request, please try:\n" \
                   "- Phrasing your question more simply or specifically.\n" \
                   "- Asking for less data (e.g., 'top 3 servers' instead of 'all').\n" \
                   "- Breaking complex questions into smaller parts."

        if "Could not parse LLM output" in error_str or "Could not parse tool invocation" in error_str:
            return "I had a little difficulty formatting my response or deciding the next step. Could you please rephrase your question?" + guidance
        if "Invalid tool" in error_str:
            return "I seem to have chosen an incorrect tool. Please rephrase your request, and I'll try a better one." + guidance
        if "no viable tool" in error_str.lower() or "did not find an Action" in error_str:
            return "I couldn't determine the best way to handle that request with my current tools. Could you try asking differently?" + guidance

        return "I'm sorry, an unexpected issue occurred while I was thinking." + guidance

    try:
        agent = create_react_agent(llm=current_llm, tools=tools, prompt=prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=enhanced_parsing_error_handler,
            max_iterations=10, # Increased max_iterations to allow more reasoning steps
            early_stopping_method="force",
            max_execution_time=100,
            return_intermediate_steps=False,
        )
        logger.info("AgentExecutor created successfully with create_react_agent and refined prompt.")
        return agent_executor

    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Agent creation failed: {e}", exc_info=True)
        raise RuntimeError(f"Agent setup failed, application cannot continue: {e}") from e

