from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from typing import Any
from src.tools.tools_description import tools
from src.config.logging_config import setup_logging
import re
logger = setup_logging()
# Global instances
llm_instance = None
vectorstore = None #  # Ensure the global LLM instance is accessible
def setup_agent(current_llm):
    global llm_instance # Ensure globals are accessible if needed by tools
    # llm_instance is already set globally in IntegratedChatSystem.__init__
# In the setup_agent function:

    # ... (system_template up to Key Reminders) ...
    system_template = """You are a Server Monitoring Assistant.
    if any type of greeting query come then you respond it not use the tools
Your primary task is to respond to user queries by selecting the appropriate tool and providing the correct input for it.
You MUST ALWAYS use the ReAct format when a tool is needed.

**IMPORTANT: All tools return their output as formatted strings that are ready to display to the user. Do NOT attempt to parse, modify, or reformat tool outputs - they are already properly formatted for presentation.**

**MANDATORY ReAct FORMAT (Thought, Action, Action Input):**
When you need to use a tool, you MUST output your reasoning and the tool call in this EXACT, multi-line format. Each part MUST be on its own line.

Thought: [Your concise reasoning for choosing the tool and formulating the input. Always include this line.]
Action: [The EXACT name of ONE tool from the available tools list. Always include this line. Do NOT invent tool names.]
Action Input: [The specific input string or JSON for the chosen tool. Always include this line. If the tool description says input is an empty string or not strictly needed, provide an empty string or a relevant placeholder like "all" or the query itself if appropriate. For JSON, provide a single-line, correctly formatted JSON string with double quotes for keys and string values.]

After outputting Action Input, STOP. The system will provide an Observation with the tool's string output.
DO NOT add any other text, commentary, or formatting around or before the Thought, Action, or Action Input lines.
DO NOT write "Observation:" yourself.
DO NOT attempt to process or reformat the tool's output - it comes pre-formatted as a string ready for display.

Example 1 (Tool expecting simple text, possibly empty):
Thought: The user wants to list all servers. I should use the ListServers tool. The tool description says Action Input can be an empty string.
Action: ListServers
Action Input:

Example 2 (Tool expecting specific text):
Thought: The user wants the top 3 CPU servers. I should use the TopCPUUtilizationServers tool and specify 'top 3'.
Action: TopCPUUtilizationServers
Action Input: top 3

Example 3 (Tool expecting JSON):
Thought: The user wants records for server SGH123 where CPU utilization is greater than 50. I should use GetFilteredServerRecords with the specified JSON structure.
Action: GetFilteredServerRecords
Action Input: {{"server_serial": "SGH123", "metric": "cpu_util", "operator": "greater_than", "value": 50}}

If the question is a simple greeting, a follow-up to a previous tool's direct output (where return_direct=true), or does not require a tool (e.g., "hello", "thank you"), respond directly using the Final Answer format:
Thought: The user said hello. I should respond in kind.
Final Answer: Hello! How can I help you today?

**Available Tools Information:**
{tools}

**Tool Names for the 'Action:' line (Case Sensitive):**
{tool_names}

**Key Reminders:**
- Your response, when using a tool, MUST strictly follow the Thought, Action, Action Input format. Each part MUST be on its own line.
- The 'Action:' line MUST contain ONLY the exact tool name from the 'Tool Names for the Action line' list. Do NOT add any other characters, markdown (like '**' or '_'), quotes, or extra newlines to the tool name itself on this line. It must be plain text.
- The 'Action Input:' line MUST follow immediately after the 'Action:' line.
- For `GetFilteredServerRecords`, the Action Input MUST be the JSON string. If the user says "server X cpu > 10", you must translate that to the JSON.
- Most tools have `return_direct=True`. Their output goes straight to the user. Do not re-process it unless asked a follow-up.
- All tool outputs are already formatted as complete, readable strings - DO NOT attempt to parse, modify, or add formatting to them.
- If you are providing a final answer directly without a tool, use the 'Final Answer:' format. Do NOT mix 'Final Answer:' with 'Action:' blocks in the same response. Once you output 'Final Answer:', your response for that turn is complete.
- Do NOT use any ANSI escape codes (like color codes such as `[0m`) in your Thought, Action, Action Input, or Final Answer sections. All these sections must be plain text.

Conversation History (for context):
{chat_history}

User's current question:
{input}

Agent scratchpad (your previous thoughts, actions, and observations for this current question):
{agent_scratchpad}"""

    # ... (rest of the setup_agent function)

    # Using ChatPromptTemplate.from_template to ensure proper handling by create_react_agent
    prompt = ChatPromptTemplate.from_template(system_template)

    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=5,
        max_token_limit=1500 # Increased slightly
    )

    def enhanced_parsing_error_handler(error: Any) -> str:
        error_str = str(error)
        logger.error(f"Agent parsing/execution error: {error_str}", exc_info=True) # Full traceback for server logs

        # Try to extract a Final Answer if the LLM put it in the error message
        final_answer_match = re.search(r"Final Answer:\s*(.*)", error_str, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            extracted_answer = final_answer_match.group(1).strip()
            if len(extracted_answer) > 5: # Basic check for non-empty answer
                logger.info(f"Extracted a potential Final Answer from error: {extracted_answer[:100]}...")
                return extracted_answer

        # Default guidance message
        guidance = " To help me process your request, please try:\n" \
                   "- Phrasing your question more simply or specifically.\n" \
                   "- Asking for less data (e.g., 'top 3 servers' instead of 'all').\n" \
                   "- Breaking complex questions into smaller parts."

        if "Could not parse LLM output" in error_str or "Could not parse tool invocation" in error_str:
             return "I had a little difficulty formatting my response or deciding the next step. Could you please rephrase your question?" + guidance
        if "Invalid tool" in error_str:
            return "I seem to have chosen an incorrect tool. Please rephrase your request, and I'll try a better one." + guidance
        if "no viable tool" in error_str.lower() or "did not find an Action" in error_str: # Agent couldn't decide on a tool
            return "I couldn't determine the best way to handle that request with my current tools. Could you try asking differently?" + guidance

        # Generic fallback for other errors caught by this handler
        return "I'm sorry, an unexpected issue occurred while I was thinking." + guidance


    try:
        # First create the agent using create_react_agent
        agent = create_react_agent(llm=current_llm, tools=tools, prompt=prompt)
        
        # Then create the AgentExecutor with the agent
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=enhanced_parsing_error_handler, # Use the robust handler
            max_iterations=3, # Allow a bit more room for complex interactions or retries
            early_stopping_method="force",
            max_execution_time=100, # Slightly longer timeout
            return_intermediate_steps=False, # Keep False for production, True for deep debugging
        )
        logger.info("AgentExecutor created successfully with create_react_agent and refined prompt.")
        return agent_executor

    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Agent creation failed: {e}", exc_info=True)
        # This will be caught by create_gradio_interface's error handling
        raise RuntimeError(f"Agent setup failed, application cannot continue: {e}") from e