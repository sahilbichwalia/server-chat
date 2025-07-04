from typing import Optional
from src.config.logging_config import setup_logging
from src.common.common import LLM_INSTANCE
import threading
import re
import gradio as gr
from src.tools.rag import auto_initialize_rag_on_startup, get_rag_system_status
from src.agents.setup import setup_agent
logger = setup_logging()

# -----------------------------
# Enhanced Chat System
# -----------------------------
class IntegratedChatSystem:
    def __init__(self):
        global llm_instance # Declare intent to modify/set globals
        self.logger = logger
        self.semaphore = threading.Semaphore(1)

        self.logger.info("Initializing IntegratedChatSystem...")
        # Ensure llm_instance is assigned to the global
        llm_instance = LLM_INSTANCE

        self.logger.info(f"ChatOllama model '{llm_instance.model}' initialized and assigned globally.")

         # Auto-initialize RAG system
        rag_success = auto_initialize_rag_on_startup(llm_instance)
        if rag_success:
            self.logger.info("RAG system auto-initialization completed successfully")
        else:
            self.logger.warning("RAG system auto-initialization failed, but continuing with setup")

        self.agent_executor = setup_agent(llm_instance) # Pass the global instance
        
        # Log final RAG status
        rag_status = get_rag_system_status()
        self.logger.info(f"Final RAG system status: {rag_status}")

        if self.agent_executor:
            self.logger.info("Chat agent setup complete.")
        else:
            # setup_agent now raises RuntimeError on failure, so this else might not be hit
            # unless setup_agent is changed to return None on some non-critical error.
            self.logger.critical("Chat agent setup did not return an executor. System may not function.")
            # This would typically be caught by the RuntimeError in setup_agent
            # raise RuntimeError("Agent Executor could not be initialized.")


    def clean_response_text(self, response_text: str) -> str:
        # Remove common error prefixes that might slip through
        prefixes_to_remove = [
            r'^Could not parse LLM output:.*?For troubleshooting.*?\n',
            r'^I encountered a parsing error:.*?\n',
            r'^Format Answering Task Error:.*?\n',
            # Add more specific prefixes if observed
        ]
        for prefix in prefixes_to_remove:
            response_text = re.sub(prefix, '', response_text, flags=re.DOTALL | re.IGNORECASE)

        # Prioritize extracting "Final Answer:" if present, as per ReAct convention
        final_answer_match = re.search(r"Final Answer:\s*(.*)", response_text, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            response_text = final_answer_match.group(1).strip()
        else:
            # If no "Final Answer:", check if it's an Observation (tool output)
            # This part is tricky because `return_direct=True` tools bypass this.
            # This cleanup is more for when `return_direct=False` or if the agent is trying to summarize.
            # For now, this simpler cleanup should be okay since most tools are direct.
            pass

        # Remove any remaining ReAct keywords if the LLM didn't stop cleanly after a Final Answer
        # This is a safeguard.
        react_keywords_pattern = r"\n*\s*(Thought:|Action:|Action Input:|Observation:).*"
        # Split by the first occurrence of any ReAct keyword block
        parts = re.split(react_keywords_pattern, response_text, 1)
        response_text = parts[0].strip()


        response_text = re.sub(r'\n{3,}', '\n\n', response_text) # Normalize multiple newlines
        return response_text.strip()

    def process_query(self, query: str, chat_history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]], Optional[str]]:
        if not self.semaphore.acquire(timeout=0.5):
            self.logger.warning("Semaphore acquisition timeout. Request concurrent or system busy.")
            return query, chat_history, None  # Keep input box, no file path

        try:
            self.logger.info(f"Processing query: '{query}'")

            if not query.strip():
                response = "Please enter a question."
                chat_history.append((query, response))
                return "", chat_history, None  # Clear input, no file

            if not self.agent_executor:
                self.logger.error("Agent executor is not initialized!")
                response = "The assistant is not properly initialized. Please check system logs."
                chat_history.append((query, response))
                return "", chat_history, None

            agent_input = {"input": query}
            agent_response_dict = self.agent_executor.invoke(agent_input)

            response_text = agent_response_dict.get("output", "Sorry, I could not process that effectively right now.")
            cleaned_response = self.clean_response_text(response_text)

            if not cleaned_response or len(cleaned_response) < 3:
                cleaned_response = "I processed your request, but there wasn't much to say, or the result was empty. Can I help with something else?"
                self.logger.warning(f"Agent returned a short/empty response. Original: '{response_text}', Using: '{cleaned_response}'")

            chat_history.append((query, cleaned_response))
            self.logger.info(f"Successfully processed query. Response: '{cleaned_response[:100]}...'")

            # 🔍 Check for CSV path in response
            csv_match = re.search(r"download it here:\s*(temp_reports[\\/][\w\-\.]+\.csv)", cleaned_response)
            csv_file = csv_match.group(1).replace("\\", "/") if csv_match else None

            if csv_file:
                return "", chat_history, gr.update(value=csv_file, visible=True)
            else:
                return "", chat_history, gr.update(visible=False)

        except Exception as e:
            self.logger.error(f"Error processing query '{query}': {e}", exc_info=True)
            error_response_msg = (
                "I encountered a system error while processing your request. Please try rephrasing or simplify your query. "
                "If the problem continues, please check the logs or contact support."
            )
            chat_history.append((query, error_response_msg))
            return "", chat_history, None

        finally:
            self.semaphore.release()
