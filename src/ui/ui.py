from src.agents.chat_integration import IntegratedChatSystem
import gradio as gr
from src.config.logging_config import setup_logging
logger = setup_logging()
# -----------------------------
# Gradio UI
# -----------------------------
def create_gradio_interface(server_data=None):
    logger.info("Creating Gradio interface...")
    try:
        # This will initialize llm_instance globally
        chat_system = IntegratedChatSystem()
        logger.info("IntegratedChatSystem initialized for Gradio.")

    except RuntimeError as e: # Catch specific RuntimeError from agent/system setup
        logger.critical(f"Fatal error during Gradio system initialization (RuntimeError): {e}", exc_info=True)
        with gr.Blocks(title="Error - Assistant Unavailable", theme=gr.themes.Soft()) as demo_error:
            gr.Markdown("## 🖥️ Assistant Initialization Failed")
            gr.Markdown(f"A critical runtime error occurred during startup: **{str(e)}**")
            gr.Markdown("The assistant is currently unavailable. Please check the application logs for more details. Ensure all dependencies (like Ollama server) are running correctly.")
        return demo_error
    except Exception as e: # Catch any other unexpected errors during setup
        logger.critical(f"Unexpected fatal error during Gradio system initialization: {e}", exc_info=True)
        with gr.Blocks(title="Error - Assistant Unavailable", theme=gr.themes.Soft()) as demo_error:
            gr.Markdown("## 🖥️ Assistant Initialization Failed")
            gr.Markdown(f"An unexpected error occurred during startup: **{str(e)}**")
            gr.Markdown("The assistant is currently unavailable. Please check the application logs.")
        return demo_error

    # If chat_system initialized successfully, proceed to create the main UI
    with gr.Blocks(title="Server Monitoring Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🖥️ Server Monitoring Assistant")
        gr.Markdown("""
        **Ask about:** Server performance (CPU, temperature, peak), Carbon footprint analysis.
        **Tips:** Be specific (e.g., "Top 3 servers with lowest CPU").
        For filtering, the agent needs to create JSON: "Show records for server SGH123 where CPU is above 50"
        """)

        chatbot_ui = gr.Chatbot(label="Assistant Response", height=600, elem_id="chatbot", show_copy_button=True, bubble_full_width=False)
        msg_input = gr.Textbox(
            label="Your question:",
            placeholder="E.g., 'List all servers', 'Top 3 servers by highest peak CPU', 'Carbon footprint for SGH949WW81 low carbon'",
            lines=2,
            show_label=False
        )

        # CSV download component (hidden by default)
        csv_output = gr.File(label="📥 Download CSV Report", visible=False)

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary", scale=3)
            clear_btn = gr.ClearButton([msg_input, chatbot_ui, csv_output], value="Clear Chat", scale=1)

        gr.Examples(
            examples=["List all servers",
                "Top 3 servers by highest peak CPU",
                "Carbon footprint for all servers using average grid",
                "Show me records for server SGH949WW81 where amb_temp is greater_than 28", # Agent must translate this to JSON
                "Identify servers with CPU utilization consistently above 85%",
                "What are the timestamps for server SGH001TEST?",
                "Hello there"],  # Keep your existing examples
            inputs=msg_input,
            label="Example Questions"
        )

        # Wire the click function
        submit_btn.click(
            fn=chat_system.process_query,
            inputs=[msg_input, chatbot_ui],
            outputs=[msg_input, chatbot_ui, csv_output]
        )

        msg_input.submit(
            fn=chat_system.process_query,
            inputs=[msg_input, chatbot_ui],
            outputs=[msg_input, chatbot_ui, csv_output]
        )


        # Wire up the chat processing function
        # submit_btn.click(chat_system.process_query, [msg_input, chatbot_ui], [msg_input, chatbot_ui])
        # msg_input.submit(chat_system.process_query, [msg_input, chatbot_ui], [msg_input, chatbot_ui])

    logger.info("Gradio interface created successfully.")
    return demo
