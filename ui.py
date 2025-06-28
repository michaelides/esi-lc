# UI elements and Chainlit interface logic for ESI Scholarly Instructor
"""
This module contains helper functions and Chainlit event handlers for the
User Interface (UI) of the ESI Scholarly Instructor application.

It includes:
- Definitions for accepted file types and maximum file size for uploads.
- Functions to set up chat settings (like LLM temperature).
- Logic to handle file uploads from the user.
- An event handler for when chat settings are updated by the user.
- A function to display the initial welcome message.

Note: The main chat lifecycle handlers (`@cl.on_chat_start`, `@cl.on_message`)
are located in `app.py`. This module focuses on UI-specific components and interactions
that are called from `app.py` or directly by Chainlit for specific UI events.
"""

import chainlit as cl
from agent import ESIConversationAgent # Used for type hinting

# --- UI Constants for File Uploads ---
ACCEPTED_FILE_TYPES = [
    "text/plain", ".txt",
    "application/pdf", ".pdf",
    "application/msword", ".doc", # Older Microsoft Word format
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx", # Modern Word
    "text/markdown", ".md",
    "text/csv", ".csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx", # Modern Excel
    # Using application/octet-stream as a general fallback for binary files like RData, RDS, SAV.
    # Specific MIME types might be more accurate if known and consistently used by browsers/OS.
    "application/octet-stream", ".rdata", ".rds", ".sav"
]
"""List of accepted MIME types and file extensions for user uploads."""

MAX_FILE_SIZE_MB = 100
"""Maximum allowable file size for uploads, in megabytes."""


async def setup_chat_settings():
    """
    Sets up the chat settings UI, specifically the LLM temperature slider.

    This function is called at the start of a chat session. It presents
    a slider to the user, allowing them to adjust the LLM's temperature.
    The selected temperature is stored in the user session.
    """
    settings_ui = await cl.ChatSettings(
        [
            cl.input_widget.Slider(
                id="temperature", # Must match the ID used in on_settings_update
                label="LLM Temperature",
                initial=0.7,       # Default temperature
                min=0.0,
                max=1.0,
                step=0.05,         # Granularity of the slider
                description="Adjust the AI's creativity. Lower values (e.g., 0.2) make responses more focused and deterministic. Higher values (e.g., 0.9) make them more creative and diverse."
            )
        ]
    ).send()
    # Store the initial temperature from the settings UI
    if settings_ui and "temperature" in settings_ui:
        cl.user_session.set("temperature", float(settings_ui["temperature"]))
        print(f"Initial temperature set from UI: {settings_ui['temperature']}")


async def handle_file_uploads(message_content_placeholder: str = "Processing your document for analysis..."):
    """
    Prompts the user to upload a file for discussion during the current session.

    If a file is uploaded, this function attempts to process it using
    `agent.process_uploaded_file` (which extracts temporary contextual content)
    and stores the file's name, path, and extracted content string in the user session.
    It provides feedback to the user regarding the upload and processing status.

    Args:
        message_content_placeholder (str): A message displayed while the file is being processed.
    """
    files = None
    # Loop to ask for a file until one is provided or the user implicitly cancels (by timeout or sending a message)
    # Note: cl.AskFileMessage is a blocking call in terms of user interaction.
    while files is None: # This loop structure might be re-evaluated based on desired UX for re-prompting.
                         # For on_chat_start, one attempt is usually sufficient.
        files = await cl.AskFileMessage(
            content="Optional: Upload a file to discuss (e.g., PDF, DOCX, TXT, CSV, XLSX, MD, RData, RDS, SAV).",
            accept=ACCEPTED_FILE_TYPES,
            max_size_mb=MAX_FILE_SIZE_MB,
            timeout=300,  # 5-minute timeout for user to upload
            raise_on_timeout=False, # If false, returns None on timeout instead of raising an error
            author="System" # System message styling
        ).send()

        if files: # If user uploaded one or more files
            uploaded_file = files[0] # Process the first uploaded file
            file_path = uploaded_file.path
            file_name = uploaded_file.name

            cl.user_session.set("uploaded_file_name", file_name)
            cl.user_session.set("uploaded_file_path", file_path) # The agent might use the path for some loaders

            # Inform user that processing has started
            processing_feedback_msg = cl.Message(content=message_content_placeholder, author="System")
            await processing_feedback_msg.send()

            from agent import process_uploaded_file # Local import for clarity, agent.py has file processors

            try:
                # Call the file processing function (potentially I/O bound) asynchronously
                file_content_str = await cl.make_async(process_uploaded_file)(file_path)

                if file_content_str and not file_content_str.lower().startswith("error:"):
                    cl.user_session.set("uploaded_file_content_str", file_content_str)
                    success_msg = f"Successfully processed `{file_name}`. You can now ask questions about its content."
                    await cl.Message(content=success_msg, author="System").send()
                else:
                    # Handle cases where processing function returns an error string or None
                    error_msg_detail = file_content_str if file_content_str else f"Could not extract readable content from `{file_name}`."
                    await cl.ErrorMessage(content=error_msg_detail, author="System").send()
                    # Clear session variables related to the failed upload
                    cl.user_session.set("uploaded_file_name", None)
                    cl.user_session.set("uploaded_file_path", None)
                    cl.user_session.set("uploaded_file_content_str", None)

            except Exception as e:
                # Catch any unexpected errors during file processing
                await cl.ErrorMessage(content=f"A critical error occurred while processing `{file_name}`: {str(e)}", author="System").send()
                print(f"Error in handle_file_uploads for {file_name}: {e}")
                cl.user_session.set("uploaded_file_name", None)
                cl.user_session.set("uploaded_file_path", None)
                cl.user_session.set("uploaded_file_content_str", None)

            break # Exit the loop after attempting to process the first uploaded file
        else:
            # No file was uploaded (e.g., user timed out or sent a message instead)
            await cl.Message(content="No file uploaded. You can continue chatting or try uploading again if needed (the bot may not explicitly ask again).", author="System").send()
            break # Exit the loop


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """
    Handles updates to chat settings when the user changes them in the UI.
    Currently, this is primarily used for updating the LLM's temperature.

    Args:
        settings (dict): A dictionary containing the updated settings,
                         where keys are the IDs of the input widgets.
    """
    print(f"Chat settings updated by user: {settings}")
    updated_temperature_str = settings.get("temperature") # ID of the slider widget

    if updated_temperature_str is not None:
        try:
            updated_temperature = float(updated_temperature_str)
            cl.user_session.set("temperature", updated_temperature) # Store in session

            agent = cl.user_session.get("agent") # type: ESIConversationAgent | None
            if agent:
                agent.update_temperature(updated_temperature) # Update the agent's LLM
                await cl.Message(
                    content=f"LLM Temperature has been updated to {updated_temperature}.",
                    author="System" # System message styling
                ).send()
            else:
                # This case should ideally not happen if agent is set in on_chat_start
                await cl.ErrorMessage(
                    content=f"Temperature setting received ({updated_temperature}), but the AI agent was not found in the session. Please try restarting the chat.",
                    author="System"
                ).send()
                print("Warning: Agent not found in session during on_settings_update.")
        except ValueError:
            await cl.ErrorMessage(
                content=f"Invalid temperature value received: {updated_temperature_str}. Temperature not updated.",
                author="System"
            ).send()
            print(f"Error: Could not convert temperature '{updated_temperature_str}' to float.")


async def display_welcome_message():
    """
    Displays the initial welcome message for the ESI Scholarly Instructor chatbot.
    Uses Markdown for formatting.
    """
    welcome_text = (
        "## Welcome to ESI: ESI Scholarly Instructor!\n\n"
        "I am here to assist you with your scholarly research, learning, and technical tasks. "
        "Feel free to ask questions, request explanations, or upload documents for discussion.\n\n"
        "**You can:**\n"
        "- Ask general knowledge questions.\n"
        "- Request information from my specialized knowledge base (if populated).\n"
        "- Upload files (PDF, DOCX, TXT, CSV, etc.) for contextual Q&A.\n"
        "- Ask for code execution or explanations (Python).\n"
        "- Adjust my response style using the 'LLM Temperature' slider in the settings (cog icon).\n\n"
        "How can I help you today?"
    )
    await cl.Message(
        content=welcome_text,
        author="ESI Assistant" # Consistent author name for the bot
    ).send()
