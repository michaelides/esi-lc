# Main application file for ESI Scholarly Instructor
"""
This is the main entry point for the ESI Scholarly Instructor Chainlit application.
It defines the chat lifecycle events (`on_chat_start`, `on_message`) and
coordinates interactions between the user interface (Chainlit) and the
backend agent logic.

Key functionalities:
- Loads environment variables.
- Initializes the conversational agent (`ESIConversationAgent`) at the start of a chat.
- Sets up UI elements like temperature sliders and file upload prompts.
- Handles incoming user messages, passes them to the agent (with any file context).
- Displays the agent's responses and manages UI updates.
"""

import chainlit as cl
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Import agent and UI components
from agent import ESIConversationAgent, load_system_prompt # process_uploaded_file is used within ui.py now
from ui import (
    setup_chat_settings,
    handle_file_uploads,
    display_welcome_message,
    # ACCEPTED_FILE_TYPES and MAX_FILE_SIZE_MB are used by ui.py internally
)


@cl.on_chat_start
async def start_chat():
    """
    Handles the start of a new chat session.

    This asynchronous function is called by Chainlit when a user initiates a new chat.
    It performs the following actions:
    1. Initializes the `ESIConversationAgent` with a default temperature and stores it in the user session.
       Handles potential errors during agent initialization (e.g., missing API keys).
    2. Sets up chat-specific settings UI, like the LLM temperature slider.
    3. Displays a welcome message to the user.
    4. Prompts the user for an optional file upload to discuss during the session.
    5. Loads and stores the system prompt in the user session (primarily for debugging/display purposes).
    """
    # Initialize and store the agent in the user session
    try:
        # Default temperature; this can be immediately updated by the user via chat settings.
        initial_temp = 0.7
        agent = ESIConversationAgent(temperature=initial_temp)
        cl.user_session.set("agent", agent)
        print("ESIConversationAgent initialized and set in user session.")
    except ValueError as e:
        # Inform user if agent initialization fails (e.g., API key issues)
        await cl.ErrorMessage(
            f"Critical Error: Could not initialize the AI Agent: {e}. "
            "Please ensure your GOOGLE_API_KEY is correctly set in the .env file and is valid. "
            "The application may not function correctly."
        ).send()
        print(f"Fatal: ESIConversationAgent initialization failed: {e}")
        return # Stop further execution in this chat if agent fails

    # Set up chat settings UI (e.g., temperature slider)
    await setup_chat_settings()
    print("Chat settings UI configured.")

    # Display a welcome message
    await display_welcome_message()
    print("Welcome message displayed.")

    # Ask for file upload (optional)
    # This function handles its own user interaction loop and messaging.
    await handle_file_uploads(message_content_placeholder="Analyzing your document for this session...")
    print("File upload process initiated (optional for user).")

    # Store system prompt for potential display or debugging (agent loads it internally too)
    cl.user_session.set("system_prompt", load_system_prompt())
    print("System prompt loaded into session (for potential debug/display).")


@cl.on_message
async def main(message: cl.Message):
    """
    Handles incoming user messages during an active chat session.

    This asynchronous function is called by Chainlit whenever the user sends a message.
    It performs the following actions:
    1. Retrieves the `ESIConversationAgent` instance from the user session.
    2. Retrieves any active uploaded file context (name and content string) from the session.
    3. Sends a "thinking..." message to the UI to indicate processing.
    4. Calls the agent's `arun` method with the user's input and any file context.
    5. Sends the agent's response back to the user.
    6. Handles potential errors during agent processing.
    7. Removes the "thinking..." message.

    Args:
        message (cl.Message): The Chainlit Message object containing the user's input.
    """
    agent = cl.user_session.get("agent") # type: ESIConversationAgent
    if not agent:
        await cl.ErrorMessage("Error: The AI agent is not available. Please try restarting the chat session.").send()
        print("Error: Agent not found in user session during on_message.")
        return

    # Retrieve context from any file uploaded earlier in the session
    uploaded_file_content_str = cl.user_session.get("uploaded_file_content_str")
    uploaded_file_name = cl.user_session.get("uploaded_file_name") # For potential logging or display

    input_text = message.content

    # UX: Indicate that the agent is processing the request
    thinking_msg = cl.Message(content="ESI is thinking...", author="System", parent_id=message.id)
    await thinking_msg.send()

    # Run the agent with the user's input and any available file context
    try:
        if uploaded_file_name and uploaded_file_content_str:
            print(f"Calling agent with input and context from file: {uploaded_file_name}")
            response_content = await agent.arun(
                user_input=input_text,
                uploaded_file_content=uploaded_file_content_str
            )
            # Optional: Clear file context after one use if desired
            # cl.user_session.set("uploaded_file_content_str", None)
            # cl.user_session.set("uploaded_file_name", None)
            # await cl.Message(
            #     content=f"(Context from `{uploaded_file_name}` was used. It's cleared for the next message unless uploaded again.)",
            #     author="System",
            #     parent_id=thinking_msg.id # To thread it correctly
            # ).send()
        else:
            print("Calling agent with input, no active file context.")
            response_content = await agent.arun(user_input=input_text)

        # Send the agent's actual response
        await cl.Message(content=response_content, author="ESI Assistant", parent_id=message.id).send()

    except Exception as e:
        error_message = f"Sorry, an unexpected error occurred while processing your request: {str(e)}"
        await cl.ErrorMessage(content=error_message, parent_id=message.id).send()
        print(f"Error during agent.arun in on_message: {e}") # Log detailed error to console
    finally:
        # Remove the "thinking..." message regardless of success or failure
        await thinking_msg.remove()


# Note: The @cl.on_settings_update decorator is defined in ui.py.
# Chainlit automatically discovers and registers handlers decorated with
# its event hooks (like @cl.on_chat_start, @cl.on_message, @cl.on_settings_update)
# from any imported Python module in the Chainlit project.

if __name__ == "__main__":
    # This block is primarily for informational purposes when the script is run directly,
    # which is not the typical way to start a Chainlit application.
    # Chainlit apps are run using the CLI: `chainlit run app.py -w`

    print("--- ESI Scholarly Instructor ---")
    print("This is the main application file for the ESI Scholarly Instructor chatbot.")
    print("To run this application with Chainlit, use the following command in your terminal:")
    print("  chainlit run app.py -w\n")

    print("Performing basic environment checks (run `python app.py` to see this):")
    try:
        load_dotenv() # Ensure .env is loaded for these checks

        google_key = os.getenv("GOOGLE_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")

        if not google_key:
            print("WARNING: GOOGLE_API_KEY not found in .env or environment. The LLM and RAG functionalities will not work.")
        else:
            print("GOOGLE_API_KEY found.")

        if not tavily_key:
            print("WARNING: TAVILY_API_KEY not found in .env or environment. Tavily Search tool will be unavailable.")
        else:
            print("TAVILY_API_KEY found.")

        # You could add a basic test of ESIConversationAgent initialization here if desired,
        # but it might be slow or require network access.
        # print("\nAttempting a quick test of agent system prompt loading...")
        # _ = load_system_prompt()
        # print("System prompt loading function called successfully.")

    except Exception as e:
        print(f"Error during basic setup check in `if __name__ == '__main__':`: {e}")
