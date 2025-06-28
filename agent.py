# Agent logic, tool definitions, and LLM integration for ESI Scholarly Instructor
"""
This module defines the core agent capabilities for the ESI Scholarly Instructor.
It includes:
- Initialization of the Language Model (LLM).
- Definition and setup of tools (Tavily Search, Python REPL, RAG).
- The main ESIConversationAgent class responsible for orchestrating LLM calls, tool usage, and memory.
- Helper functions for loading system prompts, initializing embeddings, and processing uploaded files for temporary context.
"""

import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage # Used for typing and memory interaction
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# --- Constants ---
CHROMA_DB_PATH = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db_esi")
"""Path to the persisted ChromaDB directory for RAG."""

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "esi_rag_collection")
"""Name of the collection within ChromaDB for RAG."""

SYSTEM_PROMPT_FILE = "esi_agent_instruction.md"
"""Filename for the system prompt instructions."""

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
"""API key for Google Generative AI services (Gemini LLM, Embeddings)."""

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
"""API key for Tavily Search service."""


# --- System Prompt Loader ---
def load_system_prompt() -> str:
    """
    Loads the system prompt from the specified file.
    Falls back to a default prompt if the file is not found.

    Returns:
        str: The system prompt content.
    """
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: System prompt file '{SYSTEM_PROMPT_FILE}' not found. Using default.")
        return "You are a helpful AI assistant. Please be scholarly and provide detailed explanations."


# --- LLM Initialization ---
def get_llm(temperature: float = 0.7) -> Optional[ChatGoogleGenerativeAI]:
    """
    Initializes and returns the Gemini LLM (ChatGoogleGenerativeAI).

    Args:
        temperature (float): The temperature setting for the LLM.

    Returns:
        Optional[ChatGoogleGenerativeAI]: The initialized LLM instance, or None if initialization fails.
    """
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables. LLM cannot be initialized.")
        return None
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Ensure this is the desired and available model
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True, # Important for some agent types
        )
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None


# --- Embeddings Initialization ---
def get_rag_embeddings() -> Optional[GoogleGenerativeAIEmbeddings]:
    """
    Initializes and returns Google Generative AI Embeddings for RAG.

    Returns:
        Optional[GoogleGenerativeAIEmbeddings]: The initialized embeddings instance, or None if initialization fails.
    """
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY not found in environment variables. RAG embeddings cannot be initialized.")
        return None
    try:
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Error initializing RAG embeddings: {e}")
        return None


# --- Tool Definitions ---
def get_tools(
    llm: Optional[ChatGoogleGenerativeAI],
    rag_db_path: str,
    embeddings_for_rag: Optional[GoogleGenerativeAIEmbeddings]
) -> List[Tool]:
    """
    Initializes and returns a list of tools available to the agent.

    Args:
        llm: The language model instance, required for tools like RetrievalQA.
        rag_db_path (str): Path to the ChromaDB persistence directory for the RAG tool.
        embeddings_for_rag: Embeddings instance for the RAG tool.

    Returns:
        List[Tool]: A list of initialized Tool objects.
    """
    tools = []

    # Tavily Search Tool
    if TAVILY_API_KEY:
        tools.append(TavilySearchResults(max_results=3, name="TavilySearch"))
    else:
        print("Warning: TAVILY_API_KEY not found. Tavily Search tool will not be available.")

    # Python REPL Tool
    tools.append(PythonREPLTool())

    # RAG (Scholarly Knowledge Base) Tool Setup
    if embeddings_for_rag and llm:
        if os.path.exists(rag_db_path):
            try:
                vectorstore = Chroma(
                    persist_directory=rag_db_path,
                    embedding_function=embeddings_for_rag,
                    collection_name=CHROMA_COLLECTION_NAME
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff", # Suitable for combining a few documents
                    retriever=retriever,
                    return_source_documents=True, # Useful for citations
                    verbose=True # Logs chain activity, good for debugging
                )

                # Define a synchronous wrapper for the RAG chain for the Tool's func
                def run_rag_chain_sync(query: str) -> Dict[str, Any]:
                    return rag_chain.invoke({"query": query})

                # If async execution is preferred and rag_chain supports ainvoke:
                # async def run_rag_chain_async(query: str) -> Dict[str, Any]:
                #    return await rag_chain.ainvoke({"query": query})

                rag_tool = Tool(
                    name="ScholarlyKnowledgeBase",
                    func=run_rag_chain_sync, # Use the synchronous wrapper
                    # coroutine=run_rag_chain_async, # Uncomment if async version is implemented
                    description=(
                        "Use this tool to answer questions by searching a specialized knowledge base "
                        "of scholarly documents. Input should be a specific question related to the "
                        "content of these documents. This tool returns a detailed answer and source documents."
                    ),
                )
                tools.append(rag_tool)
                print(f"RAG Tool (ScholarlyKnowledgeBase) initialized successfully from '{rag_db_path}'.")
            except Exception as e:
                print(f"Error initializing RAG tool from ChromaDB at '{rag_db_path}': {e}. RAG tool will not be available.")
        else:
            print(f"Warning: RAG database path '{rag_db_path}' does not exist. RAG tool will not be available. Run ingest.py first.")
    else:
        if not embeddings_for_rag:
            print("Warning: Embeddings for RAG not available. RAG tool cannot be initialized.")
        if not llm:
            print("Warning: LLM not available. RAG tool (QA part) cannot be initialized.")

    # Placeholder for future tools:
    # tools.append(SemanticScholarTool())
    # tools.append(WebScraperTool())
    # tools.append(PDFScraperTool())
    return tools


# --- Agent Class ---
class ESIConversationAgent:
    """
    The main conversational agent for ESI Scholarly Instructor.
    Manages the LLM, tools, memory, and execution flow.
    """
    def __init__(self, temperature: float = 0.7, window_size: int = 5):
        """
        Initializes the ESIConversationAgent.

        Args:
            temperature (float): Initial temperature for the LLM.
            window_size (int): The number of past messages to keep in conversational memory.

        Raises:
            ValueError: If the LLM cannot be initialized.
        """
        self.llm = get_llm(temperature)
        if not self.llm:
            raise ValueError("LLM could not be initialized. Check API key and ensure the model name is correct.")

        self.rag_embeddings = get_rag_embeddings()
        self.system_prompt_content = load_system_prompt()

        self.tools = get_tools(
            llm=self.llm,
            rag_db_path=CHROMA_DB_PATH,
            embeddings_for_rag=self.rag_embeddings
        )

        # Define the prompt structure for the agent
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_content),
                MessagesPlaceholder(variable_name="chat_history"), # For conversational memory
                ("human", "{input}"), # User's input
                MessagesPlaceholder(variable_name="agent_scratchpad"), # For agent's intermediate steps
            ]
        )

        # Create the agent using the LLM, tools, and prompt
        # create_openai_tools_agent is a common constructor for agents that can use tools (function calling)
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)

        # Initialize conversational memory
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history",
            return_messages=True,
            output_key="output" # Ensure this matches the agent's output key for memory population
        )

        # Create the agent executor, which runs the agent and tools
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True, # Logs agent activity, useful for debugging
            memory=self.memory,
            handle_parsing_errors="Check your output and make sure it conforms!", # Custom message or function for parsing errors
        )

    async def arun(self, user_input: str, uploaded_file_content: Optional[str] = None) -> str:
        """
        Asynchronously runs the agent with the given user input and optional file context.

        Args:
            user_input (str): The user's message or query.
            uploaded_file_content (Optional[str]): Textual content extracted from a user-uploaded file.

        Returns:
            str: The agent's response.
        """
        current_input = user_input
        if uploaded_file_content:
            # Append uploaded file content to the user input for context
            current_input = (
                f"{user_input}\n\n--- START OF UPLOADED FILE CONTEXT ---\n"
                f"{uploaded_file_content}\n"
                f"--- END OF UPLOADED FILE CONTEXT ---"
            )

        try:
            response = await self.agent_executor.ainvoke({"input": current_input})
            output = response.get("output", "Sorry, I couldn't find an answer for that.")

            # Post-processing RAG output for clarity, if RAG was used
            # This requires inspecting intermediate steps or modifying the RAG tool's output format
            if "intermediate_steps" in response:
                for action, observation in response["intermediate_steps"]:
                    if action.tool == "ScholarlyKnowledgeBase" and isinstance(observation, dict):
                        # The observation from RAG tool is the dict from RetrievalQA
                        source_docs = observation.get("source_documents")
                        if source_docs:
                            source_filenames = list(set([doc.metadata.get("filename", "Unknown Source") for doc in source_docs]))
                            if source_filenames:
                                output += f"\n\n(Sources from Knowledge Base: {', '.join(source_filenames)})"
                        break # Assume only one RAG call per arun, or refine this logic

            return output
        except Exception as e:
            print(f"Agent execution error: {e}")
            # Potentially log more details about the error (e.g., traceback)
            return f"Sorry, an error occurred while processing your request: {str(e)}"

    def update_temperature(self, temperature: float):
        """
        Updates the LLM's temperature and reinitializes agent components that depend on the LLM.

        Args:
            temperature (float): The new temperature value.
        """
        self.llm = get_llm(temperature)
        if not self.llm:
            print("Error: Failed to update LLM temperature. Using previous LLM instance.")
            return # Keep using the old LLM instance if update fails

        # Re-initialize tools, agent, and executor with the new LLM instance
        self.tools = get_tools(
            llm=self.llm,
            rag_db_path=CHROMA_DB_PATH,
            embeddings_for_rag=self.rag_embeddings
        )
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors="Check your output and make sure it conforms!",
        )
        print(f"Agent LLM temperature updated to: {temperature}")

    def get_chat_history_messages(self) -> List[Dict[str, str]]:
        """
        Retrieves the chat history from memory in a simple list of dictionaries format,
        suitable for UI display or other uses.

        Returns:
            List[Dict[str, str]]: A list of messages, each with "role" and "content".
        """
        messages = []
        for msg_obj in self.memory.chat_memory.messages: # Access messages from ConversationBufferWindowMemory
            role = "unknown"
            if isinstance(msg_obj, HumanMessage): role = "user"
            elif isinstance(msg_obj, AIMessage): role = "assistant"
            elif isinstance(msg_obj, SystemMessage): role = "system" # Should ideally not be in window memory
            messages.append({"role": role, "content": msg_obj.content})
        return messages


# --- Functions for handling uploaded files (temporary processing) ---
def process_uploaded_file(file_path: str) -> Optional[str]:
    """
    Reads content from a user-uploaded file for temporary contextual use by the agent.
    This is distinct from RAG ingestion. It extracts a summary or initial part of the content.
    For complex file types (PDF, DOCX, XLSX), it uses specific loaders.
    For simpler types (TXT, MD, CSV), it reads directly.
    For unsupported or binary types (RData, SAV), it acknowledges the file type.

    Args:
        file_path (str): The local path to the uploaded file.

    Returns:
        Optional[str]: A string containing extracted/summarized content or an acknowledgement message,
                       or None/error message if processing fails.
    """
    content: Optional[str] = None
    filename = os.path.basename(file_path)
    file_extension = os.path.splitext(filename)[1].lower()

    print(f"Processing uploaded file for temporary context: {filename} (Extension: {file_extension})")

    try:
        if file_extension == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader # Import locally
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text_content = "\n".join([page.page_content for page in pages[:3]]) # Content from first 3 pages
            content = f"The user has uploaded a PDF document named '{filename}'.\nFirst few pages content (up to 2000 chars):\n{text_content[:2000]}"
            if len(text_content) > 2000: content += "\n...(content truncated)"
            print(f"PDF '{filename}' content (first 3 pages, up to 2000 chars) loaded for temporary context.")

        elif file_extension == ".docx":
            from langchain_community.document_loaders import Docx2txtLoader # Import locally
            loader = Docx2txtLoader(file_path)
            data = loader.load()
            full_text = data[0].page_content if data else ""
            content = f"The user has uploaded a Word document (DOCX) named '{filename}'.\nContent (up to 2000 chars):\n{full_text[:2000]}"
            if len(full_text) > 2000: content += "\n...(content truncated)"
            print(f"DOCX '{filename}' content (up to 2000 chars) loaded for temporary context.")

        elif file_extension in [".txt", ".md", ".csv"]: # Treat CSV as text for this context
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f: # errors='ignore' for robustness
                raw_content = f.read()
            content = f"Content from text-based file '{filename}':\n{raw_content[:3000]}" # Limit context size
            if len(raw_content) > 3000: content += "\n...(content truncated)"
            print(f"Text content from {filename} (up to 3000 chars) loaded for temporary context.")

        elif file_extension == ".xlsx":
            import pandas as pd # Import pandas only when needed for this file type
            excel_file = pd.ExcelFile(file_path)
            content_parts = [f"Excel file '{filename}' contains sheets: {', '.join(excel_file.sheet_names)}."]
            for i, sheet_name in enumerate(excel_file.sheet_names):
                if i >= 2: # Limit to summarizing first 2 sheets
                    content_parts.append(f"... and {len(excel_file.sheet_names) - i} more sheet(s).")
                    break
                df = excel_file.parse(sheet_name)
                content_parts.append(f"\nSheet: '{sheet_name}' (first 5 rows, up to 5 columns):\n{df.iloc[:5, :5].to_string()}")
            content = "\n".join(content_parts)[:3000] # Overall limit for the summary
            if len("\n".join(content_parts)) > 3000: content += "\n...(content truncated)"
            print(f"XLSX '{filename}' summary loaded for temporary context.")

        elif file_extension in [".rdata", ".rds"]:
            content = (f"The user has uploaded an R data file ('{filename}', type: {file_extension}). "
                       "This file likely contains R objects or data frames. To analyze its content, "
                       "specific R code execution would be needed using the PythonREPLTool and appropriate R libraries (e.g., via rpy2). "
                       "You can ask the user about the expected structure or variables within it, or ask them to provide R code to run.")
            print(f"R data file '{filename}' acknowledged for temporary context. Advise user on code execution for analysis.")

        elif file_extension == ".sav": # SPSS file
            content = (f"The user has uploaded an SPSS data file ('{filename}', type: {file_extension}). "
                       "This file contains statistical data. To analyze its content, specific code execution "
                       "would be needed using the PythonREPLTool and Python libraries that can read SAV files (e.g., pyreadstat). "
                       "You can ask the user about variables they are interested in or analyses they want to perform.")
            print(f"SPSS file '{filename}' acknowledged for temporary context. Advise user on code execution for analysis.")

        else:
            print(f"Unsupported file type for direct temporary context extraction: {filename} ({file_extension})")
            content = (f"The file '{filename}' (type: {file_extension}) has been uploaded by the user. "
                       "Its content cannot be automatically displayed. You can ask the user to describe its contents or "
                       "provide specific text snippets if relevant to their query.")

    except Exception as e:
        print(f"Error processing uploaded file {filename} for temporary context: {e}")
        content = f"Error reading or processing file '{filename}' for temporary context. Please ensure it is a valid {file_extension} file. Error details: {str(e)}"
    return content


if __name__ == '__main__':
    # This block is for basic testing when running agent.py directly.
    print("Agent.py loaded. Contains agent logic and tool definitions.")
    print(f"System prompt (first 100 chars): {load_system_prompt()[:100]}...")

    llm_instance = get_llm()
    if llm_instance:
        print("LLM initialized successfully.")
    else:
        print("LLM initialization FAILED. Check GOOGLE_API_KEY.")

    # Test agent initialization if essential API keys are present
    if GOOGLE_API_KEY and TAVILY_API_KEY: # RAG tool might not load if DB is not present, but agent should still try
        print("\nAttempting to initialize ESIConversationAgent...")
        try:
            agent_instance = ESIConversationAgent(temperature=0.5)
            print("ESIConversationAgent initialized successfully.")
            print(f"Loaded Agent tools: {[tool.name for tool in agent_instance.tools]}")
            if any(tool.name == "ScholarlyKnowledgeBase" for tool in agent_instance.tools):
                print("RAG Tool (ScholarlyKnowledgeBase) is configured and loaded.")
            else:
                print("RAG Tool (ScholarlyKnowledgeBase) is NOT loaded. This might be due to missing DB path, embeddings, or LLM issues during tool setup.")
        except Exception as e:
            print(f"Error initializing ESIConversationAgent: {e}")
    else:
        print("\nSkipping ESIConversationAgent initialization test (GOOGLE_API_KEY or TAVILY_API_KEY missing).")

    # Example of testing process_uploaded_file (requires creating dummy files)
    # print("\nTesting file processing for temporary context:")
    # Create a dummy file named 'dummy.txt' with some text for testing
    # with open("dummy.txt", "w") as f:
    #     f.write("This is a test line from dummy.txt.\nSecond line for context.")
    # print(f"Processing dummy.txt: {process_uploaded_file('dummy.txt')}")
    # if os.path.exists("dummy.txt"): os.remove("dummy.txt")

    # Create a dummy PDF (e.g. using reportlab or just a placeholder name if loader handles missing gracefully)
    # print(f"Processing dummy.pdf: {process_uploaded_file('dummy.pdf')}") # Will likely show error if no real PDF
