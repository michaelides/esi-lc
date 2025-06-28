# Main application file for ESI Scholarly Instructor
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to get API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Load the system prompt
DEFAULT_SYSTEM_PROMPT_FILE = "esi_agent_instruction.md"
try:
    with open(DEFAULT_SYSTEM_PROMPT_FILE, "r") as f:
        ESI_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    ESI_SYSTEM_PROMPT = "You are a helpful AI assistant. Please act as ESI Scholarly Instructor."
    st.warning(f"Warning: System prompt file '{DEFAULT_SYSTEM_PROMPT_FILE}' not found. Using a default prompt.")


st.title("ESI: ESI Scholarly Instructor")

# API Key Management in Sidebar
st.sidebar.subheader("API Configuration")
if GEMINI_API_KEY:
    st.sidebar.success("Gemini API Key loaded from .env")
    gemini_api_key_input = GEMINI_API_KEY
else:
    gemini_api_key_input = st.sidebar.text_input("Enter Gemini API Key", type="password", help="You can set this in your .env file as GEMINI_API_KEY")

if TAVILY_API_KEY:
    st.sidebar.success("Tavily API Key loaded from .env")
    tavily_api_key_input = TAVILY_API_KEY
else:
    tavily_api_key_input = st.sidebar.text_input("Enter Tavily API Key", type="password", help="You can set this in your .env file as TAVILY_API_KEY")


# LLM settings
st.sidebar.subheader("LLM Settings")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.05, key="temperature_slider")

# Initialize LLM (basic)
# We will properly initialize the agent later, this is just to get the LLM object
llm = None
if gemini_api_key_input:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=gemini_api_key_input, temperature=temperature)
        st.sidebar.success("Gemini LLM Initialized")
    except Exception as e:
        st.sidebar.error(f"Error initializing Gemini LLM: {e}")
else:
    st.sidebar.warning("Gemini API Key not provided. LLM not initialized.")

# Initialize Tools
tools = []
if tavily_api_key_input:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key_input)
        tools.append(tavily_tool)
        st.sidebar.success("Tavily Search tool initialized.")
    except Exception as e:
        st.sidebar.error(f"Error initializing Tavily Search: {e}")
else:
    st.sidebar.warning("Tavily API Key not provided. Tavily Search tool not initialized.")

try:
    from langchain_community.tools import SemanticScholarQueryRun
    semantic_scholar_tool = SemanticScholarQueryRun()
    tools.append(semantic_scholar_tool)
    st.sidebar.success("Semantic Scholar tool initialized.")
except ImportError:
    st.sidebar.warning("Semantic Scholar tool could not be imported. Ensure 'arxiv' and 'requests' are installed if not already.") # Semantic Scholar tool has dependencies like arxiv
except Exception as e:
    st.sidebar.error(f"Error initializing Semantic Scholar tool: {e}")

# Website Scraping Tool
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests
from bs4 import BeautifulSoup

class ScrapeWebsiteInput(BaseModel):
    url: str = Field(description="The URL of the website to scrape")

class WebsiteScrapingTool(BaseTool):
    name = "scrape_website"
    description = "Useful for when you need to get the content from a specific website URL. Input should be a valid URL."
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, url: str):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status() # Raise an exception for HTTP errors
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            return text[:10000] # Return first 10k characters to avoid overly long outputs
        except requests.exceptions.RequestException as e:
            return f"Error during website scraping: {e}"
        except Exception as e:
            return f"An unexpected error occurred during scraping: {e}"

try:
    website_scraper_tool = WebsiteScrapingTool()
    tools.append(website_scraper_tool)
    st.sidebar.success("Website Scraping tool initialized.")
except Exception as e:
    st.sidebar.error(f"Error initializing Website Scraping tool: {e}")

# RAG with ChromaDB Setup
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings # Using a common open-source embedding model
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DB_PATH = "chroma_db"
# Using a sentence transformer model for embeddings.
# If GoogleGenerativeAIEmbeddings are preferred and work well, can switch.
# For now, using a local model avoids another API key for embeddings.
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_function)
    st.sidebar.success(f"ChromaDB vector store initialized/loaded from {CHROMA_DB_PATH}.")
except Exception as e:
    st.sidebar.error(f"Error initializing ChromaDB: {e}")
    vector_store = None

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Placeholder RAG Tools (to be refined)
class AddToRAGInput(BaseModel):
    text_content: str = Field(description="The text content to add to the RAG knowledge base.")
    document_id: str = Field(description="A unique ID for the document or text source.")

class AddToRAGTool(BaseTool):
    name = "add_to_rag"
    description = "Adds text content to the persistent RAG knowledge base. Use this when the user wants to remember a document for future interactions."
    args_schema: Type[BaseModel] = AddToRAGInput

    def _run(self, text_content: str, document_id: str):
        if not vector_store:
            return "Error: Vector store not initialized."
        try:
            docs = text_splitter.create_documents([text_content], metadatas=[{"source": document_id}])
            vector_store.add_documents(docs)
            vector_store.persist() # Persist changes
            return f"Content from '{document_id}' added to RAG."
        except Exception as e:
            return f"Error adding to RAG: {e}"

class QueryRAGInput(BaseModel):
    query: str = Field(description="The query to search for in the RAG knowledge base.")

class QueryRAGTool(BaseTool):
    name = "query_rag"
    description = "Queries the persistent RAG knowledge base for relevant information."
    args_schema: Type[BaseModel] = QueryRAGInput

    def _run(self, query: str):
        if not vector_store:
            return "Error: Vector store not initialized."
        try:
            results = vector_store.similarity_search(query, k=3)
            if results:
                return "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in results])
            return "No relevant information found in RAG."
        except Exception as e:
            return f"Error querying RAG: {e}"

if vector_store:
    tools.append(AddToRAGTool())
    tools.append(QueryRAGTool())
    st.sidebar.success("RAG tools (add/query) initialized.")
else:
    st.sidebar.warning("RAG tools not initialized due to ChromaDB error.")

# File Uploading and Processing
import io
from pypdf import PdfReader
import pandas as pd
import docx2txt
import pyreadr

# Initialize session state for uploaded files content and names
if 'uploaded_file_contents' not in st.session_state:
    st.session_state.uploaded_file_contents = {} # dict to store filename: content
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

def process_uploaded_file(uploaded_file_obj):
    """Processes a single uploaded file and returns its text content."""
    content = ""
    file_name = uploaded_file_obj.name
    try:
        if file_name.lower().endswith(".pdf"):
            pdf_reader = PdfReader(io.BytesIO(uploaded_file_obj.getvalue()))
            for page in pdf_reader.pages:
                content += page.extract_text() or ""
        elif file_name.lower().endswith((".txt", ".md")):
            content = uploaded_file_obj.getvalue().decode("utf-8")
        elif file_name.lower().endswith(".docx"):
            content = docx2txt.process(io.BytesIO(uploaded_file_obj.getvalue()))
        elif file_name.lower().endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(uploaded_file_obj.getvalue()))
                content = df.to_string()
            except Exception as e_csv:
                content = f"Error reading CSV {file_name}: {e_csv}"
        elif file_name.lower().endswith((".xlsx", ".xls")):
            try:
                df = pd.read_excel(io.BytesIO(uploaded_file_obj.getvalue()))
                content = df.to_string()
            except Exception as e_excel:
                content = f"Error reading Excel {file_name}: {e_excel}"
        elif file_name.lower().endswith(".rdata"):
            try:
                # pyreadr reads Rdata and returns a dictionary of dataframes
                result = pyreadr.read_r(io.BytesIO(uploaded_file_obj.getvalue()))
                content = "Content of Rdata file:\n"
                for key, df in result.items():
                    content += f"\nObject: {key}\n{df.to_string()}\n"
            except Exception as e_rdata:
                content = f"Error reading Rdata {file_name}: {e_rdata}"
        elif file_name.lower().endswith(".rds"):
            try:
                # pyreadr reads RDS and returns a single object (usually a dataframe)
                result = pyreadr.read_r(io.BytesIO(uploaded_file_obj.getvalue())) # Returns an OrderedDict
                # The actual object is typically the first (and often only) item in the OrderedDict
                if result:
                    first_key = list(result.keys())[0]
                    df = result[first_key]
                    content = f"Content of RDS file (object: {first_key}):\n{df.to_string()}"
                else:
                    content = "Could not extract data from RDS file."
            except Exception as e_rds:
                content = f"Error reading RDS {file_name}: {e_rds}"
        elif file_name.lower().endswith(".sav"):
            try:
                import pyreadstat
                df, meta = pyreadstat.read_sav(io.BytesIO(uploaded_file_obj.getvalue()))
                content = f"Content of SPSS file (.sav):\n{df.to_string()}\n\nMetadata:\n{str(meta)}"
            except ImportError:
                content = "Error: pyreadstat library is required to read .sav files. Please install it."
                st.error(content)
            except Exception as e_sav:
                content = f"Error reading SPSS (.sav) {file_name}: {e_sav}"
        else:
            content = f"File type '{file_name.split('.')[-1]}' not yet supported for direct Q&A."
            st.warning(content)
        return file_name, content
    except Exception as e:
        st.error(f"Error processing {file_name}: {e}")
        return file_name, f"Error processing file: {e}"

st.sidebar.subheader("Upload Files for Session Q&A")
uploaded_file_list = st.sidebar.file_uploader(
    "Upload (PDF, TXT, MD, DOCX, CSV, XLSX, Rdata, RDS, SAV)",
    accept_multiple_files=True,
    key="file_uploader_widget",
    help="These files are for Q&A in the current session and are not added to long-term RAG unless you explicitly ask."
)

if uploaded_file_list:
    new_files_processed = False
    for uploaded_file in uploaded_file_list:
        if uploaded_file.name not in st.session_state.uploaded_file_names:
            file_name, file_content = process_uploaded_file(uploaded_file)
            st.session_state.uploaded_file_contents[file_name] = file_content
            st.session_state.uploaded_file_names.append(file_name)
            st.sidebar.write(f"Processed: {file_name}")
            new_files_processed = True
    if new_files_processed:
        st.sidebar.success("Selected files processed for Q&A.")
        # st.experimental_rerun() # Might be useful if state needs immediate refresh for other components

# Tool for Q&A on uploaded files
class QueryUploadedFilesInput(BaseModel):
    query: str = Field(description="The query or question about the uploaded files.")
    file_name: str = Field(None, description="Optional: Specific file name to query from the uploaded files. If None, searches across all uploaded files.")

class QueryUploadedFilesTool(BaseTool):
    name = "query_uploaded_files"
    description = (
        "Answers questions about files uploaded by the user in the current session. "
        "Use this tool when the user asks something related to documents they have just uploaded. "
        "You can specify a file_name or search across all uploaded files."
    )
    args_schema: Type[BaseModel] = QueryUploadedFilesInput

    def _run(self, query: str, file_name: str = None):
        if not st.session_state.uploaded_file_contents:
            return "No files have been uploaded in this session yet."

        relevant_texts = []
        files_to_search = []

        if file_name:
            if file_name in st.session_state.uploaded_file_contents:
                files_to_search = [(file_name, st.session_state.uploaded_file_contents[file_name])]
            else:
                return f"File '{file_name}' not found in uploaded files. Available files: {', '.join(st.session_state.uploaded_file_names)}"
        else:
            files_to_search = st.session_state.uploaded_file_contents.items()

        if not files_to_search:
             return "No content available from specified files to query."

        # Simple text search for now. Could be enhanced with in-memory RAG for uploaded files.
        for fname, content in files_to_search:
            if query.lower() in content.lower(): # Basic keyword search
                 # For simplicity, returning a snippet or indicating relevance.
                 # A more advanced approach would involve splitting, embedding, and similarity search on the fly.
                snippet_index = content.lower().find(query.lower())
                snippet_start = max(0, snippet_index - 150)
                snippet_end = min(len(content), snippet_index + len(query) + 150)
                snippet = content[snippet_start:snippet_end]
                relevant_texts.append(f"From '{fname}': ...{snippet}...")

        if relevant_texts:
            return "\n".join(relevant_texts)
        else:
            return f"No direct answer found for '{query}' in the {file_name if file_name else 'uploaded files'}."

tools.append(QueryUploadedFilesTool())
st.sidebar.info("File Q&A tool for uploaded files is active.")

# Code Execution Tool
from langchain_experimental.tools import PythonREPLTool
try:
    python_repl_tool = PythonREPLTool()
    tools.append(python_repl_tool)
    st.sidebar.success("Python REPL tool initialized.")
except Exception as e:
    st.sidebar.error(f"Error initializing Python REPL tool: {e}")

# Agent Setup
from langchain.agents import AgentExecutor, create_react_agent # Using create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Set up chat history
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! I am ESI Scholarly Instructor. How can I assist you today?")

# Memory for the agent
agent_memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    output_key="output" # Ensure this matches agent's expectations if it differs
)

# System prompt template for the agent
# The ESI_SYSTEM_PROMPT is already loaded
# We need to make sure it's formatted correctly for create_react_agent if it needs specific placeholders like {tools} and {tool_names}
# For now, we assume ESI_SYSTEM_PROMPT is a general instruction.
# create_react_agent typically uses a prompt like hub.pull("hwchase17/react-chat")
# Let's try to build a compatible prompt or use a standard one and prepend our instructions.

# Attempting to use a ReAct style prompt.
# The base prompt includes placeholders for 'tools', 'tool_names', 'input', 'agent_scratchpad', and 'chat_history'.
# We will prepend our ESI_SYSTEM_PROMPT to the human message part of this structure.

# Using a known ReAct compatible prompt structure and injecting our system message
# This is a simplified version of prompts like hub.pull("hwchase17/react-chat")
# or more directly creating one.
# The key parts are: `input`, `chat_history`, `agent_scratchpad`, `tools`, `tool_names`

# Construct the ReAct prompt
# Based on common ReAct agent prompt structures.
# We will use the ESI_SYSTEM_PROMPT as a general directive.
# The agent needs specific input variables: input, agent_scratchpad, chat_history, tools, tool_names.

# Let's use the default ReAct prompt and prepend our system instructions.
# This might require some adjustment if the default prompt is too rigid.
# An alternative is to customize the prompt more deeply.

# For create_react_agent, the prompt needs to handle 'input', 'tools', 'tool_names', 'agent_scratchpad', 'chat_history'
# The ESI_SYSTEM_PROMPT provides the overall persona and guidelines.
# We'll prepend ESI_SYSTEM_PROMPT to the prompt that create_react_agent uses.

# Define the prompt for the ReAct agent
# The prompt should include the ESI_SYSTEM_PROMPT and placeholders for agent inputs.
# Base prompt structure for ReAct agent:
react_prompt_template = f"""{ESI_SYSTEM_PROMPT}

TOOLS:
------
You have access to the following tools:

{{tools}}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: The action to take. Should be one of [{{tool_names}}]
Action Input: The input to the action
Observation: The result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{{chat_history}}

New input: {{input}}
{{agent_scratchpad}}"""

prompt = ChatPromptTemplate.from_template(react_prompt_template)


agent = None
agent_executor = None

if llm and tools:
    try:
        # Create the ReAct agent
        agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

        # Create the Agent Executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=agent_memory,
            verbose=True, # Good for debugging
            handle_parsing_errors=True, # To gracefully handle LLM output parsing issues
            max_iterations=10, # Prevent overly long chains
            # early_stopping_method="generate" # Stop if LLM generates Final Answer
        )
        st.sidebar.success("Agent Executor created successfully.")
    except Exception as e:
        st.sidebar.error(f"Error creating agent executor: {e}")
else:
    st.sidebar.warning("LLM or tools not fully initialized. Agent Executor not created.")


# Display previous chat messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Placeholder for chat input
user_input = st.chat_input("Ask ESI...")

if user_input and agent_executor:
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response_content = ""

        try:
            # Construct the input for the agent_executor
            agent_input = {
                "input": user_input,
                "chat_history": agent_memory.chat_memory.messages # Pass the actual message objects
            }

            # Streamlit lacks direct support for streaming agent thoughts,
            # so we'll show "Thinking..." and then the final response.
            # For more complex streaming of thoughts, custom callbacks would be needed.
            with st.spinner("ESI is thinking..."):
                response = agent_executor.invoke(agent_input)

            if "output" in response:
                full_response_content = response["output"]
            else:
                full_response_content = "Sorry, I encountered an issue and couldn't generate a full response."
                st.error(f"Agent response missing 'output' key. Full response: {response}")

            response_placeholder.markdown(full_response_content)
            # The memory should be automatically updated by the AgentExecutor
            # msgs.add_ai_message(full_response_content) # This is handled by ConversationBufferMemory with StreamlitChatMessageHistory

        except Exception as e:
            error_message = f"Error during agent execution: {e}"
            response_placeholder.error(error_message)
            # msgs.add_ai_message(error_message) # Also handled by memory if errors are to be recorded

elif user_input and not agent_executor:
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write("Sorry, the AI agent is not properly configured. Please check API keys and settings.")
