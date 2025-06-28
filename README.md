# ESI: ESI Scholarly Instructor

**ESI (ESI Scholarly Instructor)** is an advanced chatbot application designed to assist users with scholarly research, learning, and technical tasks. It leverages Large Language Models (LLMs) like Google's Gemini, integrates with various tools for enhanced capabilities, and provides a user-friendly chat interface through Chainlit.

## Features

*   **Conversational AI:** Powered by Google Gemini LLMs (configurable, e.g., `gemini-1.5-flash`).
*   **Tool Integration:**
    *   **Tavily Search:** For general web searches and up-to-date information.
    *   **Python Code Executor:** Allows running Python code snippets for calculations, demonstrations, etc.
    *   **Retrieval Augmented Generation (RAG):**
        *   Answers questions based on a specialized knowledge base of ingested documents (PDF, TXT, MD).
        *   Uses ChromaDB for vector storage and Google Generative AI Embeddings.
        *   Includes a script (`ingest.py`) to build and update the RAG knowledge base.
*   **User File Uploads for Contextual Q&A:**
    *   Users can upload their own files (DOCX, MD, PDF, XLSX, RData, CSV, RDS, SAV, TXT) during a chat session.
    *   The content of these files is used as temporary context for answering questions about that specific file, without adding it to the persistent RAG store.
*   **Persistent Chat History:** (Handled by Chainlit's session management) Conversations are maintained during a user's session.
*   **Configurable LLM Settings:** Users can adjust LLM parameters like temperature via the UI to influence response style.
*   **Modular Design:** Code is organized into:
    *   `app.py`: Main Chainlit application, event handlers.
    *   `agent.py`: Core agent logic, LLM integration, tool definitions, memory.
    *   `ui.py`: Chainlit UI helper functions, settings, file upload handling.
    *   `ingest.py`: Script for RAG document ingestion.
    *   `esi_agent_instruction.md`: System prompt for the AI agent.
    *   `.env`: For managing API keys and environment variables.

## Setup and Installation

### Prerequisites

*   Python 3.9 or higher.
*   Access to Google Cloud Platform and a Google API Key with the "Generative Language API" (or "Vertex AI API" depending on model choice) enabled.
*   A Tavily API Key for web search capabilities.

### Installation Steps

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <repository_url>
    # cd esi-scholarly-instructor
    ```

2.  **Create a Virtual Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv .venv
    ```
    Activate the virtual environment:
    *   On macOS and Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    Install all required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    *   Copy the example environment file `.env.example` to a new file named `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Open the `.env` file and fill in your actual API keys:
        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
        TAVILY_API_KEY="YOUR_TAVILY_API_KEY_HERE"
        # SEMANTIC_SCHOLAR_API_KEY="YOUR_SEMANTIC_SCHOLAR_API_KEY_HERE" # If you implement this tool

        # Optional: You can customize ChromaDB settings if needed
        # CHROMA_COLLECTION_NAME="esi_rag_collection"
        # CHROMA_PERSIST_DIR="./chroma_db_esi"
        ```
    *   **Important:** Ensure the `.env` file is added to your `.gitignore` if you are using Git, to prevent committing your secret keys.

## Usage

### 1. Ingest Documents for RAG (Optional but Recommended)

To enable the "ScholarlyKnowledgeBase" tool, you need to ingest documents into the RAG system:

1.  Create a directory named `source_documents` in the root of the project:
    ```bash
    mkdir source_documents
    ```
2.  Place your scholarly articles, notes, or other relevant documents (PDF, TXT, MD files) into the `source_documents` directory (including subdirectories).
3.  Run the ingestion script:
    ```bash
    python ingest.py
    ```
    This will process the documents, create embeddings, and store them in a local ChromaDB instance (default path: `./chroma_db_esi`). You only need to run this when you want to add new documents or update the knowledge base.

### 2. Run the Chatbot Application

Once the setup is complete and you have (optionally) ingested documents, run the Chainlit application:

```bash
chainlit run app.py -w
```

*   `chainlit run app.py`: Starts the Chainlit server with your application.
*   `-w` (or `--watch`): Enables auto-reloading, so the app automatically restarts when you save changes to the Python files.

Open your web browser and navigate to the URL provided by Chainlit (usually `http://localhost:8000`).

### Interacting with the Chatbot

*   **Chat:** Type your questions or requests into the chat input.
*   **File Upload:** At the start of the chat, or if prompted, you can upload files for discussion. These files provide temporary context for your current conversation.
*   **Settings:** Click the cog icon (⚙️) in the chat interface to adjust LLM settings like temperature.

## Project Structure

```
.
├── .env                    # Stores API keys and environment variables (created from .env.example)
├── .env.example            # Example environment file
├── AGENTS.md               # (If any, for agent-specific instructions - not part of this initial setup)
├── README.md               # This file
├── agent.py                # Core agent logic, LLM, tools
├── app.py                  # Main Chainlit application file
├── esi_agent_instruction.md # System prompt for the LLM agent
├── ingest.py               # Script for RAG document ingestion
├── requirements.txt        # Python dependencies
├── source_documents/       # Directory for RAG source files (user-created)
├── ui.py                   # Chainlit UI helper functions
└── chroma_db_esi/          # Default directory for ChromaDB vector store (created by ingest.py)
```

## Future Enhancements (Placeholders)

The current application provides a solid foundation. Future enhancements could include:

*   **Semantic Scholar Tool:** Integration for direct academic paper searches.
*   **Web Scraper Tool:** For fetching and summarizing content from live web pages.
*   **PDF Scraper Tool (Advanced):** More sophisticated PDF content extraction beyond basic text (e.g., tables, images).
*   **More File Type Processors:** Full content extraction for RData, RDS, SAV files for temporary context.
*   **Asynchronous Tool Execution:** Making tools like RAG fully asynchronous to prevent any potential blocking.
*   **Citation Refinement:** More robust citation display from RAG results.
*   **Deployment:** Instructions for deploying the Chainlit application.

## Contributing

(Details on how to contribute to the project, if applicable.)

## License

(Specify the license for the project, e.g., MIT, Apache 2.0.)
```
