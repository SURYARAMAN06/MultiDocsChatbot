# MultiDocsChatbot
A Streamlit-based chatbot application that enables users to upload multiple document types (PDF, DOCX, TXT) and interact with them via natural language queries. The chatbot leverages the LangChain framework for conversation management, FAISS for efficient document search, and the ChatGroq model for generating intelligent responses.

Table of Contents
Features
Tech Stack
Installation
Usage
How It Works
Environment Variables
Contributing
License
Features
Document Upload: Upload multiple files such as PDFs, DOCX, and TXT.
Natural Language Querying: Interact with documents using natural language queries.
Conversational AI: Utilizes the ChatGroq model to provide intelligent and relevant responses.
FAISS Vector Search: Efficiently retrieves relevant document sections based on queries.
Session History: Maintains chat history throughout the conversation.
Tech Stack
Framework: Streamlit
LLM: LangChain & ChatGroq
Embeddings: HuggingFace Sentence Transformers
Vector Database: FAISS
Document Loaders: PyPDFLoader, Docx2txtLoader, TextLoader
Installation
Follow these steps to install and run the project locally.

Prerequisites
Python 3.8 or above
Virtual environment (optional but recommended)

Usage
Upload Files: Use the sidebar to upload multiple document files (PDF, DOCX, TXT).
Ask Questions: Type in natural language queries based on the content of the uploaded documents.
View Responses: The chatbot will provide answers based on the uploaded document's content.
Chat History: The chat history will be displayed, allowing seamless conversation tracking.
How It Works
Document Upload and Processing:

Users can upload documents (PDF, DOCX, TXT).
The text from the uploaded files is extracted using appropriate loaders (PyPDFLoader, Docx2txtLoader, TextLoader).
Text Splitting:

The extracted text is split into chunks using CharacterTextSplitter for efficient processing.
Embedding Creation:

The document chunks are converted into vector embeddings using the HuggingFace all-MiniLM-L6-v2 model.
Vector Search:

FAISS is used as the vector store to perform document similarity search and retrieve relevant chunks of text.
Conversational Chain:

The chatbot generates responses using the ChatGroq model. Conversation history is stored and managed with ConversationBufferMemory to ensure context is maintained.


Contributing
Contributions are welcome! To contribute:

Fork the repository.

License
This project is licensed under the MIT License. See the LICENSE file for details.
