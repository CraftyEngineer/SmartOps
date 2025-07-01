# SmartOps: AI-Powered Document Parsing & Query Engine

SmartOps is an AI-powered data ingestion and RAG pipeline that automates the parsing of messy PDF manuals, HTML pages, and scanned documents. It extracts useful information (including tables), embeds it in a vector database, and allows natural language querying through a chat-like interface.

Built using LangChain, ChromaDB, HuggingFace, Streamlit, and OCR tools, it is ideal for knowledge-intensive tasks such as technical support, financial analysis, and enterprise automation.

## 🔍 Features

- **Web Scraping + Smart Downloading**: Automatically searches Google, downloads PDF/HTML pages, and handles embedded links and scanned images.
  
- **Multi-Stage Parsing Pipeline**: Uses OCR (Tesseract + OpenCV), pdfplumber, Camelot, and BeautifulSoup to extract both text and tables.

- **Vector Store with Chunked Embeddings**: Splits content into semantic chunks and stores it in ChromaDB with MiniLM embeddings.

- **Chat Over Documents**: Query your docs using an intuitive Streamlit UI powered by local or cloud-hosted LLMs (e.g., Groq, OpenAI, Claude).

- **Auto Context File Creation**: All document data is appended to a context file for downstream use and reproducibility.

## 💻 How to Run

1. Clone the repository:
```bash
git clone https://github.com/CraftyEngineer/SmartOps.git
cd SmartOps
```
2. Create a virtual environment (optional but recommended):
 ```bash
python -m venv .venv
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows
```
3. Install required packages:
```bash
pip install -r requirements.txt
```
4. Set up your API keys:
```bash
export GOOGLE_API_KEY=your_key
export GOOGLE_CX=your_cx
export GROQ_API_KEY=your_groq_key
export HF_TOKEN=your_hugging_face_token
```
5. Run the app:
```bash
streamlit run app.py
```

## 📦 Tech Stack

- **Core**:  
  Python · Streamlit · LangChain · ChromaDB  
- **AI/ML**:  
  HuggingFace Transformers · OCR (Tesseract + OpenCV)  
- **Processing Tools**:  
  pdfplumber · Camelot · BeautifulSoup · aiohttp · asyncio  

## 📘 Use Cases

- Parsing scanned PDF manuals into structured data  
- Searching across unstructured financial or technical docs  
- Building internal AI assistants for documentation or SOPs  
- Rapid prototyping of domain-specific RAG pipelines  
