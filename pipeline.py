# SQLite compatibility fix for Chroma
import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import certifi
import requests
import re
import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import camelot
import cv2
import numpy as np
from PIL import Image
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
import shutil
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import random
import streamlit as st
import concurrent.futures
import aiohttp
import asyncio
from aiohttp import ClientSession
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from chromadb import PersistentClient


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CX = os.getenv('GOOGLE_CX')

def google_search(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CX,
        "q": query,
    }
    response = requests.get(url, params=params)
    results = response.json().get("items", [])
    links = [item["link"] for item in results]
    # st.info(f"🔗 Found {len(links)} links for query: {query}")
    return links
@st.cache_data(show_spinner=False, ttl=3600)
def cached_google_search(query):
    return google_search(query)

DOWNLOAD_FOLDER = "downloads"
CONTEXT_FILE = os.path.join(DOWNLOAD_FOLDER, "context.txt")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)




MAX_CONCURRENCY = 5
write_lock = asyncio.Lock()

async def async_download(session: ClientSession, url: str, i: int, save_folder: str = DOWNLOAD_FOLDER):
    try:
        # st.info(f"🔍 Trying to download: {url}")
        async with session.get(url, headers=HEADERS, ssl=False, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            # st.info(f"ℹ️ Status {resp.status} - Content-Type: {resp.headers.get('Content-Type')}")
            content_type = resp.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type or url.endswith(".pdf"):
                file_name = url.split("/")[-1] or f"file_{i}.pdf"
                file_path = os.path.join(save_folder, file_name)
                content = await resp.read()
                with open(file_path, "wb") as f:
                    f.write(content)
                # st.info(f"✅ PDF downloaded: {file_name}")
                return ("pdf", file_path, file_name, i)

            elif "text/html" in content_type or url.endswith(".html"):
                file_name = f"page_{i}.html"
                file_path = os.path.join(save_folder, file_name)
                text = await resp.text()
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                # st.info(f"✅ HTML downloaded: {file_name}")
                return ("html", file_path, file_name, i, text, url)
            # else:
                # st.warning(f"⚠️ Skipping unknown content-type: {content_type}")

    except Exception as e:
        st.error(f"❌ Failed to download {url}: {e}")
    return None

async def process_and_append(file_info):
    if not file_info:
        return

    if file_info[0] == "pdf":
        _, file_path, file_name, i = file_info
        text = await asyncio.to_thread(cached_extract_pdf_text, file_path)
        print(f"📄 Extracted {len(text)} characters from: {file_name}")


    elif file_info[0] == "html":
        _, file_path, file_name, i, html_text, url = file_info
        text = await asyncio.to_thread(cached_extract_html_text, html_text)
        print(f"📄 Extracted {len(text)} characters from: {file_name}")


        # Embedded PDF detection
        embedded_pdfs = extract_embedded_pdfs(html_text, url)
        for j, pdf_url in enumerate(embedded_pdfs):
            try:
                response = requests.get(pdf_url, headers=HEADERS, verify=certifi.where(), timeout=20)
                response.raise_for_status()
                emb_file_name = f"embedded_{i}_{j}.pdf"
                emb_file_path = os.path.join(DOWNLOAD_FOLDER, emb_file_name)
                with open(emb_file_path, "wb") as f:
                    f.write(response.content)
                emb_text = cached_extract_pdf_text(emb_file_path)

                async with write_lock:
                    append_to_context_file(emb_file_name, emb_text, f"{i}-{j}")
            except Exception as e:
                print(f"❌ Embedded PDF error: {e}")

    async with write_lock:
        append_to_context_file(file_name, text, i)

async def download_and_process_all(links):
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        f.write("")
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    tasks = []
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY, ssl=False)
    async with ClientSession(connector=connector) as session:
        for i, link in enumerate(links[:5]):
            tasks.append(async_download(session, link, i))

        files = await asyncio.gather(*tasks)
        parse_tasks = [process_and_append(file_info) for file_info in files if file_info]
        await asyncio.gather(*parse_tasks)


@st.cache_data(show_spinner=False)
def cached_extract_pdf_text(path):
    return extract_pdf_text_with_tables(path)

@st.cache_data(show_spinner=False)
def cached_extract_html_text(html):
    return extract_html_text(html)


def preprocess_image_opencv(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    denoised = cv2.medianBlur(binary, 3)
    return denoised


def extract_embedded_pdfs(html_content, base_url):
    from urllib.parse import urljoin
    soup = BeautifulSoup(html_content, "html.parser")
    embedded_pdfs = []

    for tag in soup.find_all(["iframe", "embed", "object"]):
        src = tag.get("src") or tag.get("data")
        if src and ".pdf" in src.lower():
            if src.startswith("/"):
                src = urljoin(base_url, src)
            embedded_pdfs.append(src)
    return embedded_pdfs


def extract_pdf_text_with_tables(file_path):
    text = ""
    tables_output = ""

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        print(f"❌ Error extracting text with pdfplumber: {e}")

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        tables_output += " | ".join(cell if cell else "" for cell in row) + "\n"
                    tables_output += "\n"
    except Exception as e:
        print(f"⚠️ Failed extracting tables with pdfplumber: {e}")

    if tables_output.strip() == "":
        try:
            camelot_tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            for table in camelot_tables:
                tables_output += "\n".join([" | ".join(map(str, row)) for row in table.df.values.tolist()]) + "\n\n"
        except Exception as e:
            print(f"⚠️ Camelot extraction failed: {e}")

    if len(text.strip()) < 100:
        print("🔎 Using OCR fallback with OpenCV preprocessing...")
        try:
            images = convert_from_path(file_path)
            for img in images:
                processed_img = preprocess_image_opencv(img)
                ocr_text = pytesseract.image_to_string(processed_img)
                text += ocr_text + "\n"
        except Exception as e:
            print(f"❌ OCR fallback failed: {e}")

    if tables_output.strip():
        final_output = text + "\n==== Extracted Tables ====\n" + tables_output
    else:
        final_output = text

    return final_output


def extract_html_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.extract()

    visible_text = soup.get_text(separator="\n")
    lines = [line.strip() for line in visible_text.splitlines()]
    return "\n".join(line for line in lines if line)


def append_to_context_file(file_name, text, i):
    with open(CONTEXT_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n😃😃😃\n--- File{i}: {file_name} ---\n😃😃😃\n")
        f.write(text + "\n")


def download_file(url, i, save_folder=DOWNLOAD_FOLDER):
    try:
        try:
            response = requests.get(url, headers=HEADERS, verify=False, timeout=10)
            response.raise_for_status()
        except Exception:
            response = requests.get(url, headers=HEADERS, verify=certifi.where(), timeout=20)
            response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()

        if "application/pdf" in content_type or url.endswith(".pdf"):
            file_name = url.split("/")[-1] or f"file_{i}.pdf"
            file_path = os.path.join(save_folder, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"✅ PDF downloaded as: {file_path}")

            text = cached_extract_pdf_text(file_path)
            append_to_context_file(file_name, text, i)

        elif "text/html" in content_type or url.endswith(".html"):
            file_name = f"page_{i}.html"
            file_path = os.path.join(save_folder, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"✅ HTML downloaded as: {file_path}")

            text = cached_extract_html_text(response.text)
            append_to_context_file(file_name, text, i)

            embedded_pdfs = extract_embedded_pdfs(response.text, url)
            for pdf_url in embedded_pdfs:
                print(f"🔗 Found embedded PDF: {pdf_url}")
                try:
                    pdf_response = requests.get(pdf_url, headers=HEADERS, verify=True, timeout=20)
                    pdf_response.raise_for_status()
                    embedded_file_name = pdf_url.split("/")[-1] or f"embedded_{i}.pdf"
                    embedded_file_path = os.path.join(save_folder, embedded_file_name)
                    with open(embedded_file_path, "wb") as f:
                        f.write(pdf_response.content)
                    print(f"✅ Embedded PDF downloaded as: {embedded_file_path}")

                    embedded_text = cached_extract_pdf_text(embedded_file_path)
                    append_to_context_file(embedded_file_name, embedded_text, i)
                except Exception as e:
                    print(f"❌ Failed to download embedded PDF {pdf_url}: {e}")

        else:
            print(f"⚠️ Unknown content type ({content_type}) for {url}. Skipping...")

    except Exception as e:
        print(f"❌ Skipping {url}: {e}")

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = "llama3-8b-8192"
GEMINI_MODEL = "gemini-2.0-flash-lite"
CONTEXT_FILE = "downloads/context.txt"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 2
@st.cache_data(ttl=3600)
def load_and_split_context_file(context_file):
    try:
        with open(context_file, "r", encoding="utf-8") as f:
            full_text = f.read()
            if not full_text.strip():
                raise ValueError("File is empty")

        pattern = r"😃😃😃\n--- (File\d+): (.*?) ---\n😃😃😃(.*?)(?=(😃😃😃\n--- File\d+:|$))"
        matches = re.findall(pattern, full_text, re.DOTALL)

        if not matches:
            raise ValueError("No documents found - check your delimiter pattern")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True
        )

        docs = []
        for file_id, file_name, file_content, _ in matches:
            if not file_content.strip():
                continue

            sub_docs = splitter.create_documents(
                texts=[file_content],
                metadatas=[{"file_id": file_id, "file_name": file_name}]
            )
            docs.extend(sub_docs)

        return docs

    except Exception as e:
        raise RuntimeError(f"Document processing failed: {str(e)}")


# PERSIST_DIR = f"./vector_db{random.randint(0, 99999)}"
PERSIST_DIR = "chroma_db"

def create_vector_db(docs, persist_dir=PERSIST_DIR):
    if os.path.exists(persist_dir):
        try:
            client = PersistentClient(path=persist_dir)
            existing_collections = [col.name for col in client.list_collections()]
            if "langchain" in existing_collections:
                client.delete_collection("langchain")
                print("✅ Reset existing 'langchain' collection.")
        except Exception as e:
            print(f"⚠️ Failed to reset collection: {e}")
    else:
        os.makedirs(persist_dir, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # More reliable than GPU in Colab
        encode_kwargs={"normalize_embeddings": True}  # Better for similarity
    )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},  # Better similarity metric
    )

    print(f"Created collection with {vector_db._collection.count()} embeddings")
    return vector_db


def query_groq_api(query, context):
    system_prompt = """
You are an expert in extracting structured technical information from technical manuals.

Your task is to extract the following from the document **only if available**, using the JSON format provided below.
Extract numeric operating parameters even if scattered across the document. Avoid skipping numerical ranges or electrical specifications.

Output format (strictly JSON):
{
  "asset_type": "<Type of asset or equipment>",

  "manufacturer": "<Manufacturer name, if available>",
  "model": "<Model or variant, if mentioned>",
  "operating_parameters": {
    "voltage": "<Operating voltage, if mentioned>",
    "temperature_range": "<Temperature range, if available>",
    "other": ["<Other operating parameters if any>"]
  },
  "maintenance_procedures": ["<List of maintenance actions and intervals>"],

  "expected_lifespan_years": <Number of years or null>,
  "common_failure_modes": ["<List of common failure modes>"],
  "support_procedures": ["<List of troubleshooting or support procedures>"]
}

❗ For **common_failure_modes**, **if explicit failure modes are not mentioned but there are maintenance procedures or support instructions clearly implying possible failure causes (e.g., "check for leaks" → "leaks" are a failure mode)** — extract those implied failure modes.
DO NOT fabricate — Only derive failure modes when the language clearly indicates preventative action against a potential failure, not for generic routine checks.
Example:
- Maintenance: "Check refrigerant lines for leaks." → Failure Mode: "Refrigerant leaks"

For expected_lifespan_years:

- If the document **explicitly mentions a number of years as the product's expected lifespan, service life, or designed operational duration**, extract it.
- If **warranty periods, replacement recommendations, or operational lifetime suggestions** are given and **clearly imply** an expected lifespan, extract that number.
- Do NOT fabricate values — only derive **when the connection between the text and lifespan is clear.**

Examples:
- “5-year warranty on compressor, 10-year parts warranty” → expected_lifespan_years: 10
- “Recommended replacement after 15 years of use” → expected_lifespan_years: 15
- “Designed for long service life” → NOT enough, leave as null.

If no explicit or implied value is present → output null.

Do not depend solely on finding exact field names or keywords in the document. For any field, if explicit information is missing but equivalent or clearly related details are present (even phrased differently), extract that information. Prioritize meaning and context over exact wording. However, if no relevant or equivalent information exists, return null or an empty list as appropriate.
Rules:
1. MUST FIND EACH FIELD IN THE CONTEXT
2. If not found, use null/[]
3. NEVER GUESS - only use explicit information
4. For numbers: extract exact values (e.g. "220V" not "standard voltage")
5. For lists: include ALL found instances
6. Output ONLY the JSON object, nothing else.
7. DO NOT HALLUCINATE
8. do not give ranges or multiple values, just give one inclusive of all range
9. **For voltage:** Provide **all ranges** found, as an **array of strings**.
10. **For temperature_range:** Split into the 3 categories above if applicable.
11. **Numerical values MUST include units if mentioned.** Prefer SI units.
12. **If conflicting data is present, include only the most relevant one.**
13. Output **ONLY** the JSON object. **NO commentary, explanations, or extra text.**
14. **NO ranges like “multiple” or “various” unless explicitly stated.**
"""

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"QUERY: {query}\n\nCONTEXT:\n{context}"}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def search_vector_db(query):
    return vector_db.similarity_search(query, k=2)

def get_technical_specs(model_number, vector_db, documents, k=TOP_K):
    query_templates = {
        "General Specs": [
            "technical specifications", "specifications", "datasheet",
            "model number", "manufacturer name", "product details"
        ],
        "Voltage": [
            "operating voltage", "main voltage", "input voltage", "main power supply",
            "electrical specifications", "rated voltage", "nominal voltage", "control voltage"
        ],
        "Temperature": [
            "temperature range", "ambient temperature", "operating temperature",
            "working temperature", "environment temperature"
        ],
        "Capacity": [
            "cooling capacity", "pressure rating", "power rating"
        ],
        "Maintenance": [
            "maintenance schedule", "maintenance procedures", "periodic maintenance",
            "replacement procedures", "cleaning instructions", "fan maintenance",
            "maintenance interval", "maintenance time"
        ],
        "Lifespan": [
            "expected lifespan", "operational lifespan", "designed service life",
            "product lifetime", "average lifetime", "estimated lifespan",
            "product service life", "replacement interval", "years of operation",
            "durability in years", "expected working life", "typical lifespan", "warranty period"
        ],
        "Failures": [
            "common failure modes", "failure causes", "malfunctions", "known issues",
            "potential problems", "fault conditions", "common problems",
            "faults and errors", "issues during operation", "repair causes"
        ],
        "Support": [
            "troubleshooting steps", "support procedures", "installation instructions",
            "important information", "warranty conditions", "safety warnings", "damage reasons"
        ]
    }

    queries = [f"{model_number} {q}" for group in query_templates.values() for q in group]

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers = min(8, len(queries))) as executor:
        future_to_query = {executor.submit(search_vector_db, q): q for q in queries}
        for future in concurrent.futures.as_completed(future_to_query):
            try:
                docs = future.result()
                results.extend(docs)
            except Exception as e:
                print(f"⚠️ Query failed for '{future_to_query[future]}': {e}")

    seen = set()
    unique_docs = []
    for doc in results:
        doc_hash = hash(doc.page_content)
        if doc_hash not in seen:
            seen.add(doc_hash)
            unique_docs.append(doc)

    return unique_docs

def extract_specifications(model_number, vector_db, documents):
    """Enhanced extraction workflow"""
    context_docs = get_technical_specs(model_number, vector_db, documents)
    context = "\n\n".join(
        f"--- From {doc.metadata['file_name']} ---\n{doc.page_content}"
        for doc in context_docs
    )
    return query_groq_api(
        query=f"Extract all technical specifications for {model_number}",
        context=context
    )


def query_gemini_api(query, context):

    """Query the Gemini API for technical specifications"""
    system_prompt = """
You are a helpful technical assistant. You are given chunks of a product's official manual and documentation, and your task is to answer user questions about the device.

Use only the context provided to you — do not guess, assume, or add extra information. If a direct answer cannot be found in the context, respond with: "Not enough information available in the manual."

When answering:

1. Be **clear and concise**.
2. **Cite the source** (e.g., filename or section name) if possible.
3. Extract **exact technical values**, steps, or terminology from the context when present.
4. If the user asks for steps (e.g., setup, troubleshooting, replacement), **list them in order** and use bullet points or numbers.
5. If multiple relevant contexts are found, **combine them meaningfully**.

Do **not** use outside knowledge. Base every answer only on the content you're given.

Context may include details like:

- Electrical specifications (voltage, amperage)
- Temperature range or operating conditions
- Maintenance intervals and procedures
- Safety instructions
- Troubleshooting tips
- Warranty or service information
- Expected lifespan or replacement guidelines

If the query is vague or general, return the most relevant structured and technical information possible based on the context.

Always act as if you're responding to a technician or engineer who needs precise answers.
"""

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": f"QUERY: {query}\n\nCONTEXT:\n{context}"}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
def reset_app():
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # Delete downloads folder
    if os.path.exists(DOWNLOAD_FOLDER):
        shutil.rmtree(DOWNLOAD_FOLDER)

    st.success("✅ Reset complete. You can now start fresh!")
st.set_page_config(page_title="Tech Manual Extractor", layout="wide")
st.title("📘 Technical Manual Extractor")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "device_name" not in st.session_state:
    st.session_state.device_name = None
if "result_json" not in st.session_state:
    st.session_state.result_json = None
if "llm_answer" not in st.session_state:
    st.session_state.llm_answer = None
device = st.text_input("🔍 Enter the Device Model/Name")
st.session_state.device_name = device
query = f"{device} owner's manual PDF filetype:pdf"
if st.button("🔧 Extract Technical Specifications"):
    if st.session_state.result_json:
            pass
            
    else:
        with st.spinner("⏳ Processing..."):
            try:
                links = cached_google_search(query)
                # st.write("🔗 Google Search Results:")
                # for i, link in enumerate(links):
                #     st.write(f"{i+1}. {link}")
                asyncio.run(download_and_process_all(links))
                documents = load_and_split_context_file(CONTEXT_FILE)
                vector_db = create_vector_db(documents)
                st.session_state.vector_db = vector_db
                result = extract_specifications(device, vector_db, documents)
                st.session_state.result_json = result
                

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
if st.session_state.result_json:
    st.success("✅ Extraction Complete!")
    st.subheader("📄 Extracted Technical Specs")
    st.info(st.session_state.result_json)


st.markdown("---")
if st.session_state.vector_db is not None:
    st.subheader("💬 Ask About This Device")
    user_query = st.text_input("Ask a question based on the manual:")

    if st.button("🎯 Query Device Docs"):
        with st.spinner("🔍 Searching and processing..."):
            try:
                docs = st.session_state.vector_db.similarity_search(user_query, k=30)
                context = "\n\n".join(
                    f"--- From {doc.metadata['file_name']} ---\n{doc.page_content}"
                    for doc in docs
                )
                structured_answer = query_gemini_api(
                    query=user_query,
                    context=context
                )
                st.session_state.llm_answer = structured_answer

            except Exception as e:
                st.error(f"❌ Query error: {str(e)}")
else:
    st.info("ℹ️ Extract the manual first to enable document querying.")

if st.session_state.llm_answer:
    st.success("✅ Answer generated!")
    st.subheader("🧠 LLM Response")
    st.info(st.session_state.llm_answer)

st.markdown("---")
if st.button("🔁 Reset Everything"):
    reset_app()
    st.rerun()
