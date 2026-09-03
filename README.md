# 🧠 CV Intelligence Matcher

### AI-Powered Resume Analysis & Semantic Search using RAG + LLM

> **Turn resumes into an intelligent, searchable knowledge base — completely local.**

CV Intelligence Matcher is an AI-powered, **local-first** application that uses
**Retrieval-Augmented Generation (RAG)**, **semantic vector search**,
and a **local Large Language Model (LLM)** to analyze PDF resumes and
answer natural-language questions about candidates' skills, experience,
and qualifications.

Instead of relying on traditional keyword matching, the system
understands the **semantic meaning and context** of resume content,
retrieves the most relevant information, and generates grounded answers
based only on the retrieved context.

------------------------------------------------------------------------

## 🏠 Local-First & Offline by Design

A core idea of this project is that the AI pipeline can run **locally without requiring an internet connection**.

The system uses **Ollama + Mistral for local LLM inference**, while resume processing, embeddings, vector search, and RAG orchestration run on the local environment. The project documentation explicitly highlights offline execution and the benefits of keeping inference local. fileciteturn2file0L6-L7

### 🌐 Access from Devices on the Same Local Network

The application can also be exposed on the host machine's **local network**, allowing other devices connected to the same Wi-Fi/LAN to access the system through the host computer's local IP address.

For example, if the machine running the application has:

```text
192.168.1.46
```

other devices on the same network can access:

```text
http://192.168.1.46:8000/ui
```

This makes the project useful as a **local AI service** for a small office, lab, university environment, or home network — without deploying the application to a public cloud.

> **Note:** LAN access depends on the host machine being configured to listen on the network interface (for example, `0.0.0.0`), and Windows Firewall/network settings allowing the application port.

### 🔒 Why Local & Offline AI?

Running the system locally provides several practical advantages:

- 🔐 **Privacy:** CVs and candidate information can stay inside the local environment.
- 🌐 **No internet dependency:** The core AI workflow does not need a cloud LLM API.
- 💰 **No per-request API cost:** Local inference avoids paid cloud inference for each question.
- ⚡ **Lower network latency:** Requests between the UI, backend, vector database, and local LLM stay within the local environment.
- 🛡️ **Better control:** Organizations can keep sensitive recruitment documents under their own infrastructure.
- 🏢 **LAN sharing:** Multiple authorized devices on the same local network can use the application.
- 📦 **Portable deployment:** Docker makes the application easier to reproduce across compatible environments.

The project's architecture specifically uses local LLM inference with Ollama and describes offline execution, privacy, no API cost, and faster local inference as key benefits. fileciteturn2file0L6-L7

---

## ✨ What Makes It Different?

Traditional CV screening often depends on exact keyword matches.

This project takes a different approach:

``` text
PDF Resume
    ↓
Text Extraction
    ↓
Semantic Chunking
    ↓
HuggingFace Embeddings
    ↓
Qdrant Vector Search
    ↓
Relevant Context Retrieval
    ↓
Local LLM (Ollama + Mistral)
    ↓
Grounded Answer
```

The result is a lightweight, privacy-friendly CV analysis workflow that
can answer questions such as:

-   `Who has Docker experience?`
-   `Who worked on AI projects?`
-   `What are Sara's skills?`
-   `Who has experience with Python and NLP?`
-   `هل يوجد مرشح لديه خبرة في الشبكات؟`

------------------------------------------------------------------------

## 🚀 Key Features

-   📄 **PDF Resume Upload**
-   🔍 **Semantic Resume Search**
-   🤖 **RAG-based Question Answering**
-   🧠 **Local LLM Inference with Ollama**
-   🌍 **English & Arabic Queries**
-   🛡️ **Context-Grounded Responses**
-   🚫 **Hallucination Reduction**
-   ⚡ **FastAPI REST API**
-   🐳 **Docker Support**
-   🎨 **Modern Web Interface**
-   🌙 **Dark / Light Mode**
-   📚 **Automatic PDF Text Extraction**
-   🔢 **Vector Embeddings with Sentence Transformers**

------------------------------------------------------------------------

## 🚀 At a Glance

| Capability | Description |
|---|---|
| 🧠 **RAG + LLM** | Context-aware resume question answering |
| 🔍 **Semantic Search** | Finds relevant CV information by meaning, not only keywords |
| 🌍 **Arabic + English** | Natural-language querying in both languages |
| 📴 **Offline AI** | Local LLM inference without cloud AI APIs |
| 🔐 **Privacy-First** | Sensitive resumes can remain inside the local environment |
| 🌐 **LAN Access** | Access the application from other devices on the same local network |
| 🐳 **Dockerized** | Reproducible container-based deployment |
| ⚡ **FastAPI** | Lightweight REST backend and API documentation |

---

## 🏗️ System Architecture

``` text
                         ┌─────────────────────┐
                         │        User         │
                         │  Upload PDF / Query │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      FastAPI        │
                         │      Backend        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      PyMuPDF        │
                         │   PDF Text Extract  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Recursive Character │
                         │   Text Splitter     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ HuggingFace         │
                         │ Embeddings          │
                         │ all-MiniLM-L6-v2    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Qdrant        │
                         │   Vector Database   │
                         └──────────┬──────────┘
                                    │
                             Similarity Search
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Retrieved Context │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Ollama + Mistral   │
                         │     Local LLM       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Grounded Answer    │
                         └─────────────────────┘
```

------------------------------------------------------------------------

## 🧩 RAG Pipeline

The application follows a standard Retrieval-Augmented Generation
pipeline:

### 1. Upload

A user uploads a PDF resume through the web interface or API.

### 2. Extract

Resume text is extracted using **PyMuPDF**.

### 3. Chunk

The extracted content is divided into smaller semantic chunks using:

``` text
RecursiveCharacterTextSplitter
chunk_size    = 1200
chunk_overlap = 100
```

### 4. Embed

Each chunk is converted into a vector representation using:

``` text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting embeddings have **384 dimensions**.

### 5. Store

Embeddings are stored in a **Qdrant** vector collection using cosine
similarity.

### 6. Retrieve

When the user asks a question, the system performs semantic similarity
search and retrieves the most relevant chunks.

### 7. Generate

The retrieved context is injected into a strict prompt and passed to
**Mistral through Ollama**.

### 8. Answer

The system returns an answer grounded in the retrieved resume content.

------------------------------------------------------------------------

## 🛡️ Hallucination Control

One of the important design goals is preventing the LLM from inventing
candidate information.

The prompt explicitly instructs the model to:

``` text
Do NOT guess.
Do NOT invent information.
Answer ONLY from the provided context.
```

If the requested information cannot be found in the retrieved context,
the system instructs the model to respond:

``` text
I could not find the answer in the documents.
```

The retrieval stage also filters weak matches before sending context to
the LLM.

This makes the generated answers more **context-aware, traceable, and
reliable** for resume analysis.

------------------------------------------------------------------------

## 🌍 Arabic & English Support

The application supports natural-language questions in both **English
and Arabic**.

Examples:

``` text
Who has Python experience?
```

``` text
من لديه خبرة في بايثون؟
```

The same RAG pipeline is used to retrieve relevant resume information
and generate the response.

------------------------------------------------------------------------

## 🛠️ Technology Stack

  Technology                                Role
  ----------------------------------------- ---------------------------
  **Python**                                Core programming language
  **FastAPI**                               REST API & backend
  **LangChain**                             RAG pipeline components
  **PyMuPDF**                               PDF text extraction
  **HuggingFace / Sentence Transformers**   Text embeddings
  **Qdrant**                                Vector database
  **Ollama**                                Local LLM runtime
  **Mistral**                               Language model
  **Docker**                                Containerization
  **HTML / CSS / JavaScript**               User interface
  **Jinja2**                                Template support

------------------------------------------------------------------------

## 🔌 API Endpoints

The backend exposes the following main endpoints:

  ------------------------------------------------------------------------
  Method                  Endpoint                 Purpose
  ----------------------- ------------------------ -----------------------
  `GET`                   `/`                      Health/status check

  `GET`                   `/ui`                    Web interface

  `POST`                  `/upload`                Upload a PDF resume

  `POST`                  `/process/{file_name}`   Extract, chunk, embed
                                                   and index a resume

  `POST`                  `/search`                Semantic similarity
                                                   search

  `POST`                  `/answer`                RAG-based question
                                                   answering
  ------------------------------------------------------------------------

FastAPI also provides interactive API documentation through Swagger UI.

------------------------------------------------------------------------

## 📂 Project Structure

``` text
rag_project/
│
├── files/                  # Uploaded PDF resumes
├── templates/
│   └── index.html          # Web interface
│
├── main.py                 # FastAPI application & RAG pipeline
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Container orchestration
├── .env                    # Local environment configuration
└── README.md               # Project documentation
```

> **Security note:** `.env` should remain local and should not be
> committed if it contains secrets or private configuration.

------------------------------------------------------------------------

## ⚙️ Installation

### Prerequisites

Make sure you have:

-   Python 3.11+
-   Docker / Docker Compose
-   Ollama
-   A local Mistral model available through Ollama

### 1. Clone the repository

``` bash
git clone https://github.com/kldyahia/CV-Intelligence-Matcher.git
cd CV-Intelligence-Matcher
```

### 2. Configure environment variables

Create a `.env` file:

``` env
MODEL_NAME=mistral
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Start Ollama

Make sure Ollama is running and the configured model is available
locally.

For example:

``` bash
ollama run mistral
```

------------------------------------------------------------------------

## ▶️ Run Locally

Start the FastAPI application:

``` bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open:

``` text
http://localhost:8000/ui
```

API documentation:

``` text
http://localhost:8000/docs
```

------------------------------------------------------------------------

## 🐳 Run with Docker

Build the container:

``` bash
docker compose build
```

Start the application:

``` bash
docker compose up
```

The application is exposed on:

``` text
http://localhost:8000
```

The Docker setup maps the application port `8000` and mounts the project
directory into `/app`.

> **Note:** The current implementation uses Qdrant in memory and
> connects the containerized FastAPI application to Ollama through
> `host.docker.internal`.

------------------------------------------------------------------------

## 🧪 Example Workflow

### Upload a Resume

``` http
POST /upload
```

Example response:

``` json
{
  "message": "uploaded successfully",
  "file": "resume.pdf"
}
```

### Process the Resume

``` http
POST /process/resume.pdf
```

Example response:

``` json
{
  "chunks_count": 12,
  "message": "stored in qdrant"
}
```

### Ask a Question

``` http
POST /answer
```

Example:

``` text
Who has Docker experience?
```

The system retrieves the most relevant resume chunks and generates an
answer from the retrieved context.

------------------------------------------------------------------------

## 🎯 Project Goals

The project was designed to demonstrate how modern AI techniques can
improve document-based recruitment workflows by combining:

-   **NLP**
-   **Semantic Search**
-   **Vector Databases**
-   **Retrieval-Augmented Generation**
-   **Local LLMs**
-   **REST APIs**
-   **Containerization**
-   **Modern Web UI**

The focus is not simply on generating text, but on retrieving relevant
evidence before generating an answer.

------------------------------------------------------------------------

## 🔬 Technical Highlights

### Embedding Model

``` text
sentence-transformers/all-MiniLM-L6-v2
```

-   Lightweight
-   384-dimensional embeddings
-   Designed for semantic similarity

### Vector Search

Qdrant performs similarity search using:

``` text
COSINE
```

The application retrieves the top relevant chunks and applies a
similarity threshold before constructing the LLM context.

### LLM

The application uses:

``` text
Ollama → Mistral
```

This allows inference to run locally rather than depending on a paid
cloud LLM API.

------------------------------------------------------------------------

## 📈 Future Improvements

Potential next steps include:

-   🔐 Multi-user authentication
-   ☁️ Cloud deployment
-   📊 Candidate ranking and scoring
-   🎯 Advanced filtering and ranking
-   🧾 OCR support for scanned resumes
-   📚 Multi-file simultaneous indexing
-   💬 Chat history
-   🗂️ Persistent vector storage
-   👥 Multi-user resume management
-   🤖 More advanced candidate matching

------------------------------------------------------------------------

## 💡 Why RAG for CV Analysis?

A resume contains structured and unstructured information such as:

``` text
Skills
Experience
Education
Projects
Certifications
```

A pure LLM approach may generate plausible information that is not
actually present in the resume.

RAG introduces an evidence-retrieval step:

``` text
Question
   ↓
Retrieve relevant resume information
   ↓
Provide retrieved context to LLM
   ↓
Generate grounded answer
```

This makes the system better suited for document-based question
answering where the answer should come from the provided source
material.

------------------------------------------------------------------------

## 👨‍💻 Project

**CV Intelligence Matcher**

Built as an AI/RAG project demonstrating practical integration of:

**FastAPI + LangChain + HuggingFace + Qdrant + Ollama + Mistral +
Docker**

------------------------------------------------------------------------

## ⭐ If You Find This Project Interesting

Feel free to explore the code, experiment with different resumes and
questions, and build on top of the RAG pipeline.

**⭐ Star the repository if you find it useful!**

------------------------------------------------------------------------

### #AI #RAG #LLM #NLP #FastAPI #LangChain #Qdrant #HuggingFace #Ollama #Mistral #Docker #Python #SemanticSearch #AIEngineering
