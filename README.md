# ⚡ Knowledge-Base Search Engine  
> *“Upload. Ask. Understand.” — AI-powered document Q&A system built using Node.js, ChromaDB, and cutting-edge AI APIs.*

---

## 📂 [🔗 Video preview Google Drive)]([YOUR_DRIVE_LINK_HERE](https://drive.google.com/file/d/1gloxDU0pR4lwZYvpl3cN4h_HwkgNWJn3/view?usp=sharing))
Easily download the entire source code directly from Google Drive.

---

<p align="center">
  <img src="https://img.shields.io/badge/Frontend-ReactJS-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Backend-Node.js-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/AI-Gemini_|_Groq_|_HuggingFace_|_Xenova-orange?style=for-the-badge"/><br>
  <img src="https://img.shields.io/badge/Database-ChromaDB-lightgrey?style=for-the-badge"/>
</p>

---

## 🧠 Overview

The **Knowledge-Base Search Engine** is a lightweight yet powerful AI application that enables users to:

- 📄 Upload multiple **PDF** or text documents  
- 💬 Ask **natural language questions**  
- 🤖 Get **AI-synthesized answers** directly from document content  

Built for simplicity, speed, and expandability — ideal for research, knowledge discovery, and academic summarization.

---

## ✨ Features

✅ **Multi-Document Upload:** Drag-and-drop or select multiple PDFs  
🧠 **Hybrid AI Intelligence:** Combines Gemini, Groq, HuggingFace, and Xenova for best response quality  
🔍 **Contextual Q&A:** Ask natural questions across all uploaded documents  
⚡ **ChromaDB Integration:** Embedding-based document retrieval  
🧩 **LangChain RAG Compatible:** Optional retrieval-augmented generation setup  
💻 **Fast Setup:** Plug-and-play backend with simple frontend  

---

## 🏗️ Tech Stack

| Layer              | Technology |
|--------------------|-------------|
| **Frontend**       | ReactJs *(Modern UI)* |
| **Backend**        | Node.js + Express |
| **AI Models**      | Gemini API, Groq, HuggingFace (via Xenova) |
| **Vector Store**   | ChromaDB |
| **File Parsing**   | pdf-parse |
| **Optional**       | LangChain for advanced RAG |

---
# Project Architecture
---
``` bash
knowledge-base-engine/
│
├── server.js # Express backend
├── .env # API keys & config
├── uploads/ # Uploaded PDF files
├── frontend/ # React app (Vite)
│ ├── src/
│ │ ├── App.jsx
│ │ └── main.jsx
│ └── package.json
└── README.md
```
---
## ⚙️ Quick Setup

### 🔧 1. Clone or Download

```bash
git clone https://github.com/yourusername/knowledge-base-engine.git
cd knowledge-base-engine
```
### 📦 2. Install Backend Dependencies
```bash
npm install express multer pdf-parse openai cors dotenv chromadb
npm install @langchain/community @langchain/openai langchain
```

### 🔑 3. Environment Setup

Create a .env file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```
### 4. Start Backend (Port 5001)
```bash
node server.js
```
### 5. Start Frontend (Port 5173)
```bash
npm run dev
```
---
### 🚀 Usage

Upload one or more PDF files

Enter your question in the input box

### Receive detailed, synthesized answers from your document database
---

## 🧪 Example Query

### User: “Summarize the main topics across all uploaded documents.”
### AI: “The files primarily discuss vector embeddings using ChromaDB, hybrid AI inference with HuggingFace and Gemini APIs, and document-based retrieval using LangChain.”
---

🧭 Future Improvements

🧱 Persistent ChromaDB storage

🔐 User authentication + session history

☁️ Deployment on Render / Vercel

🧩 Hybrid RAG with memory-based context

---

# 👨‍💻 Author

## Ayushmaan kumar Yadav<br>
🎓 CSE Undergraduate | 💪 Fitness & Tech Enthusiast<br>
🔗 [GitHub](https://github.com/ayushmaan19)
