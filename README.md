## 🔍 Mistral-Powered Web QA App (LangChain + OpenRouter)

This project is a lightweight yet powerful **question-answering system** that allows users to:

> 🧠 Ask questions about the content of **any public web page URL**, powered by the free `mistralai/mistral-small-3.2-24b-instruct` model via OpenRouter.

---

### 🚀 What It Does

* ✅ Takes a URL as input
* ✅ Loads and splits the page into chunks using LangChain
* ✅ Converts chunks into vector embeddings (MiniLM)
* ✅ Performs semantic retrieval using FAISS
* ✅ Uses Mistral 3.2 (24B, free via OpenRouter) to synthesize answers

---

### 🧠 Why It's Not Just "API Wrapping"

This isn’t just plugging into an API. The system includes:

* 📄 **Document Parsing**: URL fetching, text extraction, chunking logic
* 📊 **Semantic Search**: Uses HuggingFace embeddings + FAISS vector search
* 🔄 **Retrieval-Augmented Generation (RAG)**: Uses retrieved text as context for LLM to answer accurately
* 🧩 **LangChain Chains**: Modular chaining logic to connect retriever + LLM

This pipeline simulates the behavior of a fine-tuned ML model for Q\&A — without requiring training from scratch.

---

### 🛠️ Tech Stack

| Component      | Tool                                                                                    |
| -------------- | --------------------------------------------------------------------------------------- |
| Text Retrieval | LangChain (`WebBaseLoader`, `RecursiveCharacterTextSplitter`)                           |
| Embeddings     | `sentence-transformers/all-MiniLM-L6-v2`                                                |
| Vector Store   | FAISS                                                                                   |
| LLM (Free API) | `mistralai/mistral-small-3.2-24b-instruct:free`                                         |
| Frontend       | Gradio                                                                                  |
| Deployment     | Hugging Face Spaces                                                                     |

---

### 🧑‍💻 Use Cases

* Ask questions about documentation or articles
* Educational summary generation
* Build RAG apps without training your own LLM
* Great base for interview-prep bots, study assistants, etc.

---

### 🔐 API Key Management

To use this on Hugging Face Spaces, we store the OpenRouter API key securely via **Space Secrets** (`ArjunHF`).

Absolutely! Here's an expanded and clear explanation of **LangChain’s role** in your project, written in clean, professional Markdown, ideal for `README.md` or project documentation:

---

### 🔗 What is LangChain Doing in This Project?

LangChain acts as the **orchestrator** that connects different components—document loaders, chunking logic, embeddings, retrievers, and LLMs—into a single, smart pipeline.

Here’s how LangChain powers the entire flow:

---

#### 📥 1. **Document Loading**

> `WebBaseLoader`

* LangChain uses `WebBaseLoader` to fetch and clean the raw content from a given web URL.
* It abstracts away boilerplate scraping code.
* Returns a list of `Document` objects for downstream processing.

---

#### ✂️ 2. **Text Splitting**

> `RecursiveCharacterTextSplitter`

* Large documents are split into manageable **overlapping text chunks**.
* This improves LLM comprehension and retrieval granularity.
* LangChain handles chunk boundaries intelligently using recursion on characters, newlines, sentences, etc.

---

#### 🧠 3. **Embedding & Vector Store**

> `HuggingFaceEmbeddings + FAISS`

* Each chunk is converted into a dense vector using a pretrained embedding model.
* These embeddings are stored in a **FAISS index** via LangChain’s `VectorStore` interface.
* LangChain lets you use this vector store as a retriever later on.

---

#### 🔍 4. **Context Retrieval**

> `retriever.as_retriever()`

* When a user asks a question, LangChain performs **semantic search** over the FAISS index to find the most relevant chunks.
* These are passed as context to the LLM for more grounded answers.

---

#### 🧠 5. **Answer Generation**

> `RetrievalQA` chain

* LangChain uses a **Retrieval-Augmented Generation (RAG)** setup via `RetrievalQA.from_chain_type()`.
* It plugs in the retriever + the OpenRouter-backed Mistral LLM.
* Automatically forms prompts like:
  `"Given the context: <retrieved_docs> — answer the question: <user_question>"`

---

#### ✅ Final Benefit of LangChain

LangChain makes it easy to:

* Chain steps modularly without rewriting logic
* Swap components (e.g., replace FAISS with Qdrant, Mistral with Claude, or add memory)
* Build **production-quality LLM pipelines** using open components

---

> Without LangChain, this would require 200+ lines of manual data handling and orchestration logic. With it, you built a smart, extensible RAG system in \~30 lines of code.
