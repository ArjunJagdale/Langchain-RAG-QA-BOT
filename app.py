import gradio as gr
import os
import requests

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# ✅ Read OpenRouter API key from HF secret
OPENROUTER_API_KEY = os.environ.get("ArjunHF")

class OpenRouterChatModel(ChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=OPENROUTER_API_KEY,
            model_name="mistralai/mistral-small-3.2-24b-instruct:free",
            **kwargs
        )

def qa_on_url(url, question):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(split_docs, embeddings)
        retriever = vectordb.as_retriever()

        llm = OpenRouterChatModel(temperature=0.2)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        return qa_chain.run(question)
    except Exception as e:
        return f"❌ Error: {e}"

iface = gr.Interface(
    fn=qa_on_url,
    inputs=[gr.Textbox(label="Enter Web URL"), gr.Textbox(label="Your Question")],
    outputs="text",
    title="🔎 Ask Questions About Any Webpage (Mistral 3.2 via OpenRouter + LangChain)",
    description="⚠️ This may take 10–20 seconds depending on the page length and LLM response time. Please be patient!"
)

if __name__ == "__main__":
    iface.launch()