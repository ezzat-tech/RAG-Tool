# 🤖 Serverless RAG Tool

A fast, lightweight, and modern **Retrieval-Augmented Generation (RAG)** tool that allows you to "talk" to your PDF documents. Key focus: **Efficiency**. By using serverless APIs, this app starts instantly and uses minimal local resources.

## 🚀 Features
- **PDF Text Extraction**: Seamlessly extracts text from uploaded PDF files.
- **Serverless Embeddings**: Uses `BAAI/bge-base-en-v1.5` via Hugging Face Inference API for high-speed, low-memory vector search.
- **Smart Q&A**: Powered by `Meta-Llama-3.2-3B-Instruct` for context-aware, accurate answers.
- **Streamlit UI**: A clean, interactive dashboard for easy document interaction.

## 🛠️ Technology Stack
- **Framework**: Streamlit
- **Embeddings**: Hugging Face Inference (BGE)
- **LLM**: Meta Llama 3.2 (via Inference API)
- **Search**: Scipy/Torch (Semantic Similarity)

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Secrets**:
   Create a `.streamlit/secrets.toml` file and add your Hugging Face Token:
   ```toml
   HF_TOKEN = "your_huggingface_token_here"
   ```

4. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## 🔐 Security Note
This project uses a `.gitignore` to ensure your `secrets.toml` file is never pushed to public repositories. When deploying to Streamlit Cloud, remember to paste your `HF_TOKEN` into the app's secret management dashboard.

---
*Built with ❤️ for the AI community.*
