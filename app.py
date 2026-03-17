import streamlit as st
from huggingface_hub import InferenceClient
import numpy as np
from pypdf import PdfReader

# --- CONFIGURATION ---
HF_TOKEN = st.secrets["HF_TOKEN"]
client = InferenceClient(api_key=HF_TOKEN)

def query_qa_api(question, context):
    # Using Llama 3.2 for conversational and generative answers
    messages = [
        {"role": "system", "content": "You are a focused document assistant. Answer the question using ONLY the provided text. If the answer isn't there, simply say: 'I'm sorry, I couldn't find that information in the document.' Never mention 'context', 'prompt', or 'provided text' in your response."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    
    return client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=messages,
        max_tokens=300,
        temperature=0.1
    )

def get_embeddings(text_list):

    return client.feature_extraction(
        model="BAAI/bge-base-en-v1.5",
        text=text_list
    )

# Header
st.title("RAG")
st.write("Upload a document and ask any question about it as if you are speaking to it!")

# File uploader
uploaded_file = st.file_uploader("Upload a document", type="pdf")

context = ""

if uploaded_file is not None:
    # If a file is uploaded, extract all text from it
    reader = PdfReader(uploaded_file)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()
    context = pdf_text
    
    words = context.split()
    word_count = len(words)
    if word_count > 10000:
        st.warning(f"Your document is too long with {word_count} words")
        st.stop()

    # NEW: Only scan the document if we haven't seen this file before
    if "my_id" not in st.session_state or st.session_state["my_id"] != uploaded_file.name:
        with st.spinner("Scanning document for the first time..."):
            # 1. Create the chunks
            chunk_size = 200
            overlap = 50
            step = chunk_size - overlap
            my_chunks = []
            for i in range(0, len(words), step):
                my_chunks.append(" ".join(words[i : i + chunk_size]))
            
            # 2. Turn them into numbers (The heavy work!)
            my_embeddings = get_embeddings(my_chunks)
            
            # 3. Save them into our 'Memory Box' (session_state)
            st.session_state["my_id"] = uploaded_file.name
            st.session_state["my_chunks"] = my_chunks
            st.session_state["my_embeddings"] = my_embeddings

    st.info("Document uploaded successfully!")

    # Question input
    with st.form("my_question_form"):
        question = st.text_input("Ask any question:")
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        # Grab the processed data from our 'Memory Box' instead of recalculating!
        paragraphs = st.session_state["my_chunks"]
        para_embeddings = st.session_state["my_embeddings"]

        if not paragraphs:
            st.warning("Could not find any clear text in the document.")
            st.stop()

        with st.spinner("Searching..."):

            bge_prefix = "Represent this sentence for searching relevant passages: "
            question_embedding = get_embeddings(bge_prefix + question)
            # Step 3: Mathematical Search (Cosine Similarity using Numpy)
            q_emb = np.array(question_embedding)
            if q_emb.ndim == 1:
                q_emb = q_emb.reshape(1, -1)
            p_emb = np.array(para_embeddings)
            
            # Normalize the vectors
            q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
            p_norm = p_emb / np.linalg.norm(p_emb, axis=1, keepdims=True)
            
            # Compute similarities
            similarities = np.dot(q_norm, p_norm.T)[0]
            
            # Get top 3 indices
            top_k = min(3, len(similarities))
            top_k_indices = np.argsort(similarities)[::-1][:top_k]

            best_context = ""
            for index in top_k_indices:
                best_context += paragraphs[index] + "\n\n"

            best_score = similarities[top_k_indices[0]]

            if best_score < 0.35:
                st.warning("Could not find any relevant information in the document.")
                st.stop()

        with st.spinner("Thinking..."):
            # Step 4: Ask the Cloud API using the official client
            result = query_qa_api(question=question, context=best_context)

        # DEBUG: Show the raw result from the cloud
        #if result and not hasattr(result, "error"):
        #    with st.expander("🤖 See raw AI Brain Result"):
        #        st.write(result)
            
        # step 5: show the user the AI's generated answer
        if not result or not hasattr(result, "choices"):
            st.error("API Error: Could not get a response from Llama.")
        else:
            answer_text = result.choices[0].message.content
            st.success(f"**Answer**: {answer_text}")

else:
    st.info("Please provide both a document and a question.")
