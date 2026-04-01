import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

SYSTEM_PROMPT = """You are an assistant that answers questions using only the information found in the provided context.
The context contains yearly financial data for three companies.

When answering:
- Use only the provided context (the retrieved dataset content).
- Do not invent any facts or speculate beyond what is in the context.
- If the answer is not in the context, respond exactly:

  "I don’t have that information in the provided context."

- If the answer is present, provide it clearly and, when helpful, begin with:

  "Based on the provided context..."
"""

PROMPT_TEMPLATE = """{system_prompt}

Context:
{context}

Question:
{question}

Answer:"""


def load_documents(data_dir: str):
    from langchain_core.documents import Document
    from pypdf import PdfReader

    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_name in sorted(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file_name)

        if os.path.isdir(file_path):
            continue

        if file_name.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"

            documents.append(Document(page_content=text, metadata={"source": file_path}))

    return documents


def get_llm():
    from langchain_ollama import OllamaLLM
    return OllamaLLM(model="llama3")


def simple_split_documents(documents, chunk_size=500, chunk_overlap=100):
    from langchain_core.documents import Document

    texts = []
    for doc in documents:
        content = doc.page_content
        start = 0

        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]

            texts.append(Document(page_content=chunk, metadata=doc.metadata))

            start += chunk_size - chunk_overlap

    return texts


# ✅ FIXED: Properly placed outside main loop
def compute_valuation_matrix(docs_with_scores, query):
    valued_docs = []
    
    for doc, score in docs_with_scores:
        content = doc.page_content
        
        # Similarity score (60%)
        similarity = 1 / (1 + score)
        
        # Length score (20%)
        length_score = min(len(content) / 500, 1)
        
        # Redundancy penalty (10%)
        redundancy_penalty = content.count("\n") * 0.01
        
        # Keyword matching (30%)
        query_keywords = set(query.lower().split())
        content_lower = doc.page_content.lower()
        keyword_matches = sum(1 for kw in query_keywords if kw in content_lower)
        keyword_score = min(keyword_matches / len(query_keywords), 1) if query_keywords else 0
        
        final_score = (
            0.4 * similarity +
            0.2 * length_score +
            0.3 * keyword_score -
            0.1 * redundancy_penalty
        )
        
        valued_docs.append((doc, final_score))
    
    valued_docs.sort(key=lambda x: x[1], reverse=True)
    return valued_docs

    valued_docs.sort(key=lambda x: x[1], reverse=True)
    return valued_docs


def main():
    documents = load_documents(DATA_DIR)

    if not documents:
        raise RuntimeError("No documents found.")

    texts = simple_split_documents(documents)

    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    from langchain_chroma import Chroma

    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIR)

    llm = get_llm()

    while True:
        query = input("\nEnter question (or 'quit'): ").strip()

        if query.lower() in {"quit", "exit", "q"}:
            break

        # 🔍 Retrieve with scores
        docs_with_scores = db.similarity_search_with_score(query, k=5)

        # 🧠 Apply valuation matrix
        valued_docs = compute_valuation_matrix(docs_with_scores, query)

        # 🎯 Select best chunks
        top_docs = [doc for doc, _ in valued_docs[:2]]

        context = "\n".join([doc.page_content for doc in top_docs])

        # 📊 Debug: show scores
        print("\n=== Valuation Matrix ===")
        for i, (doc, score) in enumerate(valued_docs, 1):
            print(f"\n{i}. Score: {score:.4f}")
            print(f"Source: {doc.metadata.get('source', 'N/A')}")
            print(f"Preview: {doc.page_content[:150]}...")
        # 🧠 Build prompt
        full_prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            context=context,
            question=query
        )

        # 🤖 Generate answer
        answer = llm.invoke(full_prompt)

        print("\n=== Answer ===")
        print(answer)

        # 📚 Show sources
        print("\n=== Top Sources ===")
        for i, doc in enumerate(top_docs, 1):
            print(f"\n--- Source {i} ---")
            print(doc.page_content[:300])

        


if __name__ == "__main__":
    main()