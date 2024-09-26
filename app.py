# import requests
# from flask import Flask, request, render_template,jsonify
# from langchain_community.llms import Ollama
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.prompts import PromptTemplate
# from transformers import pipeline
# from langchain_huggingface import HuggingFaceEmbeddings

# # Use a CPU-friendly embedding model
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



# app = Flask(__name__)

# folder_path = "db"

# # Endpoint for Ollama's local embedding service
# ollama_endpoint = "http://127.0.0.1:11434/api/embeddings"

# cached_llm = Ollama(model="phi3")

# # embedding = Ollama(model="all-minilm")
# # embedding_model = Ollama(model="nomic-embed-text") 
# # data chunking
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
# )

# # ollama endpoint api generation
# '''
# # Function to generate embeddings using Ollama's API
# def generate_embeddings(chunk_texts):
#     embeddings = []
#     for text in chunk_texts:
#         payload = {
#             "model": "all-minilm",  # Assuming this is your local model name
#             "prompt": text
#         }
#         try:
#             response = requests.post(ollama_endpoint, json=payload)
#             response.raise_for_status()
#             response_data = response.json()
#             embeddings.append(response_data['embedding'])
#         except requests.exceptions.RequestException as e:
#             print(f"Request error: {e}")
#         except requests.exceptions.JSONDecodeError as e:
#             print(f"JSON decode error: {e}")
#             print("Response content:", response.content)
#     return embeddings
# '''
# # sending the prompt to the model input is the input and context is from the db
# raw_prompt = PromptTemplate.from_template(
#     """ 
#     <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
#     [INST] {input}
#            Context: {context} 
#            Answer:
#     [/INST]
# """
# )

# @app.route("/")
# def index():
#     return render_template("index.html")

# # the api end point for taking the query input 
# @app.route("/ai", methods=["POST"])
# def aiPost():
#     print("Post /ai called")
#     json_content = request.json
#     query = json_content.get("query")

#     print(f"query: {query}")

#     # response = cached_llm.invoke(query)
#     response="This endpoint is working fine"

#     print(response)

#     response_answer = {"answer": response}
#     return response_answer

# # asking questions from these pdfs

# @app.route("/ask_pdf", methods=["POST"])
# def askPDFPost():
#     print("Post /ask_pdf called")
#     json_content = request.json
#     query = json_content.get("query")

#     print(f"query: {query}")

#     print("Loading vector store")
#     vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
#     # creating retreiver
#     print("Creating chain")
#     retriever = vector_store.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#             "k": 20, # picking 20 most similar chunks
#             "score_threshold": 0.1,
#         },
#     )

#     document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
#     chain = create_retrieval_chain(retriever, document_chain)

#     result = chain.invoke({"input": query})

#     print(result)

#     sources = []
#     for doc in result["context"]:
#         sources.append(
#             {"source": doc.metadata["source"], "page_content": doc.page_content}
#         )

#     response_answer = {"answer": result["answer"], "sources": sources}
#     return response_answer

# # uploading files to the pdf directory 
# @app.route("/pdf", methods=["POST"])
# def pdfPost():
#     file = request.files["file"]
#     file_name = file.filename
#     save_file = "pdf/" + file_name
#     file.save(save_file)
#     print(f"filename: {file_name}")
#     # check if its saved in the pdf directory
#     '''
#     response ={"status":"successfully uploaded","file_name":file_name}
#     print(response)
#     return response
#     '''
   
#     loader = PDFPlumberLoader(save_file)
#     docs = loader.load_and_split()
#     print(f"docs len={len(docs)}")

#     chunks = text_splitter.split_documents(docs)
#     print(f"chunks len={len(chunks)}")

#     vector_store = Chroma.from_documents(
#         documents=chunks, embedding=embedding, persist_directory=folder_path
#     )

#     vector_store.persist()

#     response = {
#         "status": "Successfully Uploaded",
#         "filename": file_name,
#         "doc_len": len(docs),
#         "chunks": len(chunks),
#     }
#     return response


# def start_app():
#     app.run(host="0.0.0.0", port=8080, debug=True) # flask app where it is running


# if __name__ == "__main__":
#     start_app()
import requests
from flask import Flask, request,render_template,jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from flask_cors import CORS


# Use a CPU-friendly embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

app = Flask(__name__)
CORS(app)

folder_path = "db"

# Cached LLM for testing
cached_llm = Ollama(model="phi3")

# Data chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

# Hardcoded response for testing
def get_hardcoded_response():
    return "This is a hardcoded response. The model is working fine."

@app.route("/")
def index():
    return render_template("index.html")

# Endpoint for testing the model
@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    # Use hardcoded response
    response = get_hardcoded_response()

    print(response)

    response_answer = {"answer": response}
    return response_answer

# Asking questions from PDFs
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    # Hardcoded response for testing
    response = get_hardcoded_response()
    sources = [{"source": "dummy_source", "page_content": "dummy_content"}]

    print(response)

    response_answer = {"answer": response, "sources": sources}
    return response_answer

# Uploading files to the PDF directory
@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    # Dummy response for file upload
    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": 0,
        "chunks": 0,
    }
    return response

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
