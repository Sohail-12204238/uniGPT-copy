# from langchain_qdrant import Qdrant
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_cohere import ChatCohere
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.tools import DuckDuckGoSearchRun
# from qdrant_client import QdrantClient

# # --- Configuration ---
# VECTOR_STORE_PATH = "./vector_store"
# COLLECTION_NAME = "Uni_data"
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# COHERE_API_KEY = "dRrV13nDT200p0mV7g3hKyBfmgIAf2YWxspzO4cl" # It's better to use environment variables for keys

# def initialize_rag_chain():
#     """
#     Initializes and returns the complete RAG chain with a fallback mechanism.
#     """
#     # 1. Setup Embeddings
#     encode_kwargs = {'normalize_embeddings': False}
#     embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs=encode_kwargs)

#     # 2. Connect to the existing Qdrant vector store
#     client = QdrantClient(path=VECTOR_STORE_PATH)
#     vector_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)

#     # 3. Create a retriever
#     retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 2})

#     # 4. Setup the LLM
#     llm = ChatCohere(
#         model="command-a-03-2025",
#         temperature=0.3,
#         max_tokens=300,
#         cohere_api_key=COHERE_API_KEY
#     )

#     # 5. Create the prompt template
#     prompt = ChatPromptTemplate.from_template("""
#     You are an assistant that answers students' questions about university details, rules, and guidelines.

#     - Use ONLY the provided context to answer.
#     - If the context is empty or irrelevant, respond with: "NO_CONTEXT_FOUND".
#     - Never invent information outside the context.

#     <context>
#     {context}
#     </context>

#     Question: {input}
#     """)

#     # 6. Create the chains
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
#     # 7. Setup the web search fallback tool
#     web_search = DuckDuckGoSearchRun()

#     # 8. Define the final query function with fallback logic
#     def answer_with_fallback(query: str) -> str:
#         # First, try to get an answer from the retrieval chain
#         response = retrieval_chain.invoke({"input": query})
        
#         # Check if the response from the documents is valid
#         if response["answer"].strip() != "NO_CONTEXT_FOUND":
#             return response["answer"]
#         else:
#             # If not, fall back to a web search
#             print("Fallback to web search...")
#             return "From Internet: " + web_search.run(query)

#     return answer_with_fallback

# from langchain_qdrant import Qdrant
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_cohere import ChatCohere
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# # 1. Import Tavily instead of DuckDuckGo
# from langchain_community.tools.tavily_search import TavilySearchResults
# from qdrant_client import QdrantClient
# import os

# # --- Configuration ---
# VECTOR_STORE_PATH = "./vector_store"
# COLLECTION_NAME = "Uni_data"
# MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# COHERE_API_KEY = "dRrV13nDT200p0mV7g3hKyBfmgIAf2YWxspzO4cl" 

# # 2. Add your Tavily API Key
# # It's best practice to set this as an environment variable
# os.environ["TAVILY_API_KEY"] = "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"

# def initialize_rag_chain():
#     """
#     Initializes and returns the complete RAG chain with a fallback mechanism.
#     """
#     # 1. Setup Embeddings
#     encode_kwargs = {'normalize_embeddings': False}
#     embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs=encode_kwargs)

#     # 2. Connect to the existing Qdrant vector store
#     client = QdrantClient(path=VECTOR_STORE_PATH)
#     vector_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)

#     # 3. Create a retriever
#     retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 2})

#     # 4. Setup the LLM
#     llm = ChatCohere(
#         model="command-a-03-2025",  # Using a current, recommended model
#         temperature=0.3,
#         max_tokens=300,
#         cohere_api_key=COHERE_API_KEY
#     )

#     # 5. Create the prompt template
#     prompt = ChatPromptTemplate.from_template("""
#     You are an assistant that answers students' questions about university details, rules, and guidelines.

#     - Use ONLY the provided context to answer.
#     - If the context is empty or irrelevant, respond with: "NO_CONTEXT_FOUND".
#     - Never invent information outside the context.

#     <context>
#     {context}
#     </context>

#     Question: {input}
#     """)

#     # 6. Create the chains
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
#     # 7. Setup the web search fallback tool
#     # Replaced DuckDuckGo with TavilySearchResults
#     web_search = TavilySearchResults(k=3)

#     # 8. Define the final query function with fallback logic
#     def answer_with_fallback(query: str) -> str:
#         # First, try to get an answer from the retrieval chain
#         response = retrieval_chain.invoke({"input": query})
        
#         # Check if the response from the documents is valid
#         if response["answer"].strip() != "NO_CONTEXT_FOUND":
#             return response["answer"]
#         else:
#             # If not, fall back to Tavily search
#             print("Fallback to Tavily Search...")
            
#             # 3. Handle Tavily's structured output
#             search_results = web_search.invoke(query)
            
#             if not search_results:
#                 return "From Internet: I couldn't find any information on that topic."
            
#             # Format the results into a readable string for the user
#             formatted_results = "\n\n".join(
#                 [f"Source: {res['url']}\nContent: {res['content']}" for res in search_results]
#             )

#             return "From Internet:\n" + formatted_results

#     return answer_with_fallback

# In rag_core.py

from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools.tavily_search import TavilySearchResults
from qdrant_client import QdrantClient
import os

# --- Configuration ---
# VECTOR_STORE_PATH = "./vector_store"
COLLECTION_NAME = "Uni_data1"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
# --- Configuration ---
# ... other lines
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")

def initialize_rag_chain():
    """
    Initializes and returns the complete RAG chain with a fallback mechanism.
    """
    # ... (Steps 1, 2, 3: Embeddings, Vector Store, Retriever are unchanged)
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs=encode_kwargs)
    # client = QdrantClient(path=VECTOR_STORE_PATH)
    client = QdrantClient(
    url="https://d33be019-c2e3-4e1e-bcfc-424baddabad3.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.e75LvvcviBo8f0E6bte2bX8jR7T9FyYdhEADxt0NPRA",
    )
    vector_store = Qdrant(client=client, collection_name=COLLECTION_NAME, embeddings=embeddings)
    retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 2})

    # 4. Setup the LLM
    llm = ChatCohere(
        model="command-a-03-2025",
        temperature=0.3,
        max_tokens=300,
        cohere_api_key=COHERE_API_KEY
    )

    # 5. Create the prompt template for the main RAG chain
    rag_prompt = ChatPromptTemplate.from_template("""
    You are an assistant working for Lovely Professional University(LPU) that answers students' questions about university details, rules, and guidelines about Lovely Professional University(LPU) only.
    - Use ONLY the provided context to answer.
    - If the context is empty or irrelevant, respond with: "NO_CONTEXT_FOUND".
    - Never invent information outside the context.
    <context>{context}</context>
    Question: {input}
    """)

    # 6. Create the main RAG chain
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 7. Setup the web search fallback tool
    web_search = TavilySearchResults(k=3)

    # 8. Define the final query function with enhanced fallback logic

    # <-- NEW: Create a separate prompt and chain for summarizing web results -->
    summarization_prompt = ChatPromptTemplate.from_template("""
    Based on the following search results, provide a short and straightforward answer to the user's question.
    Do not mention the sources in your final answer. Just provide the answer directly.
    
    Search Results:
    {context}
    
    Question:
    {question}
    """)
    summarization_chain = summarization_prompt | llm

    def answer_with_fallback(query: str) -> str:
        # First, try to get an answer from the retrieval chain
        response = retrieval_chain.invoke({"input": query})
        
        if response["answer"].strip() != "NO_CONTEXT_FOUND":
            return response["answer"]
        else:
            print("Fallback to Tavily Search and Summarize...")
            
            # Get search results from Tavily
            search_results = web_search.invoke(query)
            
            if not search_results:
                return "I couldn't find any information on that topic using a web search."
            
            # Format the results into a single context string
            formatted_context = "\n\n".join(
                [f"Source: {res['url']}\nContent: {res['content']}" for res in search_results]
            )
            
            # <-- NEW: Use the summarization chain to get a concise answer -->
            summarized_response = summarization_chain.invoke({
                "context": formatted_context,
                "question": query
            })

            return summarized_response.content

    return answer_with_fallback