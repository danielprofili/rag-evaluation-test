import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

# Load environment variables from the .env file
load_dotenv()

print('Loading PDF content...')
loader = PyPDFLoader('TechCorp_Official_Employee_Handbook.pdf')
document = loader.load()

# print(document[0].page_content)

print('Chunking text...')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # size of each chunk
    chunk_overlap=50 # each chunk shares 50 chars with neighbors
)
chunks = text_splitter.split_documents(document)
# print(chunks[0].page_content)

print('Creating vector database...')
embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# configure db to act as a document retriever
# only pull the top 2 relevant chunks
retriever = vector_db.as_retriever(search_kwargs={'k' : 2})

# Define the hidden prompt structure for the LLM
template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Initialize the free Gemini model tier
# note: Gemini 3.8 is really slow
# note: ideally for this demo we'd want temperature=0, but gemini 3.6 ignores the
#       temperature flag
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Helper function to stitch retrieved chunks into a single text block
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Connect everything together using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# decorator so that this is traced in LangSmith
@traceable()
def rag_ask(user_question):
    # return rag_chain.invoke(user_question)
    docs = retriever.invoke(user_question)
    return {"answer" : rag_chain.invoke(user_question).content, "documents" : docs}
