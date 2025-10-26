from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persist_directory="db/vector_store"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "What is the C++ and how its different from golang ?"

retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)


# print(f"User Query: {query}\n")
# print("---contents of relevant documents---\n")

# for i, doc in enumerate(relevant_docs):
#     print(f"Document {i+1} content preview: {doc.page_content}...\n")


#combine relevant documents into a single string and query 

prompt = f'''
         Based on the following documents, answer the question: {query}
            Documents:
            {''.join([doc.page_content for doc in relevant_docs])}
         '''


llm = ChatOpenAI(model="gpt-4o", temperature=0)

messages = [
    SystemMessage(content="You are a helpful assistant that provides concise and accurate answers based on the provided documents.") ,
    HumanMessage(content=prompt)
]

response = llm.invoke(messages)
print("-----Response from LLM-----\n")

print(response.text)


