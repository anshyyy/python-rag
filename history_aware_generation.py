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

llm = ChatOpenAI(model="gpt-4o")

chat_history = []


def ask_question(query: str):
    
    print(f"User Query: {query}\n")
 
    if chat_history:
        messages = [
            SystemMessage(content="Given the conversation history, rewrite the new questions to be standalone and searchable, just return the rewritten question."),
        ] + chat_history + [HumanMessage(content=query)]

        result = llm.invoke(messages)
        search_question = result.text.strip()
        print(f"Rewritten question for retrieval: {search_question}\n")
    else:
        search_question = query
        


    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(search_question)

    print(f"foud relevant documents: {len(relevant_docs)}\n")

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
    return response.text


def start_chat():
    print("Ask me anything based on the documents! Type 'exit' to quit.")
    while True:
        user_query = input("Your question: ")
        if user_query.lower().trim() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        answer = ask_question(user_query)
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    pass
