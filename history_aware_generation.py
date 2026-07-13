from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

# Connection to our document database
persistent_directory  = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


# Set up AI model
model = ChatOpenAI(model="gpt-4o")

# Keep track of our conversation history
chat_history = []


def ask_question(question):
    print(f"\nQuestion: {question}")

    # Reformulating the question to be standalone if there is chat history
    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, reqrite the new question to be standalone and searchable. Just return the rewritten question."),
        ] + chat_history + [
            HumanMessage(content= f"New question: {question}")
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Rewritten question: {search_question}")

    # if there is no chat history, we can use the question as is
    else:
        search_question = question

    # Find relevant documents

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_question)

    newline = "\n"

    combined_input = f"""Based on the following documents, please answer this question: {search_question}

    Documents:
    {newline.join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=search_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer


def start_chat():
    print("Ask me a question! Type 'quit' to exit.")

    while True:
        question = input("\n Your question: ")

        if question.lower() == "quit":
            print("Exiting chat. Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()


