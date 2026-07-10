# IMPORTANT NOTE : EVERYTIME WE UPDATE DOCS, WE HAVE TO RUN THE INGESTINO PIPELINE AGAIN OR THE VECTOR DB NOT UPDATED (maybe fix later)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# we always have to load the environemnt because we have our OPENAI_API_KEY in the .env
load_dotenv()

persistent_directory  = "db/chroma_db"

# Same embedding as in ingestion_pipeline.py, we HAVE to use the same embedding model as the ingestion, or else they won't understand each other
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# We already created a vector store so why create again?
# That's because we're pointing the vector store to the persistent directory where we stored the vector database in ingestion_pipeline.py
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query= "What was Microsoft's firsts hardware product release?"

# retrieve the top 3 most similar chunks to the query from the vector store
# retriever = db.as_retriever( search_kwargs={"k": 3})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.3})   # cosine similary scores are calculated for every chunk, 0 is not similar at all, 1 is identical
                                   # 0.3 is the minimum or else don't select

relevant_docs = retriever.invoke(query)

print(f"Query: {query}")
print("-- Context --")
for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:\n{doc.page_content}\n")


combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you cannot find the answer in the documents, please respond with "There is not enough information provided in the documents.relevant_docs.
"""

model = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=combined_input)
]

result = model.invoke(messages)

print("\n --- Generated Response ---")

print ("Content only:")
print(result.content)
