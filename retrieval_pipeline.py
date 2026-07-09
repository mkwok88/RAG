# IMPORTANT NOTE : EVERYTIME WE UPDATE DOCS, WE HAVE TO RUN THE INGESTINO PIPELINE AGAIN OR THE VECTOR DB NOT UPDATED (maybe fix later)

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

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

query= "What was NVIDIA's first graphics accelerator called?"

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


