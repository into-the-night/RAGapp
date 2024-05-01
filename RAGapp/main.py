import os

from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
import chromadb

## set API Key
GOOGLE_API_KEY = "YOUR_API_KEY"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

## load document
documents = SimpleDirectoryReader(input_files=["final_data.csv"])
documents = documents.load_data()

## create vectordb
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection("quickstart")

## set embedding model
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)
Settings.embed_model = embed_model

## set llm model
Settings.llm = Gemini(api_key=GOOGLE_API_KEY)

## transformation
text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=10)

## create vector store
vector_store = ChromaVectorStore(chroma_collection=collection)

## save vectordb on disk (use only one time)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, transformations=[text_splitter])

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

query_engine = index.as_query_engine(
    similarity_top_k=1,
)

## create chat
while True:
    response = query_engine.query("Answer concisely but well worded"+str(input("Enter the query: ")))
    print(response)
    print("Continue?(y/n)")
    choice = input()
    if choice == 'y':
        pass
    else:
        break
