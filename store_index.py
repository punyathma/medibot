from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Extract text from PDF and split into chunks
extracted_data = load_pdf_file(data='D:/medibot/medibot/Data')
text_chunks = text_split(extracted_data)

# Optional: If using HuggingFace embeddings
# embeddings = download_hugging_face_embeddings()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot1"

# Check if index exists
existing_indexes = pc.list_indexes()
if index_name not in existing_indexes:
    # Create index if it doesn't exist
    pc.create_index(
        name=index_name,
        dimension=1536,  # Match this dimension with the embedding size (OpenAI text-embedding-ada-002 uses 1536)
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index {index_name} already exists.")

# Initialize Pinecone vector store and upsert text chunks
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

# Upsert the embeddings into Pinecone
docsearch.add_texts(texts=text_chunks)
print("Text chunks upserted successfully!")
