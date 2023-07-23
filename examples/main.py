import os
from getpass import getpass

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader

from langchain.document_loaders import TextLoader

if 'OPENAI_API_KEY' not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass('OPENAI API KEY:')

loader = TextLoader("./document.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "milvus", "port": "19530"},
)