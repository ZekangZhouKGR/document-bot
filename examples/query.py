import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from langchain.document_loaders import TextLoader

from langchain.document_loaders import TextLoader

if 'OPENAI_API_KEY' not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass('OPENAI API KEY:')

loader = TextLoader("./document.txt")

embeddings = OpenAIEmbeddings()

vector_db = Milvus(
    embedding_function=embeddings, 
    connection_args={
        "host": "milvus", "port":"19530"
    }
)

query = "衣服"
docs = vector_db.similarity_search(query)

for doc in docs:
    print('------------ doc ----------')
    print(doc.page_content)
