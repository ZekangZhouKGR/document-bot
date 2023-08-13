import chromadb

server_setting = {
    "host": "chroma",
    "port": "8000"
}

chroma_client = chromadb.HttpClient(**server_setting)

collections = chroma_client.list_collections()
print(collections)