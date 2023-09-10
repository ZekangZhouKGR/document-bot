#!/bin/python3
"""
name: list_chromdb_database
description: 使用chromadb库连接ChromaDB数据库，获取数据库中的所有集合
domain: 数据库
tags: chromadb, ChromaDB, 数据库连接
"""

import chromadb

server_setting = {
    "host": "chroma",
    "port": "8000"
}

chroma_client = chromadb.HttpClient(**server_setting)

collections = chroma_client.list_collections()
print(collections)