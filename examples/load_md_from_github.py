import sys
import re
import pathlib

from git import Repo
import chromadb
from langchain.text_splitter import Language
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.chat_models import ChatOpenAI


def get_repo_name(repo_url):
    # 提取仓库名的正则表达式
    pattern = r'\/([^\/]+)\.git$'

    # 使用正则表达式搜索仓库地址中的仓库名
    matches = re.search(pattern, repo_url)

    if matches:
        # 返回匹配结果中的第一个捕获组（即仓库名）
        return matches.group(1)
    else:
        # 如果未匹配到仓库名，则返回空字符串或抛出自定义异常
        return ''

model_name = "gpt-3.5-turbo"
server_setting = {
    "host": "chroma",
    "port": "8000"
}
# 示例用法
repo_url = sys.argv[1]
repo_name = get_repo_name(repo_url)
repo_path = pathlib.Path("/home/vscode/git_cache") / repo_name
if not repo_path.exists():
    repo = Repo.clone_from(repo_url, to_path=str(repo_path.absolute()))
else:
    repo = Repo(str(repo_path.absolute()))

chroma_client = chromadb.HttpClient(**server_setting)
collections_name = [c.name for c in chroma_client.list_collections()]

if repo_name not in collections_name:
    chroma_client.create_collection(name=repo_name)

embedding = OpenAIEmbeddings(disallowed_special=())
db = Chroma(collection_name=repo_name, client=chroma_client, embedding_function=embedding)

# Load
def load_documents_from_repo(repo_path):
    """Loads Markdown documents from a Git repository into a list of texts.

    Args:
    repo_path: Path object pointing to the root of the Git repository. 

    Returns:
    texts: A list of splitted texts extracted from the Markdown documents in the repository.

    For each .md file in the repository, it will be loaded using the UnstructuredMarkdownLoader.
    The extracted documents will be extended to the documents list.

    The documents are then split into chunks using the RecursiveCharacterTextSplitter configured for Markdown.
    The splitted texts are returned from this function.

    Any errors during loading the Markdown files will be printed out but skipped.
    """
    documents = []
    for mdx_file in repo_path.glob("**/*.md"):
        loader = UnstructuredMarkdownLoader(str(mdx_file))
        try:
            doc = loader.load()
            documents.extend(doc)
        except:
            print('parser error for', str(mdx_file))
    print(len(documents))
    
    markdown_spliter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=4000,
        chunk_overlap=200)
    texts = markdown_spliter.split_documents(documents)
    print(len(texts))
    return texts

texts = load_documents_from_repo(repo_path)
db.add_documents(texts)
retriever = db.as_retriever(
    search_type="mmr",  # Also test "similarity"
    search_kwargs={"k": 8},
)

llm = ChatOpenAI(model_name=model_name)
memory = ConversationSummaryMemory(
    llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory)

questions = [
]

for question in questions:
    result = qa(question)
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
