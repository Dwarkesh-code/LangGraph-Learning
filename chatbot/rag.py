from langchain_unstructured import UnstructuredLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

def gen_vectorstores(file) :
    docs = UnstructuredLoader(file_path=file, strategy="fast").load()
    splits = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embedding_model)

    retriever = vectorstore.as_retriever(search_type="mmr")

    return retriever


def make_retriever_tool(retriever):
    @tool
    def search_document(query: str) -> str:
        """Search the uploaded document for relevant context."""
        docs = retriever.invoke(query, k=4, fetch_k=15)
        return "\n\n".join(d.page_content for d in docs)
    return search_document