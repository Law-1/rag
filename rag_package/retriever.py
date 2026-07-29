from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
import shutil

from .config import DASHSCOPE_API_KEY, EMBEDDING_MODEL_NAME

def build_retriever(documents: List[Document], persist_dir: str = "./chroma_db"):
    """构建混合检索器（BM25+向量）+重排序，返回可调用的检索器"""
    # 嵌入模型
    embeddings = DashScopeEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )

    # 文本切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"切分后共 {len(chunks)} 个文本块")

    # ----------------- 向量库构建 -----------------
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        try:
            # 尝试加载现有集合
            temp_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            if temp_store._collection is not None:
                temp_store.delete_collection()
                print(f"已删除旧集合，准备重建")
        except Exception as e:
            # 如果加载/删除失败，说明数据损坏或版本不兼容，直接删除整个目录
            print(f"清理旧向量库失败，删除目录: {e}")
            shutil.rmtree(persist_dir, ignore_errors=True)

    # 统一创建（目录不存在或已清理，都会在这里新建）
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

    # BM25
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 3

    # 向量检索器
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 混合检索
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    # 重排序模型
    # reranker_model = HuggingFaceCrossEncoder(
    #     model_name="BAAI/bge-reranker-base",
    #     model_kwargs={'device': 'cpu'}
    # )
    # compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

    model_path = r"D:\PycharmProjects\RAG\local_models\models--BAAI--bge-reranker-base\snapshots\2cfc18c9415c912f9d8155881c133215df768a70"

    reranker_model = HuggingFaceCrossEncoder(
        model_name=model_path,  # 本地路径，不会触发网络下载
        model_kwargs={'device': 'cpu'}
    )
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

    # 最终检索器
    final_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    # 为了方便后续动态调整 k，把子检索器也返回（或存储在 final_retriever 上）
    return final_retriever, bm25_retriever, vector_retriever, compressor