from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .config import MODEL_NAME, DASHSCOPE_API_KEY, BASE_URL

# Prompt 模板（可单独定义）
RAG_PROMPT = ChatPromptTemplate.from_template("""你是专业的文档问答助手，请严格基于以下参考资料回答用户的问题。
规则：
1. 只能使用参考资料中的信息，资料里没有的内容就直接说"参考资料中未提及相关内容"
2. 答案中引用的内容，用[1][2]标注对应的资料序号
3. 回答简洁准确，分点说明

参考资料：
{context}

用户问题：{question}""")

def create_llm():
    """创建并返回 LLM 实例"""
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        temperature=0.1,
    )

def rag_answer(query: str, retriever, bm25_retriever,
               vector_retriever, compressor, llm, k: int = 3):
    """
    执行 RAG 问答
    :param retriever: 最终的重排序检索器
    :param bm25_retriever: BM25检索器，用于动态设k
    :param vector_retriever: 向量检索器，用于动态设k
    :param compressor: 重排序压缩器，用于动态设top_n
    :param k: 返回文档数
    """
    # 动态调整
    bm25_retriever.k = k
    vector_retriever.search_kwargs["k"] = k
    compressor.top_n = k

    docs = retriever.invoke(query)

    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"[{i}] {doc.page_content}")
        sources.append({
            "序号": i,
            "文件": doc.metadata.get("source", "未知文件"),
            "页码": doc.metadata.get("page", "无页码"),
        })

    context = "\n".join(context_parts)
    chain = RAG_PROMPT | llm
    response = chain.invoke({"context": context, "question": query})

    return {
        "answer": response.content,
        "sources": sources,
    }