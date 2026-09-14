import os
import shutil
from contextlib import asynccontextmanager
from dotenv import load_dotenv
# 加载环境变量
load_dotenv(override=True)
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel


# 导入自定义模块
from rag_package import load_from_directory, build_retriever, rag_answer, load_documents_from_directory
from rag_package.chain import create_llm




# ---------- 全局状态 ----------
# 这些对象将在启动时初始化或上传文档后更新
retriever = None
bm25_retriever = None
vector_retriever = None
compressor = None
llm = None

# 上传目录和持久化目录
UPLOAD_DIR = "./docs"
PERSIST_DIR = "./chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- 启动事件 ----------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    启动时初始化向量库和检索器。
    如果 chroma_db 已存在，则直接从持久化目录加载向量库，
    并重新扫描 docs 目录下的所有 文件 以重建 BM25 和混合检索器。
    如果不存在，则等待用户上传 文件 后创建。
    """
    global retriever, bm25_retriever, vector_retriever, compressor, llm

    # 初始化 LLM（只需一次）
    llm = create_llm()

    # 检查是否已有持久化向量库和文档
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("检测到已存在的向量库，正在重建检索器...")
        # 重新加载所有 文件 文档以获取完整 chunks（BM25 需要）
        # all_docs = load_from_directory(UPLOAD_DIR)  # 会加载 docs 目录下的所有 文件
        all_docs = load_documents_from_directory(UPLOAD_DIR)
        if all_docs:
            retriever, bm25_retriever, vector_retriever, compressor = build_retriever(
                all_docs, persist_dir=PERSIST_DIR
            )
            print("检索器初始化完成。")
        else:
            print("警告：向量库存在但未找到任何 文件 文件，请上传文件。")
    else:
        print("未发现向量库，等待用户上传文档。")

    yield  # 服务运行期间

# 创建 FastAPI 实例时传入 lifespan
app = FastAPI(title="极简RAG问答系统", lifespan=lifespan)

# ---------- 请求体 ----------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

# ---------- 辅助函数：根据当前文档重建检索器 ----------
def rebuild_retriever():
    """重新加载 docs 目录下所有 文件 并构建检索器"""
    global retriever, bm25_retriever, vector_retriever, compressor
    # docs = load_from_directory(UPLOAD_DIR)
    docs = load_documents_from_directory(UPLOAD_DIR)
    if not docs:
        raise ValueError("没有可用的文档，请先上传 文件")
    retriever, bm25_retriever, vector_retriever, compressor = build_retriever(
        docs, persist_dir=PERSIST_DIR
    )

# ---------- API 接口 ----------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """接收前端上传的 文件，保存并更新检索器"""
    allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.csv', '.xlsx', '.pptx',
                          '.html', '.json', '.xml', '.rst'}

    # filename = file.filename.lower()
    # is_empty_name = not filename
    # is_support_ext = any(filename.endswith(ext) for ext in allowed_extensions)
    #
    # if is_empty_name or not is_support_ext:
    # return {"status": "error", "message": f"不支持的文件类型，允许: {', '.join(allowed_extensions)}"}

    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return {"status": "error", "message": f"不支持的文件类型，允许: {', '.join(allowed_extensions)}"}

    # 保存文件
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 重建检索器（包含新上传的文件）
    try:
        rebuild_retriever()
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/chat")
async def rag_chat(request: QueryRequest):
    """接收问题，返回答案和参考来源"""
    if retriever is None:
        return {"error": "系统尚未初始化，请先上传文档"}

    try:
        result = rag_answer(
            query=request.question,
            retriever=retriever,
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            compressor=compressor,
            llm=llm,
            k=request.top_k,
        )
        return result
    except Exception as e:
        return {"error": str(e)}

# 可选：健康检查
@app.get("/health")
def health():
    return {"status": "ok", "retriever_ready": retriever is not None}

# 111