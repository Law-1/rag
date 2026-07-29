import os
# 镜像源设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv

load_dotenv(override=True)



# 可以在这里统一读取所需环境变量，便于管理
MODEL_NAME = os.getenv("model_name")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("embedding_model_name")
BASE_URL = "https://llm-bo0uovnwd1ta06b3.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"