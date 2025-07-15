# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

# 导入系统和操作系统相关的库，用于路径操作和系统参数设置
import sys
import os

# 将上级目录加入系统路径，方便导入minirag模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入csv用于读写，tqdm用于进度条显示
import csv
from tqdm import trange
# 导入MiniRAG相关模块和函数
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    hf_model_complete,  # HuggingFace推理函数
    hf_embed,           # HuggingFace embedding函数
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

# 指定用于文本嵌入的模型
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse

# 解析命令行参数的函数
def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="PHI")  # 指定LLM模型
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")  # 输出路径
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")  # 工作目录
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")  # 数据集路径
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )  # 查询集路径
    args = parser.parse_args()
    return args

# 获取命令行参数
args = get_args()

# 根据参数选择不同的LLM模型
if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "qwen":
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
else:
    print("Invalid model name")
    exit(1)

# 读取各类路径参数
WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath
print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)

# 如果工作目录不存在则创建
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 初始化MiniRAG对象，配置LLM和Embedding
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,  # 使用HuggingFace模型推理
    # llm_model_func=gpt_4o_mini_complete,  # 可选：OpenAI GPT-4o
    llm_model_max_token_size=200,         # LLM最大token数
    llm_model_name=LLM_MODEL,             # LLM模型名称
    embedding_func=EmbeddingFunc(
        embedding_dim=384,                # 嵌入维度
        max_token_size=1000,              # 嵌入最大token数
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)

# 读取问题和标准答案列表
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])

# 运行实验并记录结果到输出文件
def run_experiment(output_path):
    headers = ["Question", "Gold Answer", "minirag"]  # CSV表头

    q_already = []  # 已经处理过的问题
    if os.path.exists(output_path):
        with open(output_path, mode="r", encoding="utf-8") as question_file:
            reader = csv.DictReader(question_file)
            for row in reader:
                q_already.append(row["Question"])

    row_count = len(q_already)
    print("row_count", row_count)

    with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if row_count == 0:
            writer.writerow(headers)  # 首次写入表头

        # 遍历所有未处理的问题，进行推理并写入结果
        for QUESTIONid in trange(row_count, len(QUESTION_LIST)):  #
            QUESTION = QUESTION_LIST[QUESTIONid]
            Gold_Answer = GA_LIST[QUESTIONid]
            print()
            print("QUESTION", QUESTION)
            print("Gold_Answer", Gold_Answer)

            try:
                minirag_answer = (
                    rag.query(QUESTION, param=QueryParam(mode="mini"))
                    .replace("\n", "")
                    .replace("\r", "")
                )  # 调用MiniRAG进行问答
            except Exception as e:
                print("Error in minirag_answer", e)
                minirag_answer = "Error"

            writer.writerow([QUESTION, Gold_Answer, minirag_answer])  # 写入结果

    print(f"Experiment data has been recorded in the file: {output_path}")


# if __name__ == "__main__":

run_experiment(OUTPUT_PATH)  # 直接运行实验
