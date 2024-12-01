from flask import Flask, request, jsonify
from flask_cors import CORS  # 引入CORS
import json
import os
import prompt
from gpt_call import gpt_call
app = Flask(__name__)
CORS(app)  # 启用CORS，允许跨域请求
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_path_embeddings = os.getenv('MODEL_PATH_embeddings', '/data/yikun/LLM/bge-base-zh-v1.5')
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
local_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_path_embeddings,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
    query_instruction="为这个句子生成表示以用于检索相关文章："
)

cee_file = "cee.json"
with open(cee_file, 'r', encoding='utf-8') as f:
    cee = json.load(f)
def get_exception_branches(cee):
    branches = []
    for child in cee.get('children', []):
        if child['name'] == 'Error' or child['name'] == 'Exception':
            branches.extend(child.get('children', []))
    return branches
exception_branches = get_exception_branches(cee)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

loader=TextLoader(file_path=cee_file)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
print(all_splits)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
#构建向量数据库


# Prepare the scenario descriptions
scenario_property_desc = ""
for branch in exception_branches:
    scenario_property_desc += f"{branch['name']}: {branch.get('scenario', '')},{branch.get('property', '')}\n"


# 路由：返回欢迎信息
@app.route('/')
def hello():
    return "Hello, Welcome to Flask!"
@app.route('/echo', methods=['POST'])
def echo_string():
    data = request.get_json()  # 获取请求中的JSON数据
    #print(data)
    if not data or 'text' not in data:
        return jsonify({"error": "No 'text' field provided"}), 400
    code = data['text']
    detect_error = prompt.find_error_prompt.format(scenario=scenario_property_desc,code=code)
    print(detect_error)
    error_name = gpt_call("gpt-4", detect_error)
    print(error_name)
    if "None" in error_name:
        return jsonify({"echo": "代码并无异常"})

    docs = vectorstore.similarity_search(error_name)
    print(docs)
    explain_error = prompt.explain_error_prompt.format(scenario=docs,code=code,error_name=error_name)
    error_explain=gpt_call("gpt-4", explain_error)
    return jsonify({"echo": error_explain})
# 路由：返回JSON数据

# 路由：处理POST请求，接收JSON数据
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000,use_reloader=False)  # 启动服务器