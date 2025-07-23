# server/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import subprocess
import os
import threading

app = Flask(__name__)
CORS(app)  # 处理跨域问题

# 安全密钥（实际部署应从环境变量读取）
API_KEY = "your_secure_api_key_here"


@app.route('/api/submit', methods=['POST'])
def submit_job():
    # 身份验证
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # 接收数据和文件
    data = request.json
    files = request.files

    # 生成唯一任务ID
    job_id = str(uuid.uuid4())
    input_dir = f"/data/inputs/{job_id}"
    os.makedirs(input_dir)

    # 保存输入数据
    with open(f"{input_dir}/params.json", 'w') as f:
        json.dump(data, f)

    # 保存上传的文件
    for file_key in files:
        files[file_key].save(f"{input_dir}/{file_key}.bin")

    # 异步执行模型（实际生产环境应用Celery/RQ）
    def run_model():
        result = subprocess.run(
            ["python", "run_model.py", input_dir],
            capture_output=True,
            text=True
        )
        # 处理结果...（保存到数据库/文件系统）

    threading.Thread(target=run_model).start()

    return jsonify({"job_id": job_id})


@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    # 检查结果是否就绪
    result_file = f"/data/results/{job_id}.json"
    if not os.path.exists(result_file):
        return jsonify({"status": "processing"}), 202

    # 返回结果
    with open(result_file) as f:
        result = json.load(f)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')