# ENSO贝叶斯不对称性分析系统 - Docker容器化
# 基于Python 3.11的完整分析环境

# 使用官方Python 3.11 slim镜像作为基础镜像
FROM python:3.11-slim

# 设置维护者信息
LABEL maintainer="nasa-91"
LABEL description="ENSO Bayesian Asymmetry Analysis System"
LABEL version="2.0"

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目文件到容器中
COPY . .

# 创建数据目录和输出目录
RUN mkdir -p /app/data /app/outputs

# 暴露端口（如果将来需要Web界面）
EXPOSE 8080

# 默认命令：显示帮助信息
CMD ["python", "cli.py", "--help"]

# 使用说明：
#
# 构建镜像：
#   docker build -t enso-bayesian-analysis:v2.0 .
#
# 运行基本分析：
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs \
#          enso-bayesian-analysis:v2.0 \
#          python cli.py --data data/nino34.csv --output outputs/
#
# 运行交互式容器：
#   docker run -it -v $(pwd):/app enso-bayesian-analysis:v2.0 bash
#
# 查看日志：
#   docker logs <container_id>
#
# 进入运行中的容器：
#   docker exec -it <container_id> /bin/bash
