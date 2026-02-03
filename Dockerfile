# 使用官方 Python 3.13 镜像作为基础镜像
FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/TerraceCN/qwen3-asr-openai"
LABEL org.opencontainers.image.description="Convert Qwen3 ASR Bailian API to OpenAI-Compatible API."
LABEL org.opencontainers.image.licenses="MIT"

# 设置工作目录
WORKDIR /app

# 安装 uv 包管理器
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 使用 uv 安装依赖
RUN uv sync --frozen --no-dev

# 将 uv 创建的虚拟环境添加到 PATH
ENV PATH="/app/.venv/bin:$PATH"

# 拷贝项目文件
ADD . .

# 暴露端口
EXPOSE 8000

# 启动应用
CMD ["python3", "main.py"]