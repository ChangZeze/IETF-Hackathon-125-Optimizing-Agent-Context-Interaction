"""
配置模块

负责初始化和配置OpenAI客户端，用于连接DeepSeek API。
包含：
- HTTP客户端配置（超时、SSL验证）
- OpenAI客户端实例
- API密钥管理
"""

import os
import httpx
from openai import OpenAI
http_client = httpx.Client(
    verify = False,
    timeout=60.0,
)

client = OpenAI(
    api_key = os.getenv("DEEPSEEK_API_KEY", "xxx"),
    base_url="https://api.deepseek.com/v1",
    http_client = http_client,
)