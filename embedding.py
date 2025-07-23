"""
硅基流动(SiliconFlow) 嵌入API客户端

该模块提供了用于调用硅基流动嵌入API的客户端类。
支持单个文本和批量文本的嵌入向量生成。
"""

import os
import time
from typing import List, Optional
import requests
from dotenv import load_dotenv


class EmbeddingClient:
    """硅基流动嵌入API客户端

    用于调用硅基流动的文本嵌入服务，支持单个文本和批量文本的嵌入向量生成。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """初始化嵌入客户端

        Args:
            api_key: API密钥，如果未提供则从环境变量SILICONFLOW_API_KEY获取
            base_url: API基础URL，如果未提供则从环境变量API_BASE_URL获取
            model: 嵌入模型名称，如果未提供则从环境变量EMBEDDING_MODEL获取
            timeout: 请求超时时间（秒），如果未提供则从环境变量REQUEST_TIMEOUT获取
            max_retries: 最大重试次数，如果未提供则从环境变量MAX_RETRIES获取
        """
        # 加载环境变量
        load_dotenv()

        # 设置配置参数
        self.api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        self.base_url = base_url or os.getenv(
            "API_BASE_URL", "https://api.siliconflow.cn/v1"
        )
        self.model = model or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.timeout = timeout or int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.max_retries = max_retries or int(os.getenv("MAX_RETRIES", "3"))

        # 验证必要参数
        if not self.api_key:
            raise ValueError(
                "API密钥未设置，请在.env文件中设置SILICONFLOW_API_KEY或通过参数传入"
            )

        # 设置请求头
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 构建API端点
        self.embeddings_url = f"{self.base_url.rstrip('/')}/embeddings"

    def embed_text(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量

        Args:
            text: 要生成嵌入向量的文本

        Returns:
            文本的嵌入向量列表

        Raises:
            ValueError: 当文本为空时
            requests.RequestException: 当API请求失败时
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")

        payload = {"model": self.model, "input": text}

        response_data = self._make_request(payload)

        # 提取嵌入向量
        if "data" in response_data and len(response_data["data"]) > 0:
            return response_data["data"][0]["embedding"]
        else:
            raise ValueError("API响应格式错误：未找到嵌入向量数据")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """生成多个文本的嵌入向量

        Args:
            texts: 要生成嵌入向量的文本列表

        Returns:
            文本嵌入向量的列表，每个元素对应一个输入文本的嵌入向量

        Raises:
            ValueError: 当文本列表为空时
            requests.RequestException: 当API请求失败时
        """
        if not texts:
            raise ValueError("文本列表不能为空")

        # 过滤空文本
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("文本列表中没有有效的文本")

        payload = {"model": self.model, "input": valid_texts}

        response_data = self._make_request(payload)

        # 提取嵌入向量
        if "data" in response_data:
            # 按索引排序确保顺序正确
            sorted_data = sorted(response_data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        else:
            raise ValueError("API响应格式错误：未找到嵌入向量数据")

    def _make_request(self, payload: dict) -> dict:
        """发送HTTP请求到嵌入API

        Args:
            payload: 请求负载

        Returns:
            API响应数据

        Raises:
            requests.RequestException: 当请求失败时
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.embeddings_url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )

                # 检查HTTP状态码
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    raise requests.RequestException(
                        f"认证失败：API密钥无效 (状态码: {response.status_code})"
                    )
                elif response.status_code == 429:
                    raise requests.RequestException(
                        f"请求频率限制：请稍后重试 (状态码: {response.status_code})"
                    )
                elif response.status_code >= 500:
                    raise requests.RequestException(
                        f"服务器错误：{response.status_code}"
                    )
                else:
                    raise requests.RequestException(
                        f"请求失败：{response.status_code} - {response.text}"
                    )

            except requests.exceptions.Timeout as e:
                last_exception = requests.RequestException(f"请求超时：{e}")
            except requests.exceptions.ConnectionError as e:
                last_exception = requests.RequestException(f"连接错误：{e}")
            except requests.exceptions.RequestException as e:
                # 对于认证错误等不需要重试的错误，直接抛出
                if "认证失败" in str(e):
                    raise e
                last_exception = e

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries:
                wait_time = 2**attempt  # 指数退避
                time.sleep(wait_time)

        # 所有重试都失败了，抛出最后一个异常
        raise last_exception or requests.RequestException("请求失败：未知错误")


embedding_service = EmbeddingClient()
