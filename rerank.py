"""
硅基流动(SiliconFlow) 重排序API客户端

该模块提供了用于调用硅基流动重排序API的客户端类。
支持根据查询对文档列表进行相关性重排序。
"""

import os
import time
from typing import List, Optional, Dict, Any
import requests
from dotenv import load_dotenv


class RerankClient:
    """硅基流动重排序API客户端

    用于调用硅基流动的文档重排序服务，根据查询对文档列表进行相关性排序。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """初始化重排序客户端

        Args:
            api_key: API密钥，如果未提供则从环境变量SILICONFLOW_API_KEY获取
            base_url: API基础URL，如果未提供则从环境变量API_BASE_URL获取
            model: 重排序模型名称，如果未提供则从环境变量RERANK_MODEL获取
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
        self.model = model or os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
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
        self.rerank_url = f"{self.base_url.rstrip('/')}/rerank"

    def rerank(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """对文档列表进行重排序

        Args:
            query: 查询文本，用于计算与文档的相关性
            documents: 要重排序的文档列表
            top_k: 返回前k个最相关的文档，如果未指定则返回所有文档

        Returns:
            重排序后的文档列表，每个元素包含：
            - document: 包含原始文档文本的字典
            - index: 文档在原始列表中的索引
            - relevance_score: 相关性分数

        Raises:
            ValueError: 当查询或文档列表为空时
            requests.RequestException: 当API请求失败时
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        if not documents:
            raise ValueError("文档列表不能为空")

        # 过滤空文档
        valid_documents = [doc for doc in documents if doc and doc.strip()]
        if not valid_documents:
            raise ValueError("文档列表中没有有效的文档")

        # 构建请求负载
        payload = {"model": self.model, "query": query, "documents": valid_documents}

        # 如果指定了top_k，添加到请求中
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k必须大于0")
            payload["top_k"] = min(top_k, len(valid_documents))

        response_data = self._make_request(payload)

        # 提取重排序结果并添加文档文本
        if "results" in response_data:
            results = []
            api_results = response_data["results"]

            # 如果指定了top_k，在客户端进行截取
            if top_k is not None:
                api_results = api_results[:top_k]

            for result in api_results:
                # 添加文档文本到结果中
                enhanced_result = {
                    "document": {"text": valid_documents[result["index"]]},
                    "index": result["index"],
                    "relevance_score": result["relevance_score"],
                }
                results.append(enhanced_result)
            return results
        else:
            raise ValueError("API响应格式错误：未找到重排序结果")

    def rerank_with_scores(
        self, query: str, documents: List[str], top_k: Optional[int] = None
    ) -> List[tuple]:
        """对文档列表进行重排序并返回简化格式

        Args:
            query: 查询文本
            documents: 要重排序的文档列表
            top_k: 返回前k个最相关的文档

        Returns:
            元组列表，每个元组包含 (document_text, relevance_score, original_index)
        """
        results = self.rerank(query, documents, top_k)

        return [
            (result["document"]["text"], result["relevance_score"], result["index"])
            for result in results
        ]

    def _make_request(self, payload: dict) -> dict:
        """发送HTTP请求到重排序API

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
                    self.rerank_url,
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


rerank_service = RerankClient()