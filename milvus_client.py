"""
Milvus 向量数据库客户端

该模块提供了用于连接和操作Milvus向量数据库的客户端类。
支持集合管理、数据插入、向量搜索、查询和索引管理等功能。
"""

import os
import time
import warnings
from typing import List, Optional, Dict, Any, Union, Tuple
from dotenv import load_dotenv

from pymilvus import (
    MilvusClient as PyMilvusClient,
)
from pymilvus.exceptions import MilvusException


class MilvusClient:
    """Milvus向量数据库客户端

    用于连接和操作Milvus向量数据库，支持集合管理、数据插入、向量搜索等功能。
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        db_name: Optional[str] = None,
    ):
        """初始化Milvus客户端

        Args:
            uri: Milvus服务URI，如果未提供则从环境变量MILVUS_URI获取
            token: 认证令牌，如果未提供则从环境变量MILVUS_TOKEN获取
            timeout: 连接超时时间（秒），如果未提供则从环境变量REQUEST_TIMEOUT获取
            max_retries: 最大重试次数，如果未提供则从环境变量MAX_RETRIES获取
            db_name: 数据库名称，默认为"default"
        """
        # 加载环境变量
        load_dotenv()

        # 设置配置参数
        self.uri = uri or os.getenv("MILVUS_URI")
        self.token = token or os.getenv("MILVUS_TOKEN")
        self.timeout = timeout or int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.max_retries = max_retries or int(os.getenv("MAX_RETRIES", "3"))
        self.db_name = db_name or "default"

        # 验证必要参数
        if not self.uri:
            raise ValueError(
                "Milvus URI未设置，请在.env文件中设置MILVUS_URI或通过参数传入"
            )

        if not self.token:
            raise ValueError(
                "Milvus Token未设置，请在.env文件中设置MILVUS_TOKEN或通过参数传入"
            )

        # 初始化客户端连接
        self._client = None
        self._connection_alias = "default"
        self._ensure_connection()

    def _ensure_connection(self) -> None:
        """确保与Milvus服务的连接

        Raises:
            MilvusException: 当连接失败时
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # 使用新的MilvusClient连接方式
                self._client = PyMilvusClient(
                    uri=self.uri,
                    token=self.token,
                    db_name=self.db_name,
                    timeout=self.timeout,
                )

                # 测试连接
                self._client.list_collections()
                return

            except MilvusException as e:
                error_message = str(e).lower()

                if "authentication" in error_message or "token" in error_message:
                    raise MilvusException("认证失败：Token无效或已过期")
                elif "connection" in error_message or "network" in error_message:
                    last_exception = MilvusException(f"连接错误：{e}")
                elif "timeout" in error_message:
                    last_exception = MilvusException(f"连接超时：{e}")
                else:
                    last_exception = e

            except Exception as e:
                last_exception = MilvusException(f"连接失败：{e}")

            # 如果不是最后一次尝试，等待后重试
            if attempt < self.max_retries:
                wait_time = 2**attempt  # 指数退避
                warnings.warn(
                    f"连接失败，{wait_time}秒后重试 (尝试 {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(wait_time)

        # 所有重试都失败了，抛出最后一个异常
        raise last_exception or MilvusException("连接失败：未知错误")

    def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息

        Returns:
            包含连接信息的字典
        """
        return {
            "uri": self.uri,
            "db_name": self.db_name,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "connected": self._client is not None,
        }

    def close_connection(self) -> None:
        """关闭连接"""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                warnings.warn(f"关闭连接时出现警告：{e}")
            finally:
                self._client = None

    def flush_collection(self, collection_name: str) -> None:
        """刷新集合，确保数据持久化

        Args:
            collection_name: 集合名称

        Raises:
            ValueError: 当集合名称无效时
            MilvusException: 当刷新失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 刷新集合数据 - 使用正确的方法签名
            self._client.flush(collection_name)

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"刷新集合失败：{e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close_connection()

    def create_collection(
        self,
        name: str,
        schema: Optional[Dict[str, Any]] = None,
        dimension: Optional[int] = None,
        **kwargs,
    ) -> bool:
        """创建集合

        Args:
            name: 集合名称
            schema: 集合模式定义，如果未提供则使用默认模式
            dimension: 向量维度，当使用默认模式时必须提供
            **kwargs: 其他创建参数

        Returns:
            创建成功返回True

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当创建失败时
        """
        if not name or not name.strip():
            raise ValueError("集合名称不能为空")

        try:
            # 检查集合是否已存在
            if self.has_collection(name):
                warnings.warn(f"集合 '{name}' 已存在")
                return True

            # 如果提供了自定义schema，使用自定义schema
            if schema:
                self._client.create_collection(
                    collection_name=name, schema=schema, **kwargs
                )
            else:
                # 使用默认schema创建集合
                if not dimension:
                    raise ValueError("使用默认模式时必须提供向量维度")

                self._client.create_collection(
                    collection_name=name, dimension=dimension, **kwargs
                )

            return True

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"创建集合失败：{e}")

    def drop_collection(self, name: str) -> bool:
        """删除集合

        Args:
            name: 集合名称

        Returns:
            删除成功返回True

        Raises:
            ValueError: 当集合名称无效时
            MilvusException: 当删除失败时
        """
        if not name or not name.strip():
            raise ValueError("集合名称不能为空")

        try:
            if not self.has_collection(name):
                warnings.warn(f"集合 '{name}' 不存在")
                return True

            self._client.drop_collection(collection_name=name)
            return True

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"删除集合失败：{e}")

    def has_collection(self, name: str) -> bool:
        """检查集合是否存在

        Args:
            name: 集合名称

        Returns:
            存在返回True，否则返回False

        Raises:
            ValueError: 当集合名称无效时
            MilvusException: 当检查失败时
        """
        if not name or not name.strip():
            raise ValueError("集合名称不能为空")

        try:
            return self._client.has_collection(collection_name=name)

        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise MilvusException(f"检查集合存在性失败：{e}")

    def list_collections(self) -> List[str]:
        """列出所有集合

        Returns:
            集合名称列表

        Raises:
            MilvusException: 当列出集合失败时
        """
        try:
            return self._client.list_collections()

        except Exception as e:
            raise MilvusException(f"列出集合失败：{e}")

    def get_collection_info(self, name: str) -> Dict[str, Any]:
        """获取集合信息

        Args:
            name: 集合名称

        Returns:
            包含集合信息的字典

        Raises:
            ValueError: 当集合名称无效时
            MilvusException: 当获取信息失败时
        """
        if not name or not name.strip():
            raise ValueError("集合名称不能为空")

        try:
            if not self.has_collection(name):
                raise ValueError(f"集合 '{name}' 不存在")

            # 获取集合基本信息
            info = {"name": name, "exists": True}

            # 尝试获取更多详细信息
            try:
                # 获取集合统计信息
                stats = self._client.get_collection_stats(collection_name=name)
                info.update(stats)
            except Exception:
                # 如果获取统计信息失败，忽略但继续
                pass

            return info

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"获取集合信息失败：{e}")

    # ==================== 索引管理功能 ====================

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_params: Dict[str, Any],
        index_name: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """创建索引

        Args:
            collection_name: 集合名称
            field_name: 字段名称
            index_params: 索引参数，包含index_type和metric_type等
            index_name: 索引名称，如果未提供则自动生成
            **kwargs: 其他索引创建参数

        Returns:
            创建成功返回True

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当创建失败时

        Examples:
            # 创建向量索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 1024}
            }
            client.create_index("my_collection", "vector", index_params)

            # 创建标量索引
            index_params = {
                "index_type": "STL_SORT"
            }
            client.create_index("my_collection", "category", index_params)
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not field_name or not field_name.strip():
            raise ValueError("字段名称不能为空")

        if not index_params or not isinstance(index_params, dict):
            raise ValueError("索引参数不能为空且必须是字典类型")

        if "index_type" not in index_params:
            raise ValueError("索引参数必须包含index_type")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 检查是否已存在索引
            if self.has_index(collection_name, field_name):
                warnings.warn(f"字段 '{field_name}' 已存在索引")
                return True

            # 使用新的PyMilvus客户端API创建索引
            # 准备索引参数
            prepared_index_params = self._client.prepare_index_params()

            # 添加索引配置
            prepared_index_params.add_index(
                field_name=field_name,
                index_type=index_params.get("index_type"),
                metric_type=index_params.get("metric_type", "L2"),
                index_name=index_name or f"{field_name}_index",
                params=index_params.get("params", {}),
            )

            # 创建索引
            self._client.create_index(
                collection_name=collection_name,
                index_params=prepared_index_params,
                **kwargs,
            )

            return True

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"创建索引失败：{e}")

    def drop_index(
        self,
        collection_name: str,
        field_name: str,
        index_name: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """删除索引

        Args:
            collection_name: 集合名称
            field_name: 字段名称
            index_name: 索引名称，如果未提供则删除该字段的所有索引
            **kwargs: 其他删除参数

        Returns:
            删除成功返回True

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当删除失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not field_name or not field_name.strip():
            raise ValueError("字段名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 检查索引是否存在
            if not self.has_index(collection_name, field_name):
                warnings.warn(f"字段 '{field_name}' 不存在索引")
                return True

            # 删除索引
            if index_name:
                self._client.drop_index(
                    collection_name=collection_name, index_name=index_name, **kwargs
                )
            else:
                # 如果没有指定索引名称，删除字段的默认索引
                self._client.drop_index(
                    collection_name=collection_name,
                    index_name=f"{field_name}_index",
                    **kwargs,
                )

            return True

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"删除索引失败：{e}")

    def has_index(
        self, collection_name: str, field_name: str, index_name: Optional[str] = None
    ) -> bool:
        """检查索引是否存在

        Args:
            collection_name: 集合名称
            field_name: 字段名称
            index_name: 索引名称，如果未提供则检查该字段是否有任何索引

        Returns:
            存在返回True，否则返回False

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当检查失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not field_name or not field_name.strip():
            raise ValueError("字段名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                return False

            # 获取索引列表
            try:
                indexes = self._client.list_indexes(collection_name=collection_name)

                if not indexes:
                    return False

                # 如果指定了索引名称，直接检查
                if index_name:
                    return index_name in indexes

                # 如果没有指定索引名称，检查是否有该字段的索引
                # 尝试默认索引名称
                default_index_name = f"{field_name}_index"
                if default_index_name in indexes:
                    return True

                # 检查所有索引，看是否有针对该字段的
                for idx_name in indexes:
                    try:
                        idx_info = self._client.describe_index(
                            collection_name=collection_name, index_name=idx_name
                        )
                        if (
                            isinstance(idx_info, dict)
                            and idx_info.get("field_name") == field_name
                        ):
                            return True
                    except Exception:
                        continue

                return False

            except Exception:
                return False

        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise MilvusException(f"检查索引存在性失败：{e}")

    def list_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """列出集合的所有索引

        Args:
            collection_name: 集合名称

        Returns:
            索引信息列表，每个元素包含索引的详细信息

        Raises:
            ValueError: 当集合名称无效时
            MilvusException: 当列出索引失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 获取索引列表
            indexes = self._client.list_indexes(collection_name=collection_name)

            # 处理索引信息
            index_list = []
            if indexes:
                for index_name in indexes:
                    try:
                        # 获取每个索引的详细信息
                        index_info = self._client.describe_index(
                            collection_name=collection_name, index_name=index_name
                        )

                        # 处理索引信息
                        if isinstance(index_info, dict):
                            index_list.append(
                                {
                                    "index_name": index_name,
                                    "field_name": index_info.get(
                                        "field_name", "unknown"
                                    ),
                                    "index_type": index_info.get(
                                        "index_type", "unknown"
                                    ),
                                    "metric_type": index_info.get(
                                        "metric_type", "unknown"
                                    ),
                                    "params": index_info.get("params", {}),
                                    "index_state": index_info.get("state", "unknown"),
                                }
                            )
                        else:
                            index_list.append(
                                {
                                    "index_name": index_name,
                                    "field_name": "unknown",
                                    "index_type": "unknown",
                                    "metric_type": "unknown",
                                    "params": {},
                                    "index_state": "unknown",
                                }
                            )
                    except Exception:
                        # 如果获取某个索引信息失败，跳过但继续处理其他索引
                        index_list.append(
                            {
                                "index_name": index_name,
                                "field_name": "unknown",
                                "index_type": "unknown",
                                "metric_type": "unknown",
                                "params": {},
                                "index_state": "unknown",
                            }
                        )

            return index_list

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"列出索引失败：{e}")

    def get_index_info(self, collection_name: str, index_name: str) -> Dict[str, Any]:
        """获取索引详细信息

        Args:
            collection_name: 集合名称
            index_name: 索引名称

        Returns:
            包含索引详细信息的字典

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当获取信息失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not index_name or not index_name.strip():
            raise ValueError("索引名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 获取索引信息
            index_info = self._client.describe_index(
                collection_name=collection_name, index_name=index_name
            )

            # 处理返回的索引信息
            if isinstance(index_info, dict):
                return {
                    "index_name": index_name,
                    "field_name": index_info.get("field_name", "unknown"),
                    "index_type": index_info.get("index_type", "unknown"),
                    "metric_type": index_info.get("metric_type", "unknown"),
                    "params": index_info.get("params", {}),
                    "index_state": index_info.get("state", "unknown"),
                }
            else:
                return {
                    "index_name": index_name,
                    "field_name": "unknown",
                    "index_type": "unknown",
                    "metric_type": "unknown",
                    "params": {},
                    "index_state": "unknown",
                }

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"获取索引信息失败：{e}")

    def create_vector_index(
        self,
        collection_name: str,
        field_name: str = "vector",
        index_type: str = "IVF_FLAT",
        metric_type: str = "L2",
        index_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> bool:
        """创建向量索引的便捷方法

        Args:
            collection_name: 集合名称
            field_name: 向量字段名称，默认为"vector"
            index_type: 索引类型，支持IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, FLAT等
            metric_type: 距离度量类型，支持L2, IP, COSINE等
            index_params: 索引参数，如果未提供则使用默认参数
            **kwargs: 其他创建参数

        Returns:
            创建成功返回True

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当创建失败时
        """
        # 设置默认索引参数
        if index_params is None:
            index_params = self._get_default_index_params(index_type)

        # 构建完整的索引参数
        full_index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": index_params,
        }

        return self.create_index(
            collection_name=collection_name,
            field_name=field_name,
            index_params=full_index_params,
            **kwargs,
        )

    def create_scalar_index(
        self,
        collection_name: str,
        field_name: str,
        index_type: str = "STL_SORT",
        **kwargs,
    ) -> bool:
        """创建标量索引的便捷方法

        Args:
            collection_name: 集合名称
            field_name: 标量字段名称
            index_type: 索引类型，支持STL_SORT, Trie, INVERTED等
            **kwargs: 其他创建参数

        Returns:
            创建成功返回True

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当创建失败时
        """
        # 构建标量索引参数
        index_params = {"index_type": index_type}

        return self.create_index(
            collection_name=collection_name,
            field_name=field_name,
            index_params=index_params,
            **kwargs,
        )

    def _get_default_index_params(self, index_type: str) -> Dict[str, Any]:
        """获取默认索引参数

        Args:
            index_type: 索引类型

        Returns:
            默认索引参数字典
        """
        default_params = {
            "IVF_FLAT": {"nlist": 1024},
            "IVF_SQ8": {"nlist": 1024},
            "IVF_PQ": {"nlist": 1024, "m": 16, "nbits": 8},
            "HNSW": {"M": 16, "efConstruction": 200},
            "FLAT": {},
            "AUTOINDEX": {},
        }

        return default_params.get(index_type.upper(), {})

    def get_index_build_progress(
        self, collection_name: str, index_name: str
    ) -> Dict[str, Any]:
        """获取索引构建进度

        Args:
            collection_name: 集合名称
            index_name: 索引名称

        Returns:
            包含构建进度信息的字典

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当获取进度失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not index_name or not index_name.strip():
            raise ValueError("索引名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 获取索引信息（包含构建状态）
            index_info = self._client.describe_index(
                collection_name=collection_name, index_name=index_name
            )

            if isinstance(index_info, dict):
                total_rows = index_info.get("total_rows", 0)
                indexed_rows = index_info.get("indexed_rows", 0)

                return {
                    "total_rows": total_rows,
                    "indexed_rows": indexed_rows,
                    "progress_percent": (
                        indexed_rows / max(total_rows, 1) * 100
                        if total_rows > 0
                        else 100
                    ),
                    "state": index_info.get("state", "unknown"),
                }
            else:
                return {
                    "total_rows": 0,
                    "indexed_rows": 0,
                    "progress_percent": 0,
                    "state": "unknown",
                }

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"获取索引构建进度失败：{e}")

    def wait_for_index(
        self, collection_name: str, index_name: str, timeout: int = 300
    ) -> bool:
        """等待索引构建完成

        Args:
            collection_name: 集合名称
            index_name: 索引名称
            timeout: 超时时间（秒）

        Returns:
            构建完成返回True，超时返回False

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当等待失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not index_name or not index_name.strip():
            raise ValueError("索引名称不能为空")

        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                progress = self.get_index_build_progress(collection_name, index_name)

                if progress.get("state") in ["Finished", "Completed"]:
                    return True
                elif progress.get("state") == "Failed":
                    raise MilvusException("索引构建失败")

                # 等待1秒后再次检查
                time.sleep(1)

            return False  # 超时

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"等待索引构建失败：{e}")

    def insert_data(
        self,
        collection_name: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """插入数据到集合

        Args:
            collection_name: 集合名称
            data: 要插入的数据，可以是单条记录（字典）或多条记录（字典列表）
            **kwargs: 其他插入参数

        Returns:
            包含插入结果的字典，包含插入的ID等信息

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当插入失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not data:
            raise ValueError("插入数据不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 确保数据是列表格式
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("数据必须是字典或字典列表")

            # 验证数据不为空
            if not data:
                raise ValueError("插入数据不能为空")

            # 执行插入操作
            result = self._client.insert(
                collection_name=collection_name, data=data, **kwargs
            )

            # 刷新集合以确保数据持久化
            try:
                self.flush_collection(collection_name)
            except Exception as e:
                warnings.warn(f"数据插入成功但刷新失败：{e}")

            return {
                "insert_count": len(data),
                "ids": result.get("ids", []) if isinstance(result, dict) else [],
                "success": True,
            }

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"插入数据失败：{e}")

    def upsert_data(
        self,
        collection_name: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """插入或更新数据到集合（upsert操作）

        Args:
            collection_name: 集合名称
            data: 要插入或更新的数据，可以是单条记录（字典）或多条记录（字典列表）
            **kwargs: 其他upsert参数

        Returns:
            包含upsert结果的字典

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当upsert失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not data:
            raise ValueError("upsert数据不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 确保数据是列表格式
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("数据必须是字典或字典列表")

            # 验证数据不为空
            if not data:
                raise ValueError("upsert数据不能为空")

            # 执行upsert操作
            result = self._client.upsert(
                collection_name=collection_name, data=data, **kwargs
            )

            # 刷新集合以确保数据持久化
            try:
                self.flush_collection(collection_name)
            except Exception as e:
                warnings.warn(f"数据upsert成功但刷新失败：{e}")

            return {
                "upsert_count": len(data),
                "ids": result.get("ids", []) if isinstance(result, dict) else [],
                "success": True,
            }

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"upsert数据失败：{e}")

    def update_data(
        self,
        collection_name: str,
        data: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """更新集合中的数据（使用upsert实现）

        Args:
            collection_name: 集合名称
            data: 要更新的数据，可以是单条记录（字典）或多条记录（字典列表）
            **kwargs: 其他更新参数

        Returns:
            包含更新结果的字典

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当更新失败时
        """
        # 更新操作实际上使用upsert实现
        result = self.upsert_data(collection_name, data, **kwargs)
        result["update_count"] = result.pop("upsert_count", 0)
        return result

    def delete_data(
        self,
        collection_name: str,
        ids: Union[List[Union[str, int]], str, int] = None,
        filter_expr: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """删除集合中的数据

        Args:
            collection_name: 集合名称
            ids: 要删除的记录ID，可以是单个ID或ID列表
            filter_expr: 删除条件表达式（如果不提供ids）
            **kwargs: 其他删除参数

        Returns:
            包含删除结果的字典

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当删除失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not ids and not filter_expr:
            raise ValueError("必须提供要删除的ID或过滤条件")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 如果提供了IDs，构建删除条件
            if ids:
                if isinstance(ids, (str, int)):
                    ids = [ids]
                elif not isinstance(ids, list):
                    raise ValueError("IDs必须是字符串、整数或它们的列表")

                # 构建ID过滤表达式
                if isinstance(ids[0], str):
                    id_list = "', '".join(str(id) for id in ids)
                    filter_expr = f"id in ['{id_list}']"
                else:
                    id_list = ", ".join(str(id) for id in ids)
                    filter_expr = f"id in [{id_list}]"

            # 执行删除操作
            result = self._client.delete(
                collection_name=collection_name, filter=filter_expr, **kwargs
            )

            # 刷新集合以确保删除操作持久化
            try:
                self.flush_collection(collection_name)
            except Exception as e:
                warnings.warn(f"数据删除成功但刷新失败：{e}")

            return {
                "delete_count": result.get("delete_count", 0)
                if isinstance(result, dict)
                else 0,
                "success": True,
            }

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"删除数据失败：{e}")

    def search_data(
        self,
        collection_name: str,
        data: Union[List[List[float]], List[float]],
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        metric_type: str = "L2",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """在集合中搜索向量

        Args:
            collection_name: 集合名称
            data: 查询向量，可以是单个向量或向量列表
            limit: 返回结果数量限制
            search_params: 搜索参数，如{"nprobe": 10}
            output_fields: 要返回的字段列表
            filter_expr: 过滤条件表达式
            metric_type: 距离度量类型 (L2, IP, COSINE等)
            **kwargs: 其他搜索参数

        Returns:
            搜索结果列表，每个结果包含id、distance和指定的输出字段

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当搜索失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not data:
            raise ValueError("查询向量不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 确保数据是正确的格式
            if isinstance(data[0], (int, float)):
                # 单个向量，转换为向量列表
                data = [data]

            # 设置默认搜索参数
            if search_params is None:
                search_params = {"metric_type": metric_type}
            else:
                search_params = dict(search_params)
                if "metric_type" not in search_params:
                    search_params["metric_type"] = metric_type

            # 执行搜索
            results = self._client.search(
                collection_name=collection_name,
                data=data,
                limit=limit,
                search_params=search_params,
                output_fields=output_fields,
                filter=filter_expr,
                **kwargs,
            )

            # 处理搜索结果，确保返回格式一致
            processed_results = []
            if results:
                for result_group in results:
                    if hasattr(result_group, "__iter__"):
                        for hit in result_group:
                            hit_dict = {
                                "id": getattr(hit, "id", None),
                                "distance": getattr(hit, "distance", None),
                                "score": getattr(hit, "score", None),
                            }
                            # 添加其他字段
                            if hasattr(hit, "entity"):
                                hit_dict.update(hit.entity)
                            elif hasattr(hit, "fields"):
                                hit_dict.update(hit.fields)
                            processed_results.append(hit_dict)
                    else:
                        # 单个结果的情况
                        hit_dict = {
                            "id": getattr(result_group, "id", None),
                            "distance": getattr(result_group, "distance", None),
                            "score": getattr(result_group, "score", None),
                        }
                        if hasattr(result_group, "entity"):
                            hit_dict.update(result_group.entity)
                        elif hasattr(result_group, "fields"):
                            hit_dict.update(result_group.fields)
                        processed_results.append(hit_dict)

            return processed_results

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"搜索数据失败：{e}")

    def search_by_text(
        self,
        collection_name: str,
        text: Union[str, List[str]],
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        metric_type: str = "COSINE",
        embedding_client=None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """通过文本搜索向量（自动进行文本嵌入）

        Args:
            collection_name: 集合名称
            text: 查询文本，可以是单个文本或文本列表
            limit: 返回结果数量限制
            search_params: 搜索参数
            output_fields: 要返回的字段列表
            filter_expr: 过滤条件表达式
            metric_type: 距离度量类型
            embedding_client: 嵌入客户端，如果未提供则使用默认的
            **kwargs: 其他搜索参数

        Returns:
            搜索结果列表

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当搜索失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        if not text:
            raise ValueError("查询文本不能为空")

        try:
            # 导入嵌入客户端
            if embedding_client is None:
                try:
                    from embedding import embedding_service

                    embedding_client = embedding_service
                except ImportError:
                    raise MilvusException("无法导入嵌入服务，请确保embedding模块可用")

            # 将文本转换为向量
            if isinstance(text, str):
                vectors = [embedding_client.embed_text(text)]
            else:
                vectors = embedding_client.embed_texts(text)

            # 使用向量搜索
            return self.search_data(
                collection_name=collection_name,
                data=vectors,
                limit=limit,
                search_params=search_params,
                output_fields=output_fields,
                filter_expr=filter_expr,
                metric_type=metric_type,
                **kwargs,
            )

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"文本搜索失败：{e}")

    def search_similar(
        self,
        collection_name: str,
        data: Union[List[List[float]], List[float]],
        similarity_threshold: float = 0.8,
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        metric_type: str = "COSINE",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """相似度搜索，返回相似度高于阈值的结果

        Args:
            collection_name: 集合名称
            data: 查询向量
            similarity_threshold: 相似度阈值 (0-1)
            limit: 返回结果数量限制
            search_params: 搜索参数
            output_fields: 要返回的字段列表
            filter_expr: 过滤条件表达式
            metric_type: 距离度量类型，推荐使用COSINE
            **kwargs: 其他搜索参数

        Returns:
            过滤后的搜索结果列表，只包含相似度高于阈值的结果

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当搜索失败时
        """
        if not 0 <= similarity_threshold <= 1:
            raise ValueError("相似度阈值必须在0-1之间")

        try:
            # 执行搜索
            results = self.search_data(
                collection_name=collection_name,
                data=data,
                limit=limit * 2,  # 获取更多结果以便过滤
                search_params=search_params,
                output_fields=output_fields,
                filter_expr=filter_expr,
                metric_type=metric_type,
                **kwargs,
            )

            # 根据相似度阈值过滤结果
            filtered_results = []
            for result in results:
                similarity = self._calculate_similarity(
                    result.get("distance", 0), metric_type
                )
                if similarity >= similarity_threshold:
                    result["similarity"] = similarity
                    filtered_results.append(result)

            # 按相似度排序并限制结果数量
            filtered_results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            return filtered_results[:limit]

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"相似度搜索失败：{e}")

    def _calculate_similarity(self, distance: float, metric_type: str) -> float:
        """根据距离和度量类型计算相似度

        Args:
            distance: 距离值
            metric_type: 度量类型

        Returns:
            相似度值 (0-1)
        """
        if metric_type.upper() == "COSINE":
            # 余弦距离转相似度: similarity = 1 - distance
            return max(0, 1 - distance)
        elif metric_type.upper() == "IP":
            # 内积：值越大越相似，需要归一化
            return max(0, min(1, distance))
        elif metric_type.upper() == "L2":
            # 欧几里得距离：距离越小越相似
            return 1 / (1 + distance)
        else:
            # 默认处理
            return max(0, 1 - distance)

    def query_data(
        self,
        collection_name: str,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """查询集合中的数据

        Args:
            collection_name: 集合名称
            filter_expr: 过滤条件表达式
            output_fields: 要返回的字段列表
            limit: 返回结果数量限制
            offset: 结果偏移量
            **kwargs: 其他查询参数

        Returns:
            查询结果列表

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当查询失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 处理空过滤条件的情况
            if not filter_expr or filter_expr.strip() == "":
                # 当没有过滤条件时，必须提供limit参数
                if limit is None:
                    limit = 1000  # 设置默认限制，避免返回过多数据
                filter_expr = ""

            # 执行查询
            results = self._client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields,
                limit=limit,
                offset=offset,
                **kwargs,
            )

            return results

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"查询数据失败：{e}")

    def search_hybrid(
        self,
        collection_name: str,
        data: Union[List[List[float]], List[float]],
        scalar_filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        metric_type: str = "L2",
        rerank: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """混合搜索，结合向量搜索和标量过滤

        Args:
            collection_name: 集合名称
            data: 查询向量
            scalar_filters: 标量字段过滤条件，如{"category": "tech", "score": {"$gt": 0.5}}
            limit: 返回结果数量限制
            search_params: 搜索参数
            output_fields: 要返回的字段列表
            metric_type: 距离度量类型
            rerank: 是否对结果进行重新排序
            **kwargs: 其他搜索参数

        Returns:
            混合搜索结果列表

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当搜索失败时
        """
        try:
            # 构建过滤表达式
            filter_expr = self._build_filter_expression(scalar_filters)

            # 执行向量搜索
            results = self.search_data(
                collection_name=collection_name,
                data=data,
                limit=limit * 2 if rerank else limit,  # 如果需要重排序，获取更多结果
                search_params=search_params,
                output_fields=output_fields,
                filter_expr=filter_expr,
                metric_type=metric_type,
                **kwargs,
            )

            # 如果需要重新排序
            if rerank and results:
                results = self._rerank_results(results, data, metric_type)

            return results[:limit]

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"混合搜索失败：{e}")

    def search_range(
        self,
        collection_name: str,
        data: Union[List[List[float]], List[float]],
        distance_range: Tuple[float, float],
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        metric_type: str = "L2",
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """范围搜索，返回距离在指定范围内的结果

        Args:
            collection_name: 集合名称
            data: 查询向量
            distance_range: 距离范围 (min_distance, max_distance)
            limit: 返回结果数量限制
            search_params: 搜索参数
            output_fields: 要返回的字段列表
            filter_expr: 过滤条件表达式
            metric_type: 距离度量类型
            **kwargs: 其他搜索参数

        Returns:
            范围搜索结果列表

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当搜索失败时
        """
        min_distance, max_distance = distance_range
        if min_distance < 0 or max_distance < min_distance:
            raise ValueError("距离范围无效")

        try:
            # 执行搜索，获取更多结果以便过滤
            results = self.search_data(
                collection_name=collection_name,
                data=data,
                limit=limit * 3,  # 获取更多结果以便范围过滤
                search_params=search_params,
                output_fields=output_fields,
                filter_expr=filter_expr,
                metric_type=metric_type,
                **kwargs,
            )

            # 根据距离范围过滤结果
            filtered_results = []
            for result in results:
                distance = result.get("distance", float("inf"))
                if min_distance <= distance <= max_distance:
                    filtered_results.append(result)

            return filtered_results[:limit]

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"范围搜索失败：{e}")

    def batch_search(
        self,
        collection_name: str,
        data_list: List[Union[List[float], str]],
        limit: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        filter_expr: Optional[str] = None,
        metric_type: str = "L2",
        embedding_client=None,
        **kwargs,
    ) -> List[List[Dict[str, Any]]]:
        """批量搜索，支持多个查询向量或文本

        Args:
            collection_name: 集合名称
            data_list: 查询数据列表，可以是向量列表或文本列表
            limit: 每个查询返回的结果数量限制
            search_params: 搜索参数
            output_fields: 要返回的字段列表
            filter_expr: 过滤条件表达式
            metric_type: 距离度量类型
            embedding_client: 嵌入客户端（当查询数据是文本时需要）
            **kwargs: 其他搜索参数

        Returns:
            批量搜索结果列表，每个元素对应一个查询的结果

        Raises:
            ValueError: 当参数无效时
            MilvusException: 当搜索失败时
        """
        if not data_list:
            raise ValueError("查询数据列表不能为空")

        try:
            results = []

            # 检查数据类型
            if isinstance(data_list[0], str):
                # 文本查询
                if embedding_client is None:
                    try:
                        from embedding import embedding_service

                        embedding_client = embedding_service
                    except ImportError:
                        raise MilvusException(
                            "无法导入嵌入服务，请确保embedding模块可用"
                        )

                # 批量转换文本为向量
                vectors = embedding_client.embed_texts(data_list)
                data_list = vectors

            # 执行批量搜索
            search_results = self._client.search(
                collection_name=collection_name,
                data=data_list,
                limit=limit,
                search_params=search_params or {"metric_type": metric_type},
                output_fields=output_fields,
                filter=filter_expr,
                **kwargs,
            )

            # 处理搜索结果
            for result_group in search_results:
                processed_group = []
                if hasattr(result_group, "__iter__"):
                    for hit in result_group:
                        hit_dict = {
                            "id": getattr(hit, "id", None),
                            "distance": getattr(hit, "distance", None),
                            "score": getattr(hit, "score", None),
                        }
                        if hasattr(hit, "entity"):
                            hit_dict.update(hit.entity)
                        elif hasattr(hit, "fields"):
                            hit_dict.update(hit.fields)
                        processed_group.append(hit_dict)
                results.append(processed_group)

            return results

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"批量搜索失败：{e}")

    def _build_filter_expression(
        self, filters: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """构建过滤表达式

        Args:
            filters: 过滤条件字典

        Returns:
            过滤表达式字符串
        """
        if not filters:
            return None

        expressions = []
        for field, condition in filters.items():
            if isinstance(condition, dict):
                # 复杂条件，如 {"$gt": 0.5, "$lt": 1.0}
                for op, value in condition.items():
                    if op == "$gt":
                        expressions.append(f"{field} > {value}")
                    elif op == "$gte":
                        expressions.append(f"{field} >= {value}")
                    elif op == "$lt":
                        expressions.append(f"{field} < {value}")
                    elif op == "$lte":
                        expressions.append(f"{field} <= {value}")
                    elif op == "$eq":
                        if isinstance(value, str):
                            expressions.append(f"{field} == '{value}'")
                        else:
                            expressions.append(f"{field} == {value}")
                    elif op == "$ne":
                        if isinstance(value, str):
                            expressions.append(f"{field} != '{value}'")
                        else:
                            expressions.append(f"{field} != {value}")
                    elif op == "$in":
                        if isinstance(value[0], str):
                            value_list = "', '".join(str(v) for v in value)
                            expressions.append(f"{field} in ['{value_list}']")
                        else:
                            value_list = ", ".join(str(v) for v in value)
                            expressions.append(f"{field} in [{value_list}]")
            else:
                # 简单条件，如 "category": "tech"
                if isinstance(condition, str):
                    expressions.append(f"{field} == '{condition}'")
                else:
                    expressions.append(f"{field} == {condition}")

        return " and ".join(expressions) if expressions else None

    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        query_vector: Union[List[float], List[List[float]]],
        metric_type: str,
    ) -> List[Dict[str, Any]]:
        """重新排序搜索结果

        Args:
            results: 搜索结果
            query_vector: 查询向量（预留用于未来的高级重排序算法）
            metric_type: 度量类型

        Returns:
            重新排序后的结果
        """
        # 这里可以实现更复杂的重排序逻辑
        # 目前简单按距离排序，未来可以使用query_vector进行更精确的重排序
        _ = query_vector  # 标记参数已使用，避免警告

        if metric_type.upper() in ["COSINE", "IP"]:
            # 对于余弦相似度和内积，距离越大越相似
            results.sort(key=lambda x: x.get("distance", 0), reverse=True)
        else:
            # 对于欧几里得距离，距离越小越相似
            results.sort(key=lambda x: x.get("distance", float("inf")))

        return results

    def get_data_count(self, collection_name: str) -> int:
        """获取集合中的数据数量

        Args:
            collection_name: 集合名称

        Returns:
            数据数量

        Raises:
            ValueError: 当集合名称无效时
            MilvusException: 当获取数量失败时
        """
        if not collection_name or not collection_name.strip():
            raise ValueError("集合名称不能为空")

        try:
            # 检查集合是否存在
            if not self.has_collection(collection_name):
                raise ValueError(f"集合 '{collection_name}' 不存在")

            # 尝试多种方法获取数据数量
            try:
                # 方法1: 使用get_collection_stats
                stats = self._client.get_collection_stats(
                    collection_name=collection_name
                )
                if isinstance(stats, dict):
                    # 尝试不同的键名
                    for key in ["row_count", "num_entities", "count", "total"]:
                        if key in stats:
                            return int(stats[key])

                # 如果统计信息中没有找到计数，尝试查询方法
                results = self._client.query(
                    collection_name=collection_name,
                    filter="",
                    output_fields=["id"],
                    limit=100000,  # 设置一个较大的限制
                )
                return len(results) if results else 0

            except Exception:
                # 如果统计信息获取失败，尝试查询方法
                try:
                    results = self._client.query(
                        collection_name=collection_name,
                        filter="",
                        output_fields=["id"],
                        limit=100000,
                    )
                    return len(results) if results else 0
                except Exception:
                    return 0

        except Exception as e:
            if isinstance(e, (ValueError, MilvusException)):
                raise e
            raise MilvusException(f"获取数据数量失败：{e}")

    def __del__(self):
        """析构函数，确保连接被正确关闭"""
        self.close_connection()


# 创建全局服务实例，便于直接导入使用
milvus_service = MilvusClient()
