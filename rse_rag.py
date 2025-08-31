"""
RSE RAG系统

该模块提供了基于RSE（Relevant Segment Extraction）技术的RAG（检索增强生成）系统。
通过计算文本块的相关性值并使用最大子数组算法找到最佳段落组合，
提升检索质量和回答准确性。
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from LLM import GeminiLLM
from file_reader import file_reader_service, FileReader
from embedding import embedding_service, EmbeddingClient
from milvus_client import milvus_service


@dataclass
class RSERAGConfig:
    """RSE RAG配置类

    用于配置RSE RAG系统的各项参数，包括文本分块、段落选择和生成参数。
    """

    chunk_size: int = 1000  # 文本块大小
    chunk_overlap: int = 200  # 块之间的重叠
    collection_name: str = "RAG_learn"  # 向量数据库集合名称
    irrelevant_chunk_penalty: float = 0.2  # 不相关块的惩罚值
    max_segment_length: int = 20  # 单个段落的最大长度
    total_max_length: int = 30  # 所有段落的最大总长度
    min_segment_value: float = 0.2  # 被考虑的段落的最小值
    system_prompt: str = """您是基于上下文智能应答的AI助手，需根据提供的文档段落回答用户问题。
这些文档段落是通过相关性检索匹配到当前问题的上下文内容。
请严格依据以下要求执行：
1. 整合分析所有相关段落信息
2. 生成全面准确的综合回答
3. 当上下文不包含有效信息时，必须明确告知无法回答"""


class RSERAG:
    """RSE RAG系统

    基于RSE（Relevant Segment Extraction）技术的RAG系统，
    通过计算文本块相关性值并使用最大子数组算法找到最佳段落组合。
    """

    def __init__(
        self,
        config: Optional[RSERAGConfig] = None,
        llm_client: Optional[GeminiLLM] = None,
        file_reader: Optional[FileReader] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        milvus_client=None,
    ):
        """初始化RSE RAG系统

        Args:
            config: RSE RAG配置，如果未提供则使用默认配置
            llm_client: LLM客户端实例，如果未提供则创建新实例
            file_reader: 文件读取器实例，如果未提供则使用全局服务
            embedding_client: 嵌入客户端实例，如果未提供则使用全局服务
            milvus_client: Milvus客户端实例，如果未提供则使用全局服务
        """
        # 配置参数
        self.config = config or RSERAGConfig()

        # 服务实例
        self.llm_client = llm_client or GeminiLLM()
        self.file_reader = file_reader or file_reader_service
        self.embedding_client = embedding_client or embedding_service
        self.milvus_client = milvus_client or milvus_service

        # 设置日志
        self.logger = logging.getLogger(__name__)

    def chunk_text(
        self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> List[str]:
        """将文本分割为块"""
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        step = chunk_size - overlap

        chunks = []
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def process_document(
        self, file_path: str, chunk_size: Optional[int] = None
    ) -> Tuple[List[str], Dict[str, Any]]:
        """处理文档以供RSE使用
        
        从文档中提取文本，分割文本块（无重叠）并创建向量存储，
        与网页版本的处理逻辑保持一致。
        
        Args:
            file_path: 文档路径 
            chunk_size: 每个块的字符大小，如果未指定则使用配置中的值
            
        Returns:
            包含块列表和文档信息的元组
        """
        self.logger.info("从文档中提取文本...")
        text = self.file_reader.read_file(file_path)
        
        self.logger.info("将文本切分为非重叠段落...")
        chunk_size = chunk_size or self.config.chunk_size
        # RSE使用无重叠分块 (overlap=0)
        chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=0)
        self.logger.info(f"创建了 {len(chunks)} 个块")

        self.logger.info("为块生成嵌入向量...")
        chunk_embeddings = self.embedding_client.embed_texts(chunks)
        
        # 添加带有元数据的文档（包括块索引以便后续重建）
        data_to_insert = [
            {
                "vector": embedding,
                "text": chunk,
                "metadata": {"chunk_index": i, "source": file_path},
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings))
        ]

        self.logger.info("存储到向量数据库中...")
        self.milvus_client.insert_data(self.config.collection_name, data_to_insert)
        
        # 跟踪原始文档结构以便段落重建
        doc_info = {
            "chunks": chunks,
            "source": file_path,
        }
        
        self.logger.info(f"文档处理完成，共处理 {len(chunks)} 个文本块")
        return chunks, doc_info

    def calculate_chunk_values(
        self,
        query: str,
        chunks: List[str],
        irrelevant_chunk_penalty: Optional[float] = None,
    ) -> List[float]:
        """计算文本块的相关性值
        
        通过结合相关性和位置计算块的值，与网页版本逻辑保持一致。
        """
        irrelevant_chunk_penalty = (
            irrelevant_chunk_penalty or self.config.irrelevant_chunk_penalty
        )
        
        self.logger.info("创建查询嵌入...")
        query_embedding = self.embedding_client.embed_text(query)

        self.logger.info("获取所有带有相似度分数的块...")
        results = self.milvus_client.search_data(
            self.config.collection_name,
            query_embedding,
            limit=len(chunks),
            output_fields=["text", "metadata"],
            metric_type="COSINE",
        )

        # 兼容不同的Milvus返回格式，创建从块索引到相关性分数的映射
        relevance_scores = {}
        for result in results:
            # 尝试不同的数据结构格式
            chunk_index = None
            similarity_score = 0.0
            
            if isinstance(result, dict):
                # 尝试获取chunk_index
                if "entity" in result and isinstance(result["entity"], dict):
                    # Milvus 2.x格式
                    metadata = result["entity"].get("metadata", {})
                    if isinstance(metadata, dict):
                        chunk_index = metadata.get("chunk_index")
                elif "metadata" in result:
                    # 简化格式
                    metadata = result["metadata"]
                    if isinstance(metadata, dict):
                        chunk_index = metadata.get("chunk_index")
                
                # 尝试获取相似度分数 (COSINE距离转换为相似度)
                if "score" in result:
                    # 如果是相似度分数，直接使用
                    similarity_score = result["score"]
                elif "distance" in result:
                    # 如果是距离，转换为相似度（对于COSINE，distance越小相似度越高）
                    # 相似度 = 1 - distance （假设distance在[0,2]范围内）
                    distance = result["distance"]
                    similarity_score = max(0.0, 1.0 - distance)
                
                if chunk_index is not None:
                    relevance_scores[chunk_index] = similarity_score

        self.logger.debug(f"获取到 {len(relevance_scores)} 个块的相关性分数")

        # 计算块值（相关性分数减去惩罚）
        chunk_values = []
        for i in range(len(chunks)):
            # 获取相关性分数，如果不在结果中则默认为0.0
            score = relevance_scores.get(i, 0.0)
            # 应用惩罚以将不相关的块转换为负值
            value = score - irrelevant_chunk_penalty
            chunk_values.append(value)

        return chunk_values

    def find_best_segments(
        self,
        chunk_values: List[float],
        max_segment_length: Optional[int] = None,
        total_max_length: Optional[int] = None,
        min_segment_value: Optional[float] = None,
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """使用最大子数组和算法的变体找到最佳段落

        Args:
            chunk_values: 每个块的值
            max_segment_length: 单个段落的最大长度，如果未提供则使用配置中的值
            total_max_length: 所有段落的最大总长度，如果未提供则使用配置中的值
            min_segment_value: 被考虑的段落的最小值，如果未提供则使用配置中的值

        Returns:
            包含最佳段落的（开始，结束）索引列表和对应分数的元组

        Raises:
            ValueError: 当块值列表为空或参数无效时
        """
        if not chunk_values:
            raise ValueError("块值列表不能为空")

        max_segment_length = max_segment_length or self.config.max_segment_length
        total_max_length = total_max_length or self.config.total_max_length
        min_segment_value = min_segment_value or self.config.min_segment_value

        if max_segment_length <= 0:
            raise ValueError("单个段落的最大长度必须大于0")
        if total_max_length <= 0:
            raise ValueError("所有段落的最大总长度必须大于0")

        self.logger.info("寻找最佳连续文本段落...")
        
        best_segments = []
        segment_scores = []
        total_included_chunks = 0

        # 继续寻找段落直到达到限制
        while total_included_chunks < total_max_length:
            best_score = min_segment_value  # 段落的最低阈值
            best_segment = None

            # 尝试每个可能的起始位置
            for start in range(len(chunk_values)):
                # 如果该起始位置已经在选定的段落中，则跳过(重叠内容部分)
                if any(start >= s[0] and start < s[1] for s in best_segments):
                    continue

                # 尝试每个可能的段落长度
                for length in range(
                    1, min(max_segment_length, len(chunk_values) - start) + 1
                ):
                    end = start + length

                    # 如果结束位置已经在选定的段落中，则跳过
                    if any(end > s[0] and end <= s[1] for s in best_segments):
                        continue

                    # 计算段落值为块值的总和
                    segment_value = sum(chunk_values[start:end])

                    # 如果这个段落更好，则更新最佳段落
                    if segment_value > best_score:
                        best_score = segment_value
                        best_segment = (start, end)

            # 如果找到了一个好的段落，则添加它
            if best_segment:
                best_segments.append(best_segment)
                segment_scores.append(best_score)
                total_included_chunks += best_segment[1] - best_segment[0]
                self.logger.info(f"找到段落 {best_segment}，得分 {best_score:.4f}")
            else:
                # 没有更多的好段落可找
                break

        # 按段落的起始位置排序以便于阅读
        best_segments.sort(key=lambda x: x[0])
        self.logger.info(f"共找到 {len(best_segments)} 个最佳段落")

        return best_segments, segment_scores

    def reconstruct_segments(
        self, chunks: List[str], best_segments: List[Tuple[int, int]]
    ) -> List[Dict[str, Any]]:
        """重构段落文本"""
        return [
            {"text": " ".join(chunks[start:end]), "segment_range": (start, end)}
            for start, end in best_segments
            if 0 <= start < end <= len(chunks)
        ]

    def format_segments_for_context(self, segments: List[Dict[str, Any]]) -> str:
        """格式化段落为上下文文本"""
        context = []
        for i, segment in enumerate(segments):
            start, end = segment["segment_range"]
            context.extend(
                [
                    f"分段{i + 1}（包含文本块{start}至{end - 1}）：",
                    segment["text"],
                    "-" * 80,
                ]
            )
        return "\n\n".join(context)

    def generate_response(
        self, query: str, context: str, system_prompt: Optional[str] = None
    ) -> str:
        """生成基于上下文的回答"""
        system_prompt = system_prompt or self.config.system_prompt
        user_prompt = f"""
        上下文内容：
        {context}

        问题：{query}

        请基于上述上下文内容提供专业可靠的回答。
        """

        response = self.llm_client.generate_text(
            user_prompt, system_instruction=system_prompt, temperature=0
        )
        return response.strip()

    def add_document(
        self, file_path: str, chunk_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """添加文档到向量数据库"""
        chunks, doc_info = self.process_document(file_path, chunk_size)
        return {
            "file_path": file_path,
            "chunks_count": len(chunks),
            "doc_info": doc_info,
            "status": "success",
        }

    def search_and_answer(
        self,
        query: str,
        source_file: Optional[str] = None,
        irrelevant_chunk_penalty: Optional[float] = None,
        max_segment_length: Optional[int] = None,
        total_max_length: Optional[int] = None,
        min_segment_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        """使用RSE技术进行搜索并生成回答
        
        完整的RSE RAG管道，包含相关段落提取（Relevant Segment Extraction）。
        """
        self.logger.info("\n=== 开始带有相关段落提取的RAG ===")
        self.logger.info(f"查询: {query}")
        
        # 从向量数据库中搜索相关文本块
        chunks = self._search_relevant_chunks(query, source_file)

        if not chunks:
            self.logger.warning("未找到相关的文本块")
            return {
                "query": query,
                "segments": [],
                "response": "抱歉，在数据库中未找到与您的问题相关的信息。",
                "chunks_found": 0,
            }

        # 根据查询计算相关性分数和块值
        self.logger.info("\n计算相关性分数和块值...")
        chunk_values = self.calculate_chunk_values(
            query, chunks, irrelevant_chunk_penalty
        )
        
        # 根据块值找到最佳文本段落
        best_segments, scores = self.find_best_segments(
            chunk_values, max_segment_length, total_max_length, min_segment_value
        )

        # 从最佳块中重建文本段落
        self.logger.info("\n从块中重建文本段落...")
        segments = self.reconstruct_segments(chunks, best_segments)
        
        # 将段落格式化为语言模型的上下文字符串
        context = self.format_segments_for_context(segments)
        
        # 使用上下文从语言模型生成响应
        self.logger.info("正在使用相关段落作为上下文生成响应...")
        response = self.generate_response(query, context)

        self.logger.info("\n=== 最终响应 ===")
        self.logger.info(response)

        return {
            "query": query,
            "segments": segments,
            "response": response,
            "chunks_found": len(chunks),
            "segment_scores": scores,
        }

    def _search_relevant_chunks(
        self, query: str, source_file: Optional[str] = None
    ) -> List[str]:
        """从向量数据库中搜索相关的文本块"""
        self.logger.info("创建查询嵌入并检索块...")
        query_embedding = self.embedding_client.embed_text(query)

        results = self.milvus_client.search_data(
            self.config.collection_name,
            query_embedding,
            limit=100,
            output_fields=["text", "metadata"],
            metric_type="COSINE"
        )

        # 兼容不同的Milvus返回格式，提取块索引和文本
        chunks_with_index = []
        for result in results:
            if isinstance(result, dict):
                chunk_index = None
                chunk_text = None
                
                # 尝试获取chunk_index和text
                if "entity" in result and isinstance(result["entity"], dict):
                    # Milvus 2.x格式
                    entity = result["entity"]
                    metadata = entity.get("metadata", {})
                    if isinstance(metadata, dict):
                        chunk_index = metadata.get("chunk_index")
                    chunk_text = entity.get("text", "")
                elif "metadata" in result and "text" in result:
                    # 简化格式
                    metadata = result["metadata"]
                    if isinstance(metadata, dict):
                        chunk_index = metadata.get("chunk_index")
                    chunk_text = result.get("text", "")
                
                # 只有当成功获取到索引和文本时才添加到结果中
                if chunk_index is not None and chunk_text:
                    chunks_with_index.append((chunk_index, chunk_text))

        # 按块索引排序以保持文档中的原始顺序
        chunks_with_index.sort(key=lambda x: x[0])
        chunk_texts = [chunk_text for _, chunk_text in chunks_with_index]
        
        self.logger.info(f"检索到 {len(chunk_texts)} 个相关文本块")
        return chunk_texts



# 创建全局服务实例，便于直接导入使用
rse_rag_service = RSERAG()


if __name__ == "__main__":
    # 使用示例
    rse_rag = RSERAG()

    # 分离式使用：先添加文档，再搜索（推荐）
    # rse_rag.add_document("Agent基础.md")
    result = rse_rag.search_and_answer("思维链(CoT)是什么？举个例子")
    print(f"回答: {result['response']}")

