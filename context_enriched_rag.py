from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service
import hashlib
from typing import List, Dict, Any, Optional


class ContextEnrichedRAG:
    """
    上下文增强RAG（检索增强生成）系统

    该类实现了基于上下文增强的文档检索和问答功能。
    通过在最相似的文本块周围添加上下文信息来提高检索质量，
    从而生成更准确和连贯的答案。使用Milvus进行向量存储。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200,
        batch_size: int = 64,
        default_context_size: int = 1,
        system_prompt: Optional[str] = None,
        collection_name: str = "RAG_learn",
    ):
        """
        初始化上下文增强RAG系统

        Args:
            chunk_size: 每个文本块的最大长度，默认1000
            overlap: 相邻块之间的重叠长度，默认200
            batch_size: 嵌入向量批处理大小，默认64
            default_context_size: 默认上下文窗口大小，默认1
            system_prompt: 系统提示词，如果为None则使用默认提示词
            collection_name: Milvus集合名称
        """
        self.llm_client = GeminiLLM()
        self.file_reader = file_reader_service
        self.embedding_client = embedding_service
        self.milvus_client = milvus_service
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.default_context_size = default_context_size
        self.collection_name = collection_name

        # 设置默认系统提示词
        self.system_prompt = system_prompt or (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )

        # 存储处理后的数据
        self.text_chunks: List[str] = []
        self.original_text: str = ""

    def chunk_text(self, text: str) -> List[str]:
        """
        对文本进行分块处理

        Args:
            text: 输入的文本字符串

        Returns:
            分块后的文本列表
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        if self.overlap >= self.chunk_size:
            raise ValueError("重叠长度不能大于或等于块大小")

        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():  # 只添加非空块
                chunks.append(chunk)

        return chunks

    def _batch_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本的嵌入向量"""
        if not texts:
            raise ValueError("文本列表不能为空")
        
        try:
            # 如果文本数量超过批处理大小，分批处理
            if len(texts) <= self.batch_size:
                return self.embedding_client.embed_texts(texts)
            
            # 分批处理大量文本
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.embedding_client.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            raise Exception(f"嵌入向量生成失败: {e}")

    def context_enriched_search(
        self,
        query: str,
        context_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行上下文增强搜索，使用Milvus存储

        Args:
            query: 查询文本
            context_size: 上下文窗口大小，如果为None则使用默认值

        Returns:
            包含上下文的搜索结果列表
        """
        if not query or not query.strip():
            raise ValueError("查询文本不能为空")

        if context_size is None:
            context_size = self.default_context_size

        try:
            # 使用Milvus搜索最相似的文本块
            search_results = self.milvus_client.search_by_text(
                collection_name=self.collection_name,
                text=query,
                limit=1,  # 只获取最相似的一个结果
                output_fields=["text", "source", "chunk_index"],
                metric_type="COSINE",
                embedding_client=self.embedding_client
            )
            
            if not search_results:
                return []
            
            # 获取最相似文text块的索引
            top_result = search_results[0]
            # 正确访问Milvus搜索结果的结构
            top_index = top_result.get("entity", {}).get("chunk_index", 0) if "entity" in top_result else top_result.get("chunk_index", 0)
            
            # 查询相同源文件的所有文本块
            source = top_result.get("entity", {}).get("source", "") if "entity" in top_result else top_result.get("source", "")
            
            if not source:
                # 如果没有源信息，直接返回单个结果
                return [{"text": top_result.get("entity", {}).get("text", ""), "chunk_index": 0, "is_main": True}]
            all_chunks_results = self.milvus_client.query_data(
                collection_name=self.collection_name,
                filter_expr=f"source == '{source}'",
                output_fields=["text", "chunk_index"],
                limit=1000  # 设置一个较大的限制
            )
            
            # 按块索引排序
            all_chunks_results.sort(key=lambda x: x.get("entity", {}).get("chunk_index", 0) if "entity" in x else x.get("chunk_index", 0))
            
            # 计算上下文范围
            start_idx = max(0, top_index - context_size)
            end_idx = min(len(all_chunks_results), top_index + context_size + 1)
            
            # 返回包含上下文的结果
            context_results = []
            for i in range(start_idx, end_idx):
                if i < len(all_chunks_results):
                    chunk = all_chunks_results[i]
                    # 正确访问Milvus查询结果的结构
                    chunk_text = chunk.get("entity", {}).get("text", "") if "entity" in chunk else chunk.get("text", "")
                    chunk_index = chunk.get("entity", {}).get("chunk_index", i) if "entity" in chunk else chunk.get("chunk_index", i)
                    context_results.append({
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "is_main": chunk_index == top_index  # 标记主要结果
                    })
            
            return context_results
            
        except Exception as e:
            raise Exception(f"上下文增强搜索失败: {e}")

    def generate_response(self, user_message: str) -> str:
        """
        生成回答

        Args:
            user_message: 用户消息

        Returns:
            生成的回答
        """
        if not user_message or not user_message.strip():
            raise ValueError("用户消息不能为空")

        return self.llm_client.generate_text(user_message, system_instruction=self.system_prompt)

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档并存储到向量数据库

        Args:
            file_path: 文档文件路径
            
        Returns:
            处理结果字典
        """
        if not file_path or not file_path.strip():
            raise ValueError("文件路径不能为空")

        try:
            # 读取文档
            self.original_text = self.file_reader.read_file(file_path)

            # 分块处理
            self.text_chunks = self.chunk_text(self.original_text)
            
            if not self.text_chunks:
                raise ValueError("文档分块后没有有效内容")

            # 批量创建嵌入向量
            embeddings = self._batch_embed_texts(self.text_chunks)
            
            # 准备插入数据
            data_to_insert = []
            for i, (chunk, embedding) in enumerate(zip(self.text_chunks, embeddings)):
                chunk_id = self._generate_chunk_id(file_path, i)
                data_to_insert.append({
                    "id": chunk_id,
                    "vector": embedding,
                    "text": chunk,
                    "source": file_path,
                    "chunk_index": i
                })
            
            # 批量插入到Milvus
            result = self.milvus_client.insert_data(self.collection_name, data_to_insert)
            
            return {
                "success": True,
                "chunks_count": len(self.text_chunks),
                "file_path": file_path,
                **result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }

    def query(
        self,
        question: str,
        context_size: Optional[int] = None,
        include_context_info: bool = True,
    ) -> str:
        """
        对文档进行问答查询

        Args:
            question: 用户问题
            context_size: 上下文窗口大小，如果为None则使用默认值
            include_context_info: 是否在提示中包含上下文编号信息

        Returns:
            生成的回答
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        try:
            # 执行上下文增强搜索
            context_results = self.context_enriched_search(question, context_size=context_size)
            
            if not context_results:
                return "我没有找到相关的信息来回答这个问题。"

            # 构建上下文
            if include_context_info:
                context = "\n".join([
                    f"上下文内容{i + 1}{'(主要匹配)' if result.get('is_main') else ''}:\n{result['text']}"
                    for i, result in enumerate(context_results)
                ])
            else:
                context = "\n".join([result['text'] for result in context_results])

            # 构建用户提示
            user_prompt = f"{context}\n\n用户问题：{question}"

            # 生成回答
            return self.generate_response(user_prompt)
            
        except Exception as e:
            return f"查询处理失败: {e}"

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """生成文本块的唯一ID"""
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()


# 示例使用代码（如果直接运行此文件）
if __name__ == "__main__":
    # 创建上下文增强RAG实例
    rag = ContextEnrichedRAG()

    try:
        # 处理文档（取消注释以使用）
        result = rag.process_document("Agent基础.md")
        print(f"处理结果: {result}")

        # 进行查询
        query = "思维链(CoT)是什么？举个例子"
        response = rag.query(query)

        print(f"查询: {query}")
        print(f"回答: {response}")

    except Exception as e:
        print(f"错误: {e}")