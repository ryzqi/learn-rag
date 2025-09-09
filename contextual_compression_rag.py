import hashlib
import logging
from typing import List, Optional, Dict, Any, Tuple
from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service


class ContextualCompressionRAG:
    """
    上下文压缩RAG（检索增强生成）系统
    
    该类实现了带有上下文压缩功能的文档检索和问答系统。
    通过在检索后、生成回答前对文档片段进行压缩，
    只保留与查询最相关的内容，提高回答质量并减少噪声。
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        overlap: int = 200, 
        batch_size: int = 64, 
        collection_name: str = "RAG_learn",
        compression_type: str = "selective"
    ):
        """
        初始化ContextualCompressionRAG类
        
        Args:
            chunk_size (int): 每个文本块的最大长度，默认1000
            overlap (int): 相邻块之间的重叠长度，默认200
            batch_size (int): 批处理大小，默认64
            collection_name (str): Milvus集合名称
            compression_type (str): 压缩类型 ("selective", "summary", "extraction")
        """
        self.llm_client = GeminiLLM()
        self.file_reader = file_reader_service
        self.embedding_client = embedding_service
        self.milvus_client = milvus_service
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.collection_name = collection_name
        self.compression_type = compression_type
        
        self.system_prompt = (
            "你是一个AI助手，严格根据给定的上下文进行回答。"
            "如果无法直接从提供的上下文中得出答案，请回复：'我没有足够的信息来回答这个问题。'"
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        将文本分割成重叠的块
        
        Args:
            text: 输入文本
            
        Returns:
            文本块列表
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")
            
        chunks = []
        step = self.chunk_size - self.overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + self.chunk_size]
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

    def compress_chunk(self, chunk: str, query: str, compression_type: Optional[str] = None) -> Tuple[str, float]:
        """
        压缩检索到的文本块，仅保留与查询相关的内容
        
        Args:
            chunk (str): 要压缩的文本块
            query (str): 用户查询
            compression_type (str): 压缩类型 ("selective", "summary", "extraction")
            
        Returns:
            Tuple[str, float]: 压缩后的文本块和压缩比率
        """
        if compression_type is None:
            compression_type = self.compression_type
            
        # 为不同的压缩方法定义系统提示
        if compression_type == "selective":
            system_prompt = """您是专业信息过滤专家。
            您的任务是分析文档块并仅提取与用户查询直接相关的句子或段落，移除所有无关内容。

            输出要求：
            1. 仅保留有助于回答查询的文本
            2. 保持相关句子的原始措辞（禁止改写）
            3. 维持文本的原始顺序
            4. 包含所有相关文本（即使存在重复）
            5. 排除任何与查询无关的文本

            请以纯文本格式输出，不添加任何注释。"""

        elif compression_type == "summary":
            system_prompt = """您是专业摘要生成专家。
            您的任务是创建文档块的简洁摘要，且仅聚焦与用户查询相关的信息。

            输出要求：
            1. 保持简明扼要但涵盖所有相关要素
            2. 仅聚焦与查询直接相关的信息
            3. 省略无关细节
            4. 使用中立、客观的陈述语气

            请以纯文本格式输出，不添加任何注释。"""

        else:  # extraction
            system_prompt = """您是精准信息提取专家。
            您的任务是从文档块中精确提取与用户查询相关的完整句子。

            输出要求：
            1. 仅包含原始文本中的直接引用
            2. 严格保持原始文本的措辞（禁止修改）
            3. 仅选择与查询直接相关的完整句子
            4. 不同句子使用换行符分隔
            5. 不添加任何解释性文字

            请以纯文本格式输出，不添加任何注释。"""

        # 定义带有查询和文档块的用户提示
        user_prompt = f"""
        查询: {query}

        文档块:
        {chunk}

        请严格提取与本查询相关的核心内容。
        """

        try:
            # 使用 LLM 生成响应
            response = self.llm_client.generate_text(
                user_prompt, 
                system_instruction=system_prompt
            )

            # 从响应中提取压缩后的文本块
            compressed_chunk = response.strip()

            # 计算压缩比率
            original_length = len(chunk)
            compressed_length = len(compressed_chunk)
            compression_ratio = (original_length - compressed_length) / original_length * 100 if original_length > 0 else 0

            return compressed_chunk, compression_ratio
        except Exception as e:
            # 如果压缩失败，返回原始块
            logging.warning(f"压缩失败，返回原始文本块: {e}")
            return chunk, 0.0

    def batch_compress_chunks(self, chunks: List[str], query: str, compression_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        批量压缩多个文本块
        
        Args:
            chunks (List[str]): 要压缩的文本块列表
            query (str): 用户查询
            compression_type (str): 压缩类型
            
        Returns:
            List[Tuple[str, float]]: 包含压缩比率的压缩文本块列表
        """
        if compression_type is None:
            compression_type = self.compression_type
            
        results = []
        total_original_length = 0
        total_compressed_length = 0

        # 遍历每个文本块
        for i, chunk in enumerate(chunks):
            logging.info(f"正在压缩文本块 {i+1}/{len(chunks)}...")
            # 压缩文本块并获取压缩后的文本块和压缩比率
            compressed_chunk, compression_ratio = self.compress_chunk(chunk, query, compression_type)
            results.append((compressed_chunk, compression_ratio))

            total_original_length += len(chunk)
            total_compressed_length += len(compressed_chunk)

        # 计算总体压缩比率
        overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100 if total_original_length > 0 else 0
        logging.info(f"总体压缩比率: {overall_ratio:.2f}%")

        return results

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        使用语义相似度搜索相关文档
        
        Args:
            query: 搜索查询文本
            limit: 返回的最大结果数
            
        Returns:
            搜索结果列表
        """
        if not query or not query.strip():
            raise ValueError("查询不能为空")
            
        try:
            return self.milvus_client.search_by_text(
                collection_name=self.collection_name,
                text=query,
                limit=limit,
                output_fields=["text", "source", "chunk_index"],
                metric_type="COSINE",
                embedding_client=self.embedding_client,
            )
        except Exception as e:
            raise Exception(f"搜索失败: {e}")

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理文档并存储到向量数据库
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理结果字典
        """
        try:
            # 读取文档并分块
            text = self.file_reader.read_file(file_path)
            text_chunks = self.chunk_text(text)

            if not text_chunks:
                raise ValueError("文档分块后没有有效内容")

            # 批量生成嵌入向量
            embeddings = self._batch_embed_texts(text_chunks)

            # 准备插入数据
            data_to_insert = []
            for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                chunk_id = self._generate_chunk_id(file_path, i)
                data_to_insert.append(
                    {
                        "id": chunk_id,
                        "vector": embedding,
                        "text": chunk,
                        "source": file_path,
                        "chunk_index": i,
                    }
                )

            # 批量插入到Milvus
            result = self.milvus_client.insert_data(
                self.collection_name, data_to_insert
            )

            return {
                "success": True,
                "chunks_count": len(text_chunks),
                "file_path": file_path,
                **result,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "file_path": file_path}

    def generate_response(self, query: str, context: str) -> str:
        """
        基于查询和上下文生成响应
        
        Args:
            query: 用户的问题
            context: 从搜索结果检索的上下文
            
        Returns:
            生成的响应文本
        """
        if not context or not context.strip():
            return "我没有找到相关的信息来回答这个问题。"

        user_prompt = f"上下文:\n{context}\n\n用户问题：{query}"
        return self.llm_client.generate_text(
            user_prompt, system_instruction=self.system_prompt
        )

    def query(self, question: str, limit: int = 10, compression_type: Optional[str] = None) -> Dict[str, Any]:
        """
        完整的查询流水线：搜索、压缩上下文和生成响应
        
        Args:
            question: 用户的问题
            limit: 最大搜索结果数
            compression_type: 压缩类型，如果不指定则使用实例默认值
            
        Returns:
            包含回答和压缩信息的结果字典
        """
        if not question or not question.strip():
            raise ValueError("问题不能为空")

        if compression_type is None:
            compression_type = self.compression_type

        try:
            # 1. 搜索相关文档
            search_results = self.search(question, limit)

            # 2. 提取文本块
            retrieved_chunks = []
            for result in search_results:
                text = (
                    result.get("entity", {}).get("text", "")
                    if "entity" in result
                    else result.get("text", "")
                )
                if text.strip():
                    retrieved_chunks.append(text)

            if not retrieved_chunks:
                return {
                    "question": question,
                    "answer": "我没有找到相关的信息来回答这个问题。",
                    "compression_info": {
                        "original_chunks": 0,
                        "compressed_chunks": 0,
                        "compression_ratio": 0.0
                    }
                }

            # 3. 对检索到的块应用压缩
            compressed_results = self.batch_compress_chunks(retrieved_chunks, question, compression_type)
            compressed_chunks = [result[0] for result in compressed_results]
            compression_ratios = [result[1] for result in compressed_results]

            # 4. 过滤掉任何空的压缩块
            filtered_chunks = [chunk for chunk in compressed_chunks if chunk.strip()]

            if not filtered_chunks:
                # 如果所有块都被压缩为空字符串，则使用原始块
                logging.warning("所有块都被压缩为空，使用原始块")
                filtered_chunks = retrieved_chunks
                compression_ratios = [0.0] * len(retrieved_chunks)

            # 5. 从压缩块生成上下文
            context = "\n\n---\n\n".join(filtered_chunks)

            # 6. 生成响应
            response = self.generate_response(question, context)

            # 7. 计算压缩统计信息
            avg_compression_ratio = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0

            return {
                "question": question,
                "answer": response,
                "compression_info": {
                    "compression_type": compression_type,
                    "original_chunks": len(retrieved_chunks),
                    "compressed_chunks": len(filtered_chunks),
                    "avg_compression_ratio": f"{avg_compression_ratio:.2f}%",
                    "original_context_length": sum(len(chunk) for chunk in retrieved_chunks),
                    "compressed_context_length": len(context)
                }
            }

        except Exception as e:
            return {
                "question": question,
                "answer": f"查询处理失败: {e}",
                "compression_info": {"error": str(e)}
            }

    def _generate_chunk_id(self, file_path: str, chunk_index: int) -> str:
        """生成文本块的唯一ID"""
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()


# 示例用法
if __name__ == "__main__":
    # 创建ContextualCompressionRAG实例
    rag = ContextualCompressionRAG(compression_type="selective")

    try:
        # 处理文档（取消注释以使用）
        result = rag.process_document("Agent基础.md")
        print(f"处理结果: {result}")

        # 查询问题
        query = "思维链是什么？举个例子"
        response = rag.query(query)

        print(f"查询: {query}")
        print(f"回答: {response['answer']}")
        print(f"压缩信息: {response['compression_info']}")
    except Exception as e:
        print(f"错误: {e}")





