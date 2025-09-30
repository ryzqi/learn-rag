import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
import jieba
from typing import List, Dict, Any
from LLM import GeminiLLM
from file_reader import file_reader_service
from embedding import embedding_service
from milvus_client import milvus_service
import datetime
import re


client = GeminiLLM()


class FusionRAG:
    """
    融合检索增强生成(Fusion RAG)类
    
    结合密集检索（向量搜索）和稀疏检索（BM25）的优势，使用RRF算法融合检索结果。
    
    Args:
        llm_client: LLM客户端实例 (GeminiLLM)
        embedding_client: 嵌入服务客户端实例 (EmbeddingClient)
        vector_client: 向量数据库客户端实例 (MilvusClient)
        file_reader_client: 文件读取服务客户端实例 (FileReaderService)
        collection_name: 向量数据库集合名称，默认为"RAG_learn"
        default_chunk_size: 默认文本分块大小，默认为800
        default_overlap: 默认分块重叠大小，默认为100
        default_k: 默认检索文档数量，默认为5
        rrf_k: RRF算法的k参数，默认为60
        alpha: 密集和稀疏检索的权重平衡参数，默认为0.5
        default_temperature: 默认生成温度，默认为0.2
    """
    
    def __init__(
        self,
        llm_client=None,
        embedding_client=None,
        vector_client=None,
        file_reader_client=None,
        collection_name: str = "RAG_learn",
        default_chunk_size: int = 800,
        default_overlap: int = 100,
        default_k: int = 5,
        rrf_k: int = 60,
        alpha: float = 0.5,
        default_temperature: float = 0.2,
    ):
        """
        初始化融合RAG实例
        
        Args:
            llm_client: LLM客户端，如果为None则使用默认GeminiLLM
            embedding_client: 嵌入客户端，如果为None则使用默认embedding_service
            vector_client: 向量数据库客户端，如果为None则使用默认milvus_service
            file_reader_client: 文件读取客户端，如果为None则使用默认file_reader_service
            collection_name: 向量数据库集合名称
            default_chunk_size: 默认文本分块大小
            default_overlap: 默认分块重叠大小
            default_k: 默认检索文档数量
            rrf_k: RRF算法的k参数
            alpha: 密集和稀疏检索的权重平衡参数
            default_temperature: 默认生成温度
        """
        # 依赖注入，支持默认值
        self.llm = llm_client or client
        self.embedding = embedding_client or embedding_service
        self.vector_db = vector_client or milvus_service
        self.file_reader = file_reader_client or file_reader_service
        
        # 配置参数
        self.collection_name = collection_name
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        self.default_k = default_k
        self.rrf_k = rrf_k
        self.alpha = alpha
        self.default_temperature = default_temperature
        
        # BM25索引存储
        self.bm25_index = None
        self.chunks = None
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """
        将文本分割为重叠的块。
        
        Args:
            text (str): 要分割的输入文本
            chunk_size (int): 每个块的字符数
            overlap (int): 块之间的字符重叠数
            
        Returns:
            List[Dict]: 包含文本和元数据的块字典列表
        """
        chunk_size = chunk_size or self.default_chunk_size
        overlap = overlap or self.default_overlap
        
        chunks = []  # 初始化一个空列表来存储块
        
        # 使用指定的块大小和重叠迭代文本
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]  # 提取指定大小的块
            if chunk:  # 确保不添加空块
                chunk_data = {
                    "text": chunk,  # 块文本
                    "metadata": {
                        "start_char": i,  # 文本块的起始字符索引
                        "end_char": i + len(chunk),  # 文本块的结束字符索引
                        "index": len(chunks),  # 块索引
                    },
                }
                chunks.append(chunk_data)
        
        return chunks  # 返回块列表  # 返回块列表
    
    def clean_text(self, text: str) -> str:
        """
        通过移除多余的空白字符和特殊字符来清理文本。
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 清理后的文本
        """
        # 将多个空白字符（包括换行符和制表符）替换为一个空格
        text = re.sub(r"\s+", " ", text)
        
        # 修复常见的OCR问题，将制表符和换行符替换为空格
        text = text.replace("\\t", " ")
        text = text.replace("\\n", " ")
        
        # 移除开头和结尾的空白字符，并确保单词之间只有一个空格
        text = " ".join(text.split())
        
        return text
    
    def create_bm25_index(self, chunks: List[Dict[str, Any]]):
        """
        为给定的文档块创建BM25索引
        
        Args:
            chunks: 文档块列表
            
        Returns:
            BM25Okapi: BM25索引对象
        """
        texts = [chunk["text"] for chunk in chunks]
        tokenized_docs = [list(jieba.cut(text)) for text in texts]
        
        bm25 = BM25Okapi(tokenized_docs)
        return bm25
    
    def bm25_search(self, bm25, chunks: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        使用BM25进行稀疏检索
        
        Args:
            bm25: BM25索引对象
            chunks: 文档块列表
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            List[Dict]: BM25检索结果列表
        """
        query_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(query_tokens)
        
        results = []
        for i, score in enumerate(scores):
            if i < len(chunks):
                metadata = chunks[i].get("metadata", {}).copy()
                metadata["index"] = i
                results.append({
                    "text": chunks[i]["text"],
                    "metadata": metadata,
                    "bm25_score": float(score),
                    "rank": i + 1
                })
        
        # 按BM25分数排序
        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:k]
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        使用向量搜索进行密集检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            List[Dict]: 向量检索结果列表
        """
        query_embedding = self.embedding.embed_text(query)
        results = self.vector_db.search_data(
            self.collection_name,
            query_embedding,
            limit=k,
            output_fields=["text", "metadata", "score"],
            metric_type="COSINE",
        )
        
        # 为结果添加排名信息
        for i, result in enumerate(results):
            result["rank"] = i + 1
            
        return results
    
    def reciprocal_rank_fusion(self, 
                              bm25_results: List[Dict[str, Any]], 
                              vector_results: List[Dict[str, Any]], 
                              k: int = None) -> List[Dict[str, Any]]:
        """
        使用倒数排名融合(RRF)算法融合BM25和向量检索结果
        
        Args:
            bm25_results: BM25检索结果
            vector_results: 向量检索结果
            k: RRF算法的k参数
            
        Returns:
            List[Dict]: 融合后的检索结果
        """
        k = k or self.rrf_k
        
        # 创建文档ID到分数的映射
        fusion_scores = {}
        doc_map = {}
        
        # 处理BM25结果
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result["metadata"]["index"]
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result
                doc_map[doc_id]["fusion_source"] = "bm25"
            else:
                doc_map[doc_id]["fusion_source"] = "both"
        
        # 处理向量搜索结果
        for rank, result in enumerate(vector_results, 1):
            doc_id = result["metadata"]["index"]
            fusion_scores[doc_id] = fusion_scores.get(doc_id, 0) + 1 / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = result
                doc_map[doc_id]["fusion_source"] = "vector"
            else:
                doc_map[doc_id]["fusion_source"] = "both"
        
        # 创建融合结果
        fusion_results = []
        for doc_id, score in fusion_scores.items():
            result = doc_map[doc_id].copy()
            result["fusion_score"] = score
            fusion_results.append(result)
        
        # 按融合分数排序
        fusion_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        
        return fusion_results
    
    def fusion_search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        执行融合检索：结合BM25和向量搜索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            List[Dict]: 融合检索结果
        """
        k = k or self.default_k
        
        if self.bm25_index is None or self.chunks is None:
            raise ValueError("BM25索引和文档块未初始化。请先调用process_document方法。")
        
        # 执行BM25检索
        bm25_results = self.bm25_search(self.bm25_index, self.chunks, query, k * 2)
        
        # 执行向量检索
        vector_results = self.vector_search(query, k * 2)
        
        # 使用RRF融合结果
        fusion_results = self.reciprocal_rank_fusion(bm25_results, vector_results)
        
        return fusion_results[:k]
    
    def process_document(self, 
                        file_path: str, 
                        chunk_size: int = None, 
                        overlap: int = None) -> List[Dict[str, Any]]:
        """
        处理文档，分块并创建索引
        
        Args:
            file_path: 文档文件路径
            chunk_size: 文本分块大小，如果为None则使用默认值
            overlap: 分块重叠大小，如果为None则使用默认值
            
        Returns:
            文档分块列表
        """
        # 读取文件
        text = self.file_reader.read_file(file_path)
        
        # 清理文本
        text = self.clean_text(text)
        
        # 分块
        self.chunks = self.chunk_text(text, chunk_size, overlap)
        
        # 创建BM25索引
        self.bm25_index = self.create_bm25_index(self.chunks)
        
        # 生成嵌入向量并存储到向量数据库
        if hasattr(self.embedding, "embed_chunks"):
            chunk_embeddings = self.embedding.embed_chunks([chunk["text"] for chunk in self.chunks])
        else:
            chunk_embeddings = [self.embedding.embed_text(chunk["text"]) for chunk in self.chunks]
        
        # 存储到向量数据库
        for i, (chunk, embedding) in enumerate(zip(self.chunks, chunk_embeddings)):
            data = {
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": {
                    "index": i,
                    "source": file_path,
                    "start_char": chunk["metadata"]["start_char"],
                    "end_char": chunk["metadata"]["end_char"],
                    "created_at": datetime.datetime.now().isoformat(),
                },
            }
            self.vector_db.insert_data(self.collection_name, data)
        
        return self.chunks
    
    def generate_response(self, query: str, context: str) -> str:
        """
        基于查询和上下文生成回答
        
        Args:
            query: 用户查询
            context: 检索到的上下文
            
        Returns:
            生成的回答
        """
        system_prompt = """你是一个专业的AI助手，任务是基于提供的上下文信息回答用户问题。
        请确保回答准确、相关且有帮助。如果上下文中没有足够信息回答问题，请明确说明。
        
        请遵循以下原则：
        1. 回答要基于提供的上下文
        2. 保持客观和准确
        3. 如果信息不足，要诚实说明
        4. 回答要清晰易懂"""
        
        user_prompt = f"""上下文信息：
{context}

用户问题：{query}

请基于上述上下文信息回答用户问题。"""
        
        response = self.llm.generate_text(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=self.default_temperature
        )
        
        return response
    
    def query(self, 
             query_text: str, 
             k: int = None, 
             include_metadata: bool = True) -> Dict[str, Any]:
        """
        执行完整的融合RAG查询流程
        
        Args:
            query_text: 查询文本
            k: 检索的文档数量，如果为None则使用默认值
            include_metadata: 是否在结果中包含元数据
            
        Returns:
            包含查询结果的字典，包含以下字段：
            - query: 原始查询
            - retrieved_docs: 检索到的文档列表
            - response: 生成的回答
            - retrieval_method: 使用的检索方法
            - metadata: 查询元数据（如果include_metadata为True）
        """
        k = k or self.default_k
        
        # 执行融合检索
        retrieved_docs = self.fusion_search(query_text, k)
        
        # 构建上下文
        context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])
        
        # 生成回答
        response = self.generate_response(query_text, context)
        
        # 构建结果
        result = {
            "query": query_text,
            "retrieved_docs": retrieved_docs,
            "response": response,
            "retrieval_method": "fusion_search"
        }
        
        if include_metadata:
            result["metadata"] = {
                "num_retrieved": len(retrieved_docs),
                "fusion_sources": [doc.get("fusion_source", "unknown") for doc in retrieved_docs],
                "fusion_scores": [doc.get("fusion_score", 0) for doc in retrieved_docs],
                "rrf_k": self.rrf_k,
                "alpha": self.alpha
            }
        
        return result


# 创建全局实例
fusion_rag = FusionRAG()




# 示例使用代码（如果直接运行此文件）
if __name__ == "__main__":
    # 创建融合RAG实例
    rag = FusionRAG()

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