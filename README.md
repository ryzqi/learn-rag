# RAG系统服务

一个完整的RAG（检索增强生成）系统，集成了向量数据库、文本嵌入、重排序和文件读取功能。

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 环境配置
创建 `.env` 文件并配置以下环境变量：

```env
# Milvus配置
MILVUS_URI=your_milvus_uri
MILVUS_TOKEN=your_milvus_token

# SiliconFlow API配置
SILICONFLOW_API_KEY=your_api_key

# 可选配置
REQUEST_TIMEOUT=30
MAX_RETRIES=3
EMBEDDING_MODEL=BAAI/bge-m3
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

### 基本使用

```python
from services import (
    services,           # 服务注册中心
    quick_search,       # 快速搜索
    add_text,          # 添加文本
    setup_collection   # 创建集合
)

# 1. 创建集合
setup_collection("my_docs", dimension=1024)

# 2. 添加文本
add_text("这是一段示例文本", "my_docs")

# 3. 搜索
results = quick_search("示例", "my_docs", top_k=5)
print(results)
```

## 📁 项目结构

```
RAG/
├── milvus_client.py      # Milvus向量数据库客户端
├── embedding.py          # 文本嵌入服务
├── rerank.py            # 重排序服务
├── file_reader.py       # 文件读取服务
├── services.py          # 服务注册中心（统一接口）
├── cleanup_collections.py # 集合清理工具
├── requirements.txt     # 项目依赖
├── SERVICES_GUIDE.md   # 详细使用指南
└── README.md           # 项目说明
```

## 🔧 核心功能

### 1. 向量数据库操作
- 集合管理（创建、删除、查询）
- 索引管理（向量索引、标量索引）
- 数据操作（插入、更新、删除、搜索）

### 2. 文本处理
- 多格式文档读取（PDF、DOCX、MD、TXT）
- 文本嵌入向量生成
- 智能文本分块

### 3. 智能搜索
- 向量相似度搜索
- 文本到向量搜索
- 混合搜索（向量+标量过滤）
- 重排序优化

### 4. 便捷工具
- 一键集合设置
- 批量文档处理
- 服务状态监控

## 📖 详细文档

查看 [SERVICES_GUIDE.md](./SERVICES_GUIDE.md) 获取完整的使用指南和API文档。

## 🛠 工具脚本

### 清理测试集合
```bash
python cleanup_collections.py
```

## 🔍 服务监控

```python
from services import get_service_status

# 检查所有服务状态
status = get_service_status()
for service, info in status.items():
    print(f"{service}: {'✓' if info['available'] else '✗'}")
```

## 📝 许可证

本项目采用 MIT 许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请查看 [SERVICES_GUIDE.md](./SERVICES_GUIDE.md) 或提交 Issue。
