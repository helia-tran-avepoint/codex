import fnmatch
import json
from typing import Any, Dict, List
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import qdrant_client
from llama_index.llms.ollama import Ollama
from llama_index.core import get_response_synthesizer
ollm = Ollama(model="qwq:32b", request_timeout=120.0, base_url="http://10.1.71.1:11434")
fllm = Ollama(model="deepseek-r1:32b", request_timeout=120.0, base_url="http://10.1.71.1:11434")
def main():
    # 预处理数据
 
    Settings.chunk_size = 1280
    # 初始化Couchbase连接
    client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://<host>:<port>"
    # otherwise set Qdrant instance with host and port:
    host="10.1.70.240",
    port=6333
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
    )
    aclient = qdrant_client.AsyncQdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://<host>:<port>"
    # otherwise set Qdrant instance with host and port:
    host="10.1.70.240",
    port=6333
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
    )
    
    # 配置嵌入模型
    embed_model = OllamaEmbedding(model_name="nomic-embed-text",  base_url="http://10.1.71.1:11434")
    
    jira_vector_store = QdrantVectorStore( aclient=aclient, collection_name="codex-collection_jira", enable_hybrid=False)
    # 配置向量存储
    git_vector_store = QdrantVectorStore(aclient=aclient, collection_name="codex-collection_git", enable_hybrid=False)

    code_vector_store = QdrantVectorStore(aclient=aclient, collection_name="codex-collection_code", enable_hybrid=False)

    summary_vector_store = QdrantVectorStore(aclient=aclient, collection_name="codex-collection_summary", enable_hybrid=False)
    
    


    # 创建索引
    jiraindex = VectorStoreIndex.from_vector_store(
        embed_model=embed_model,
        vector_store=jira_vector_store
    )
        # 创建索引
    gitindex = VectorStoreIndex.from_vector_store(
        embed_model=embed_model,
        vector_store=git_vector_store
    )
        # 创建索引
    codeindex = VectorStoreIndex.from_vector_store(
        embed_model=embed_model,
        vector_store=code_vector_store
    )
        # 创建索引
    summaryindex = VectorStoreIndex.from_vector_store(
        embed_model=embed_model,
        vector_store=summary_vector_store
    )
    


    base_retrievers = [jiraindex.as_retriever(), gitindex.as_retriever(), codeindex.as_retriever()]
    fusion_retriever = QueryFusionRetriever(
        retrievers=base_retrievers,
        mode="simple",
        similarity_top_k=10,  # 最终返回结果数
        num_queries=4,        # 生成的不同查询变体数
        llm=fllm,
        verbose=True,
    )
    #r= fusion_retriever.retrieve("Why does the page display Sorry, you don't have permission after the Service Providers user logs in?")
    query_engine  = RetrieverQueryEngine(fusion_retriever, get_response_synthesizer(llm=ollm))

    print("欢迎使用问答系统（输入'exit'或'quit'退出）")
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入您的问题：")
            
            # 退出条件检测
            if user_input.lower() in ('exit', 'quit'):
                print("再见！")
                break
                
            # 空输入处理
            if not user_input.strip():
                print("问题不能为空，请重新输入。")
                continue
                
            # 执行查询
            response = query_engine.query(user_input)
            print("\nContext：", response.source_nodes)
            print("\n回答：", response.response)

            
        except KeyboardInterrupt:
            print("\n检测到中断指令，退出系统中...")
            break
        except Exception as e:
            print(f"\n查询过程中发生错误：{str(e)}")

if __name__ == "__main__":
    main()