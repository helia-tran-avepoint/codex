import fnmatch
import json
import re
from typing import Any, Dict, List
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def process_json_list(
    data: List[Dict[str, Any]],
    text_fields: List[str] = [],
    template: str = "",
    array_separator: str = " ",
    id_field: str = "id"
) -> List[Document]:
    """批量处理 JSON 对象列表，返回 LlamaIndex Document 列表
    
    Args:
        data: 原始数据列表，每个元素是一个字典
        text_fields: 文本字段名列表（支持通配符，如 "name", "meta.*"）
        template: 模板字符串，使用 {{field}} 格式引用字段（默认为空）
        array_separator: 数组字段值的连接符
        id_field: 唯一标识符字段名
    """
    documents = []
    
    for item in data:
        matched_fields = set()
        text = ""
        
        if template:  # 使用模板生成文本
            # 提取模板中的字段（如 {{field}}）
            fields_in_template = set(re.findall(r'\{\{(\w+)\}\}', template))
            # 检查字段是否存在
            for field in fields_in_template:
                if field not in item:
                    raise ValueError(f"字段 '{field}' 在模板中但数据中不存在")
            
            # 处理数组字段并替换模板
            replacements = {}
            for field in fields_in_template:
                value = item[field]
                if isinstance(value, (list, tuple)):
                    replacements[field] = array_separator.join(map(str, value))
                else:
                    replacements[field] = str(value) if value is not None else ""
            
           # 使用正则表达式一次性替换所有 {{field}} 占位符
            pattern = re.compile(r'\{\{(\w+)\}\}')
            text = pattern.sub(
                lambda m: replacements.get(m.group(1), ""), 
                template
            )
            matched_fields.update(fields_in_template)

        else:  # 原有逻辑：按 text_fields 生成文本
            text_parts = []
            for pattern in text_fields:
                for field in item.keys():
                    if fnmatch.fnmatch(field, pattern) and field not in matched_fields:
                        value = item[field]
                        if isinstance(value, (list, tuple)):
                            processed = array_separator.join(map(str, value))
                        else:
                            processed = str(value) if value is not None else ""
                        text_parts.append(f"{field}: {processed}")
                        matched_fields.add(field)
            text = "\r\n".join(text_parts)
        
        # 元数据（排除已用于生成文本的字段）
        metadata = {k: v for k, v in item.items() if k not in matched_fields}
        
        # 生成唯一 ID
        doc_id = str(item.get(id_field, hash(text)))
        
        documents.append(Document(
            text=text,
            metadata=metadata,
            doc_id=doc_id
        ))
    
    return documents


def process_file_list(
    data: List[Dict[str, Any]],
    directory: str = "",
    id_field: str = "id"
) -> List[Document]:
    """批量处理 JSON 对象列表，返回 LlamaIndex Document 列表
    
    Args:
        data: 原始数据列表，每个元素是一个字典
        text_fields: 文本字段名列表（支持通配符，如 "name", "meta.*"）
        template: 模板字符串，使用 {{field}} 格式引用字段（默认为空）
        array_separator: 数组字段值的连接符
        id_field: 唯一标识符字段名
    """
    documents = []
    
    for item in data:
        with open(directory+item.get(id_field), 'r', errors='ignore') as f:
         text = f.read()
        
        
        # 元数据（排除已用于生成文本的字段）
        metadata = item
        
        # 生成唯一 ID
        doc_id = str(item.get(id_field, hash(text)))
        
        documents.append(Document(
            text=text,
            metadata=metadata,
            doc_id=doc_id
        ))
    
    return documents




    gittemplate =  """
Git Commit Record:
- Commit Hash: {{hash}}
- Author: {{author}} ({{email}})
- Commit Data: {{date}}
- Related JIRA issue #: {{jira}}
- Changed File: {{file}}  
- Commit message 
{{message}}
- Related code changes
{{code}}  """  




# 4. 主处理流程
def main():

    # 预处理数据
    with open("C:\\Users\\scyu\\Desktop\\ffl\\jiradata.json") as f:
        jira_issues = json.load(f)
        jira_r =  process_json_list(jira_issues, ["Summary","Project description","Description","Custom field (Deployment Impact Assessment)",
                                                  "Comment","Comment.*","Attachment","Attachment.*", "Sprint", "Sprint.*","Custom field*"], id_field="Issue key")


    gittemplate =  """
-commit message 
{{message}}
-code changes
{{code}}"""  
    with open("C:\\Users\\scyu\\Desktop\\ffl\\blamdata.json") as f:
        blame_data = json.load(f)
        flattened = [item for sublist in blame_data for item in sublist]
        blame_r =  process_json_list(flattened, id_field="hash", template=gittemplate)
    

    filetemplate =  """
"{{file}}" analysis:
- number of Lines of Code (LOC) for this file: {{code_lines}}
"""  
    with open("C:\\Users\\scyu\\Desktop\\ffl\\filedata.json") as f:
        file_data = json.load(f)
        file_r =  process_file_list(file_data, id_field="file", directory="C:\\Users\\scyu\\Desktop\\code\\ffl\\")
    

    summarytemplate =  """
Project Code Language Summary:
- Programming Language: {{_language}}
- Lines of Code (LOC) for this language: {{_code_count}}
- Number of source code files for this language: {{_file_count}}
- Lines of documentation for this language: {{_documentation_count}}
- Empty Lines for this language: {{_empty_count}}
- Number of lines containing strings for this language: {{_string_count}}
- Percentage of files for this language in project: {{_file_percentage}}
"""
    with open("C:\\Users\\scyu\\Desktop\\ffl\\summary.json") as f:
        summary_data = json.load(f)
        summary_r =  process_json_list(summary_data, [], id_field="_language", template=summarytemplate)
    
    

    # 创建LlamaIndex文档

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
    jira_vector_store = QdrantVectorStore(client=client, collection_name="codex-collection_jira")
    # 配置向量存储
    git_vector_store = QdrantVectorStore(client=client, collection_name="codex-collection_git")

    code_vector_store = QdrantVectorStore(client=client, collection_name="codex-collection_code")

    summary_vector_store = QdrantVectorStore(client=client, collection_name="codex-collection_summary")

    # 配置存储上下文
  
    
    # 配置嵌入模型
    embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url= "http://10.1.71.1:11434/")
    
    storage_context = StorageContext.from_defaults(vector_store=jira_vector_store)
    # 创建索引
    index = VectorStoreIndex.from_documents(
        jira_r,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # 持久化存储
    index.storage_context.persist(persist_dir="./storage")




    storage_context = StorageContext.from_defaults(vector_store=git_vector_store)
    # 创建索引
    index = VectorStoreIndex.from_documents(
        blame_r,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # 持久化存储
    index.storage_context.persist(persist_dir="./storage")




    storage_context = StorageContext.from_defaults(vector_store=code_vector_store)
    # 创建索引
    index = VectorStoreIndex.from_documents(
        file_r,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # 持久化存储
    index.storage_context.persist(persist_dir="./storage")


    storage_context = StorageContext.from_defaults(vector_store=summary_vector_store)
    # 创建索引
    index = VectorStoreIndex.from_documents(
        summary_r,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    # 持久化存储
    index.storage_context.persist(persist_dir="./storage")


if __name__ == "__main__":
    main()