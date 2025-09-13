"""
### 输入输出定义
- **输入：** 用户自然语言问题（如“如何退货？”）
- **输出：** 最相关的 FAQ 条目及其答案

### 扩展项
- 支持热更新知识库（ 自动 re-index）

### 工程化要求
- 使用 LlamaIndex 构建索引 Done
- 部署 Milvus 作为向量库 Done
- 实现文档切片优化（语义切分 + 重叠）use markdown parser Replace
"""
import sys
import uuid

from pydantic import BaseModel
from typing import Optional
from llama_index.core import Settings, Response
from llama_index.core.base.response.schema import StreamingResponse, AsyncStreamingResponse, PydanticResponse
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import MarkdownNodeParser,NodeParser
from llama_index.core import Document
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
from llama_index.core import VectorStoreIndex,StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.milvus import MilvusVectorStore
import os
from fastapi import FastAPI


api_key = os.getenv("DASHSCOPE_API_KEY")
Settings.llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    is_chat_model=True
)

local_embedding = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)
Settings.embed_model = local_embedding
# round = uuid.uuid4().__str__()
vector_store = None
def clear_milvus():
    from pymilvus import MilvusClient
    client = MilvusClient()
    client.drop_collection("llamaindexcollection")
    global vector_store
    vector_store = MilvusVectorStore(uri="http://127.0.0.1:19530/",dim=1024)
clear_milvus()

class FAQIndex:
    files_path:list=None
    parser:NodeParser =None
    vector_store = None
    store_context = None
    index = None
    query_engine=None
    _setup = False
    document_map=[]
    def __init__(self,files_path:list,parser:NodeParser):
        self.files_path = files_path
        self.parser = parser
        self.vector_store = vector_store
        self.store_context = StorageContext.from_defaults(vector_store=self.vector_store)
    def _index(self):
        try:
            dcs = []
            for target_file in self.files_path:
                is_exists = os.path.exists(target_file)
                if is_exists:
                    with open(target_file,"rb") as f:
                        file_content = f.read()
                        fd = self._file_2_Doc(file_content)
                        dcs.append(fd)
                else:
                    continue

            nodes = self.parser.get_nodes_from_documents(dcs)
            new_dcs = []

            for node in nodes:
                d = Document(text=node.get_content(), metadata=node.metadata,id_=node.node_id)
                new_dcs.append(d)
                self.document_map.append({
                    "metadata":node.metadata,
                    "id":node.node_id
                })
            self.index = VectorStoreIndex.from_documents(documents=new_dcs,storage_context=self.store_context)
            self.query_engine = self.index.as_query_engine(similarity_top_k=3)
            # self.retrieve = VectorIndexRetriever.retrieve(index=self.index,similarity_top_k=3)
            # self.query_engine =RetrieverQueryEngine(retriever=self.retrieve)
            self._setup=True
        except Exception as e:
            print(e)
    def ask(self,question:str)-> Response | StreamingResponse | AsyncStreamingResponse | PydanticResponse:
        if self._setup:
            response =  self.query_engine.query(question)
            source_nodes = response.source_nodes
            for index,node in enumerate(source_nodes):
                print(f"结果 {index + 1}:")
                print(f"ID: {node.node_id}")
                print(f"分数: {node.score}")
            return response
        else:
            return Response("not setup")
    def _file_2_Doc(self,content:bytes,**kwargs)->Document:

        md = kwargs.get("metadata",{})
        return Document(text=content,metadata=md)

    def del_index(self):
        self._setup = False
        self.vector_store.client.drop_collection(vector_store.collection_name)
        self.vector_store.client.create_collection(vector_store.collection_name)

    def re_index(self,files_path:list[str]):
        self.del_index()
        self.files_path=files_path
        self._index()
    def insert_index(self,file_path:str):
        try:
            if file_path not in self.files_path:
                self.files_path.append(file_path)
                with open(file_path,"rb")as f:
                    file_content =f.read()
                dc = Document(text=file_content)
                nodes = self.parser.get_nodes_from_documents([dc])
                for node in nodes:
                    d = Document(text=node.get_content(), metadata=node.metadata, id_=node.node_id)
                    self.document_map.append({
                        "metadata": node.metadata,
                        "id": node.node_id
                    })
                    self.index.insert(document=d)
            else:
                pass
        except Exception as e:
            print(e)
            return False
        return True
    def get_document_info(self):
        return self.document_map

app = FastAPI()



faq_index = FAQIndex(files_path=["milvus_faq/notebook.md"], parser=MarkdownNodeParser())
faq_index._index()
# - 提供 RESTful API 接口（FastAPI 封装）

class updateKbReq(BaseModel):
    files:list[str]=[]
@app.post("/knowledge_base/update")
def update_kb(req:updateKbReq):
    """
    真实环境需要额外提供上传接口，此处仅做示例
    files: list[str]
    """
    files = req.files
    if len(files)<1:
        return {"code":-1,"msg":"files error"}
    else:
        for fn in files:
            is_success = faq_index.insert_index(fn)
            if is_success:
                pass
            else:
                return {"code":-1,"msg":f"{fn} insert failed"}
    return {"code":0,"msg":"success"}

class quRequest(BaseModel):
    question:str

@app.post("/knowledge_base/qa")
def qa(req:quRequest):
    """
    question:str
    """
    question = req.question
    if len(question)<1 or question == "":
        return {"code":-1,"msg":"question is empty"}
    response = faq_index.ask(question)
    return {"code":0,"msg":response}

