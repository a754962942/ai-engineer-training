"""
## 作业二：构建一个融合文档检索、图谱推理的多跳问答系统

### 场景设定
- **用户问：** “A 公司的最大股东是谁？”

### 系统流程
1. 检索 A 公司相关信息（RAG）
2. 图谱中查找控股关系（KG）
3. 生成最终回答（LLM）

### 技术难点
- 如何将 RAG 与图谱推理融合？
- 如何设计联合评分机制？
- 如何防止错误传播？（如图谱中错误关系导致错误回答）

### 工程化要求
- 使用 Neo4j 构建企业股权图谱 DONE
- 使用 LlamaIndex 实现文档检索 DONE
- 实现多跳查询逻辑（Cypher + LLM 协同）
- 构建可解释性输出（展示推理路径）

"""
from idlelib.query import Query

from  llama_index.core import Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.dashscope import DashScopeEmbedding,DashScopeTextEmbeddingModels
import  os

api_key = os.getenv("DASHSCOPE_API_KEY")

local_llm = OpenAILike(
    model="qwen-plus",
    api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key,
    is_chat_model=True
)

Settings.llm =  local_llm

local_embedding = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
    embed_batch_size=6,
    embed_input_length=8192
)

Settings.embed_model = local_embedding
from neo4j import GraphDatabase

def set_up():
    # 首先创建6家公司节点
    create_cypher = """CREATE
    (a:Company {name: 'A公司', code: 'A0001'}),
    (b:Company {name: 'B公司', code: 'B0001'}),
    (c:Company {name: 'C公司', code: 'C0001'}),
    (d:Company {name: 'D公司', code: 'D0001'}),
    (e:Company {name: 'E公司', code: 'E0001'}),
    (f:Company {name: 'F公司', code: 'F0001'})
    """
    relationship_cypher = """
    
    MATCH
    (a:Company{code:'A0001'}),
    (b:Company{code:'B0001'}),
    (c:Company{code:'C0001'}),
    (d:Company{code:'D0001'}),
    (e:Company{code:'E0001'}),
    (f:Company{code:'F0001'})
    CREATE
  (a)-[:OWNS_STOCK {percentage: 15.5}]->(b),
  (a)-[:OWNS_STOCK {percentage: 7.2}]->(c),
  (a)-[:OWNS_STOCK {percentage: 3.8}]->(f),
  
  (b)-[:OWNS_STOCK {percentage: 12.1}]->(a),
  (b)-[:OWNS_STOCK {percentage: 5.5}]->(d),
  (b)-[:OWNS_STOCK {percentage: 4.2}]->(e),
  
  (c)-[:OWNS_STOCK {percentage: 8.3}]->(a),
  (c)-[:OWNS_STOCK {percentage: 6.7}]->(d),
  (c)-[:OWNS_STOCK {percentage: 2.9}]->(f),
  
  (d)-[:OWNS_STOCK {percentage: 9.6}]->(b),
  (d)-[:OWNS_STOCK {percentage: 11.2}]->(e),
  
  (e)-[:OWNS_STOCK {percentage: 6.4}]->(c),
  (e)-[:OWNS_STOCK {percentage: 7.8}]->(d),
  (e)-[:OWNS_STOCK {percentage: 4.5}]->(f),
  
  (f)-[:OWNS_STOCK {percentage: 5.1}]->(a),
  (f)-[:OWNS_STOCK {percentage: 3.3}]->(e)
  """
    with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "12345678")) as driver:
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN n")
            print(f"MATCH n success,result:{result.single()}")

            res = session.run(create_cypher)
            print(f"create_node success:{res.single()}")

            result2 = session.run(relationship_cypher)
            print(f"create_cypher result:{result2.single()}")
            print("✅ 测试数据创建完成")

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import  Document
parser = MarkdownNodeParser()
def document_set_up(file_path:str):
    """ only support markdown file"""
    with open(file_path) as f :
        file_content = f.read()
    document = Document(text=file_content,metadata={"file_source":file_path})
    nodes = parser.get_nodes_from_documents(documents=[document])
    dcs = []
    for node in nodes:
        dc = Document(text=node.get_content(),metadata=node.metadata)
        dcs.append(dc)
    return dcs

from llama_index.core import VectorStoreIndex
# set_up()
documents = document_set_up("graph_rag/notebook.md")
index = VectorStoreIndex.from_documents(documents=documents)
query_engine = index.as_query_engine(similarity_top_k=3)

from llama_index.core.prompts import RichPromptTemplate
prompt_template = """
你是一个Cypher 专家，请你根据给定的图Schema和问题，写出查询语句。
请不要提供与cypher语句无关的任何内容。
Shema如下：
---------------------
nodes:
[{
  "identity": -1,
  "labels": [
    "Company"
  ],
  "properties": {
    "name": "Company",
    "indexes": [],
    "constraints": []
  },
  "elementId": "-1"
}]
relationships:
[{
  "identity": -1,
  "start": -1,
  "end": -1,
  "type": "OWNS_STOCK",
  "properties": {
    "name": "OWNS_STOCK"
  },
  "elementId": "-1",
  "startNodeElementId": "-1",
  "endNodeElementId": "-1"
}]

RelationshipType	Properties	Count
"OWNS_STOCK"	["percentage"]	16

数据库数据格式如下:
A公司、B公司

注意:如果要同时查询relationship，请使用指代的方式引用relationship的属性。
例如：[r:OWNS_STOCK] RETURN r.percentage
---------------------
问题如下：
---------------------
{{ question }}
---------------------
"""
def query_by_cypher(query:str):
    print(f"query:{query}")
    with GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "12345678")) as driver:
        records,_,_ = driver.execute_query(query)
        result = [record for record in records]
        print(f"query_by_cypher:{result}")
    return result

def ask(question:str):
    qa_prompt = RichPromptTemplate(prompt_template)
    prompt = qa_prompt.format(question=question)
    cypher = local_llm.complete(prompt)
    print(f"cypher:{cypher}")
    cypher_modify = local_llm.complete(prompt=f"""请校验一下的cypher语句是否存在错误，如果有错误，请对错误问题进行修复并仅返回cypher语句，如果没有错误则原样返回
仅需要返回完整语句即可，不需要输出除cypher语句以外的任何文字。
cypher语句:{cypher.text}
""")
    graph_result = query_by_cypher(cypher_modify.text)
    cypher_2_nature_language = local_llm.complete(prompt=f"""
    请从提供的问题和cypher语句查询结果中提取所有信息并拼接为自然语言描述。
    用户问题为:
    {question}
    cypher语句执行结果如下:
    {graph_result}
    """)
    result = query_engine.query(question)
    is_correct = local_llm.complete(f"""
    对比两位评论员的不同数据来源的数据是否语意一致，如果语意一致则输出true，否则输出false。
    只需要输出true/false即可，不需要也不允许输出除true/false以外的所有内容。
    ---------------------
        评论员1结论：
    ---------------------
    {cypher_2_nature_language.text}
    ---------------------
        评论员2结论：
    ---------------------
    {result}
    ---------------------
""")
    lower_bool = str(is_correct.text).lower()
    if lower_bool  in ["true","false"]:
        print(f"评论员1结论:\n{cypher_2_nature_language.text}\n评论员2结论:\n{result}\n")
        if lower_bool == "true":
            print("结论一致")
        else:
            print(f"result.source:{[source.text for source in result.source_nodes]}")
            print("结论不一致")
    else:
        print(f"评委输出有误。{lower_bool}")
def main():
    ask("A公司的投资有哪些？")
    # result = query_engine.query("")
    # texts = []
    # for source in result.source_nodes:
    #     d = {"score":source.score,"text":source.text}
    #     texts.append(d)
    # print(f"result:{texts}")
    # print(f"texts.len:{texts.__len__()}")
    # for t in texts:
    #     print(t)
if __name__ == "__main__":
    main()