import os

os.environ['OPENAPI_API_KEY'] = 'key1'
os.environ['SERPAIP_API_KEY'] = 'key2'

# ReAct实现逻辑的完整代码：
from langchain import hub
prompt = hub.pull("hwchase17/react")
print(prompt)

# 导入openai
from langchain_openai import OpenAI

# 选择要使用的大模型
llm = OpenAI()
# 导入 SerpAPIWrapper工具包
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents.tools import BaseTool

# 实例化 SerpAPIWrapper
search = SerpAPIWrapper()
# 准备工具列表
tools = [
    BaseTool(
        name="Search",
        func=search.run,
        description="当大模型没有相关知识时，用于搜索知识"
    )
]

# 导入 create_react_agent 功能
