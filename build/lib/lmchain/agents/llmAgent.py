import time
import logging
import requests
from typing import Optional, List, Dict, Mapping, Any

import langchain
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache

logging.basicConfig(level=logging.INFO)
# 启动llm的缓存
langchain.llm_cache = InMemoryCache()


class AgentChatGLM(LLM):
    # 模型服务url
    url = "http://127.0.0.1:7866/chat"
    #url = "http://192.168.3.20:7866/chat"  #3050服务器上
    history = []

    @property
    def _llm_type(self) -> str:
        return "chatglm"

    def _construct_query(self, prompt: str) -> Dict:
        """构造请求体
        """
        query = {"query": prompt, "history": self.history}
        import json
        query = json.dumps(query)  # 对请求参数进行JSON编码

        return query

    def _construct_query_tools(self, prompt: str , tools: list ) -> Dict:
        """构造请求体
        """
        tools_info = {"role": "system",
                      "content": "你现在是一个查找使用何种工具以及传递何种参数的工具助手，你会一步步的思考问题。你根据需求查找工具函数箱中最合适的工具函数，然后返回工具函数名称和所工具函数对应的参数，参数必须要和需求中的目标对应。",
                      "tools": tools}
        query = {"query": prompt, "history": tools_info}
        import json
        query = json.dumps(query)  # 对请求参数进行JSON编码

        return query


    @classmethod
    def _post(self, url: str,              query: Dict) -> Any:

        """POST请求"""
        response = requests.post(url, data=query).json()
        return response

    def _call(self, prompt: str,              stop: Optional[List[str]] = None, tools:list = None) -> str:
        """_call"""
        if tools == None:
            # construct query
            query = self._construct_query(prompt=prompt)

            # post
            response = self._post(url=self.url,query=query)

            response_chat = response["response"];
            self.history = response["history"]

            return response_chat
        else:

            query = self._construct_query_tools(prompt=prompt,tools=tools)
            # post
            response = self._post(url=self.url, query=query)
            self.history = response["history"]  #这个history要放上面
            response = response["response"]
            try:
                #import ast
                #response = ast.literal_eval(response)
                ret = tool_register.dispatch_tool(response["name"], response["parameters"])
                response_chat = llm(prompt=ret)
            except:
                response_chat = response
            return str(response_chat)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url
        }
        return _param_dict


if __name__ == "__main__":

    import tool_register

    # 获取注册后的全部工具，并以json的形式返回
    tools = tool_register.get_tools()
    "--------------------------------------首先是对tools的定义---------------------------------------"

    llm = AgentChatGLM()
    llm.url = "http://192.168.3.20:7866/chat"
    while True:
        while True:
            human_input = input("Human: ")
            if human_input == "tools":
                break

            begin_time = time.time() * 1000
            # 请求模型
            response = llm(human_input)
            end_time = time.time() * 1000
            used_time = round(end_time - begin_time, 3)
            #logging.info(f"chatGLM process time: {used_time}ms")
            print(f"Chat: {response}")

        human_input = input("Human_with_tools_Ask: ")
        response = llm(prompt=human_input,tools=tools)
        print(f"Chat_with_tools_Que: {response}")




