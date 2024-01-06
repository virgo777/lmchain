from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from tqdm import tqdm
from lmchain.tools import tool_register


class GLMToolChain:
    def __init__(self, llm):

        self.llm = llm
        self.tool_register = tool_register
        self.tools = tool_register.get_tools()

    def __call__(self, query="", tools=None):

        if query == "":
            raise "query需要填入查询问题"
        if tools != None:
            self.tools = tools
        else:
            raise "将使用默认tools完成函数工具调用~"
        template = f"""
        你现在是一个专业的人工智能助手，你现在的需求是{query}。而你需要借助于工具在{self.tools}中找到对应的函数，用json格式返回对应的函数名和参数。
        函数名定义为function_name,参数名为params,还要求写入详细的形参与实参。

        如果找到合适的函数，就返回json格式的函数名和需要的参数，不要回答任何描述和解释。

        如果没有找到合适的函数，则返回：'未找到合适参数，请提供更详细的描述。'
        """

        flag = True
        counter = 0
        while flag:
            try:
                res = self.llm(template)

                import json
                res_dict = json.loads(res)
                res_dict = json.loads(res_dict)
                flag = False
            except:
                # print("失败输出，现在开始重新验证")
                template = f"""
                你现在是一个专业的人工智能助手，你现在的需求是{query}。而你需要借助于工具在{self.tools}中找到对应的函数，用json格式返回对应的函数名和参数。
                函数名定义为function_name,参数名为params,还要求写入详细的形参与实参。

                如果找到合适的函数，就返回json格式的函数名和需要的参数，不要回答任何描述和解释。

                如果没有找到合适的函数，则返回：'未找到合适参数，请提供更详细的描述。'

                你刚才生成了一组结果，但是返回不符合json格式，现在请你重新按json格式生成并返回结果。
                """
                counter += 1
                if counter >= 5:
                    return '未找到合适参数，请提供更详细的描述。'
        return res_dict

    def run(self, query, tools=None):
        tools = (self.tool_register.get_tools())
        result = self.__call__(query, tools)

        if result == "未找到合适参数，请提供更详细的描述。":
            return "未找到合适参数，请提供更详细的描述。"
        else:
            print("找到对应工具函数，格式如下：", result)
            result = self.dispatch_tool(result)
            from lmchain.prompts.templates import PromptTemplate
            tool_prompt = PromptTemplate(
                input_variables=["query", "result"],  # 输入变量包括中文和英文。
                template="你现在是一个私人助手，现在你的查询任务是{query},而你通过工具从网上查询的结果是{result},现在根据查询的内容与查询的结果，生成最终答案。",
                # 使用模板格式化输入和输出。
            )
            from langchain.chains import LLMChain
            chain = LLMChain(llm=self.llm, prompt=tool_prompt)

            response = (chain.run({"query": query, "result": result}))

            return response

    def add_tools(self, tool):
        self.tool_register.register_tool(tool)
        return True

    def dispatch_tool(self, tool_result) -> str:
        tool_name = tool_result["function_name"]
        tool_params = tool_result["params"]
        if tool_name not in self.tool_register._TOOL_HOOKS:
            return f"Tool `{tool_name}` not found. Please use a provided tool."
        tool_call = self.tool_register._TOOL_HOOKS[tool_name]

        try:
            ret = tool_call(**tool_params)
        except:
            import traceback
            ret = traceback.format_exc()
        return str(ret)

    def get_tools(self):
        return (self.tool_register.get_tools())


if __name__ == '__main__':
    from lmchain.agents import llmMultiAgent

    llm = llmMultiAgent.AgentZhipuAI()

    from lmchain.chains import toolchain

    tool_chain = toolchain.GLMToolChain(llm)

    from typing import Annotated


    def rando_numbr(
            seed: Annotated[int, 'The random seed used by the generator', True],
            range: Annotated[tuple[int, int], 'The range of the generated numbers', True],
    ) -> int:
        """
        Generates a random number x, s.t. range[0] <= x < range[1]
        """
        import random
        return random.Random(seed).randint(*range)


    tool_chain.add_tools(rando_numbr)

    print("------------------------------------------------------")
    query = "今天shanghai的天气是什么？"
    result = tool_chain.run(query)

    result = tool_chain.dispatch_tool(result)
    print(result)


