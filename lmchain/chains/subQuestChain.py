from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from tqdm import tqdm
from lmchain.tools import tool_register


class SubQuestChain:
    def __init__(self, llm):
        self.llm = llm

    def __call__(self, query=""):
        if query == "":
            raise "query需要填入查询问题"

        decomp_template = """
            GENERAL INSTRUCTIONS
            You are a domain expert. Your task is to break down a complex question into simpler sub-parts.

            USER QUESTION
            {user_question}

            ANSWER FORMAT
            ["sub-questions_1","sub-questions_2","sub-questions_3",...]
            """

        from langchain.prompts import PromptTemplate
        prompt = PromptTemplate(
            input_variables=["user_question"],
            template=decomp_template,
        )

        from langchain.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = (chain.run({"user_question": query}))

        import json
        sub_list = json.loads(response)

        return sub_list

    def run(self, query):
        sub_list = self.__call__(query)
        return sub_list


if __name__ == '__main__':
    from lmchain.agents import llmMultiAgent

    llm = llmMultiAgent.AgentZhipuAI()

    subQC = SubQuestChain(llm)
    response = subQC.run(query="工商银行财报中，2024财年Q1与Q2 之间，利润增长了多少？")
    print(response)