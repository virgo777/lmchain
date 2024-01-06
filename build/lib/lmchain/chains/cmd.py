#这里是执行对CMD命令进行调用的chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from lmchain.lmchain.agents import llmAgent
import os,re

class LLMCMDChain:
    def __init__(self ,llm):
        qa_prompt = PromptTemplate(template="""你现在根据需要完成对命令行的编写，要根据需求编写对应的在Windows系统终端运行的命令,不要用%question形参这种指代的参数形式,直接给出可以运行的命令。
            Question: 给我一个在Windows系统终端中可以准确执行{question}的命令。
            ,
                input_variables=["question"],
            )
            answer:""", input_variables=["question"],            )
        self.qa_chain = LLMChain(llm=llm, prompt=qa_prompt)
        self.pattern = r"```(.*?)\```"

    def run(self ,text):
        cmd_response = self.qa_chain.run(question=text)
        cmd_string = str(cmd_response).split("```")[-2][1:-1]
        os.system(cmd_string)
        return cmd_string
