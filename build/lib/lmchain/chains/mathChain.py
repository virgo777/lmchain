#这里是执行对CMD命令进行调用的chain

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from lmchain.lmchain.agents import llmAgent
import os,re,math

try:
    import numexpr  # noqa: F401
except ImportError:
    raise ImportError(
        "LMchain requires the numexpr package. "
        "Please install it with `pip install numexpr`."
    )


class LLMMathChain:
    def __init__(self ,llm):
        qa_prompt = PromptTemplate(template="""现在给你一个中文命令，请你把这个命令转化成数学公式。直接给出数学公式。这个公式会在numexpr包中调用。
            Question: 我现在需要计算{question}，结果需要在numexpr包中调用。
            ,
                input_variables=["question"],
            )
            answer:""", input_variables=["question"],            )
        self.qa_chain = LLMChain(llm=llm, prompt=qa_prompt)


    def run(self ,text):
        cmd_response = self.qa_chain.run(question=text)
        result = self._evaluate_expression(str(cmd_response))
        return result


    def _evaluate_expression(self, expression: str) -> str:
        import numexpr  # noqa: F401

        try:
            local_dict = {"pi": math.pi, "e": math.e}
            output = str(
                numexpr.evaluate(
                    expression.strip(),
                    global_dict={},  # restrict access to globals
                    local_dict=local_dict,  # add common mathematical functions
                )
            )
        except Exception as e:
            raise ValueError(
                f'LMchain._evaluate("{expression}") raised error: {e}.'
                " Please try again with a valid numerical expression"
            )

        # Remove any leading and trailing brackets from the output
        return re.sub(r"^\[|\]$", "", output)