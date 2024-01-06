

import logging
import requests
from typing import Optional, List, Dict, Mapping, Any
import langchain
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache

logging.basicConfig(level=logging.INFO)
# 启动llm的缓存
langchain.llm_cache = InMemoryCache()


class AgentZhipuAI(LLM):
    import zhipuai as zhipuai
    # 模型服务url
    url = "127.0.0.1"
    zhipuai.api_key ="1f565e40af1198e11ff1fd8a5b42771d.SjNfezc40YFsz2KC"#控制台中获取的 APIKey 信息
    model = "chatglm_pro"  # 大模型版本

    history = []

    def getText(self,role, content):
        # role 是指定角色，content 是 prompt 内容
        jsoncon = {}
        jsoncon["role"] = role
        jsoncon["content"] = content
        self.history.append(jsoncon)
        return self.history

    @property
    def _llm_type(self) -> str:
        return "AgentZhipuAI"

    @classmethod
    def _post(self, url: str,              query: Dict) -> Any:

        """POST请求"""
        response = requests.post(url, data=query).json()
        return response

    def _call(self, prompt: str,              stop: Optional[List[str]] = None,role = "user") -> str:
        """_call"""
        # construct query
        response = self.zhipuai.model_api.invoke(
            model=self.model,
            prompt=self.getText(role=role, content=prompt)
        )
        choices = (response['data']['choices'])[0]
        self.history.append(choices)
        return choices["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.
        """
        _param_dict = {
            "url": self.url
        }
        return _param_dict


if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    llm = AgentZhipuAI()

    # 没有输入变量的示例prompt
    no_input_prompt = PromptTemplate(input_variables=[], template="给我讲个笑话。")
    no_input_prompt.format()

    prompt = PromptTemplate(
        input_variables=["location", "street"],
        template="作为一名专业的旅游顾问，简单的说一下{location}有什么好玩的景点,特别是在{street}？只要说一个就可以。",
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    print(chain.run({"location": "南京", "street": "新街口"}))


    from langchain.chains import ConversationChain
    conversation = ConversationChain(llm=llm, verbose=True)

    output = conversation.predict(input="你好！")
    print(output)

    output = conversation.predict(input="南京是哪里的省会？")
    print(output)

    output = conversation.predict(input="那里有什么好玩的地方，简单的说一个就好。")
    print(output)

