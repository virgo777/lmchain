LMchain is a toolkit specifically adapted for chinese large model chains

Lmchain是专用为中国大陆用户提供免费大模型服务的工具包，目前免费推荐使用chatGLM。

免费用户可以在https://open.bigmodel.cn
注册并获取免费API。也可以使用lmchain中自带的免费key。

功能正在陆续添加中，用户可以在issues中发表内容，也可以与作者联系5847713@qq.com
欢迎提出您的想法和建议。

注意：lmchian随着GLM4的更新，已全新更新为新的API，老的基本GLM3版本的用户可以继续使用(版本最高为0.1.78)。
-----------------------------------------------------------------------------
使用方法：```pip install lmchain```
-----------------------------------------------------------------------------

>1、从一个简单的文本问答如下
```
from lmchain.agents import AgentZhipuAI
llm = AgentZhipuAI()

response = llm("你好")
print(response)

response = llm("南京是哪里的省会")
print(response)

response = llm("那里有什么好玩的地方")
print(response)
```





