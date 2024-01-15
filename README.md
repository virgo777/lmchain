LMchain is a toolkit specifically adapted for chinese large model chains.

Lmchain是专用为中国大陆用户提供免费大模型服务的工具包，目前免费推荐使用chatGLM。

免费用户可以在https://open.bigmodel.cn
注册并获取免费API。也可以使用lmchain中自带的免费key。

功能正在陆续添加中，用户可以在issues中发表内容，也可以与作者联系5847713@qq.com
欢迎提出您的想法和建议。
-----------------------------------------------------------------------------
使用方法：```pip install lmchain```
-----------------------------------------------------------------------------

>1、从一个简单的文本问答如下
```
from lmchain.agents import llmMultiAgent
llm = llmMultiAgent.AgentZhipuAI()
llm.zhipuai.api_key = "1f565e40af1198e11ff1fd8a5b42771d.SjNfezc40YFsz2KC" #你个人注册可正常使用的API KEY
response = llm("南京是哪里的省会？")
print(response)

response = llm("那里有什么好玩的地方？")
print(response)
```

>2、除此之外，lmchain还有对复杂任务拆解的功能，例如：
```
from lmchain.agents import llmMultiAgent
llm = llmMultiAgent.AgentZhipuAI()


query = "工商银行财报中，2023 Q3相比，2024 Q1的收益增长了多少？"

from lmchain.chains import subQuestChain
subQC = subQuestChain.SubQuestChain(llm)
response = subQC.run(query=query)

print(response)
```
>3、调用大模型Embedding tool对文本进行嵌入embedding计算的方法
```
from lmchain.vectorstores import embeddings  # 导入embeddings模块
embedding_tool = embeddings.GLMEmbedding()  # 创建一个GLMEmbedding对象
embedding_tool.zhipuai.api_key = "1f565e40af1198e11ff1fd8a5b42771d.SjNfezc40YFsz2KC" #你个人注册可正常使用的API KEY

inputs = ["lmchain还有对复杂任务拆解的功能", "目前lmchain还提供了对工具函数的调用方法", "Lmchain是专用为中国大陆用户提供免费大模型服务的工具包"] * 50

#由于此时对embedding的处理，对原始传入的文本顺序做了变更，
# 因此需要采用新的文本list排序
aembeddings,atexts = (embedding_tool.aembed_documents(inputs))
print(aembeddings)

#每条文本内容被embedding处理为[1,1024]大小的序列
import numpy as np
aembeddings = (np.array(aembeddings))
print(aembeddings.shape)
```
>4、目前lmchain还提供了对工具函数的调用方法
```
from lmchain.agents import llmMultiAgent
llm = llmMultiAgent.AgentZhipuAI()

from lmchain.chains import toolchain

tool_chain = toolchain.GLMToolChain(llm)

query = "说一下上海的天气"
response = tool_chain.run(query)
print(response)
```

>5、添加自定义工具并调用的方法
```
from lmchain.agents import llmMultiAgent
llm = llmMultiAgent.AgentZhipuAI()

from lmchain.chains import toolchain
tool_chain = toolchain.GLMToolChain(llm)

from typing import  Annotated
#下面的play_game是自定义的工具
def play_game(
    #使用Annotated对形参进行标注[形参类型，形参用途描述，是否必须]
    num: Annotated[int, 'use the num to play game', True],
):
    #函数内注释是为了向模型提供对函数用途的解释
    """
    一个数字游戏，
    随机输入数字，按游戏规则输出结果的游戏
    """
    if num % 3:
        return 3
    if num % 5:
        return 5
    return 0

tool_chain.add_tools(play_game)
query = "玩一个数字游戏，输入数字3"
result = tool_chain.run(query)

print(result)

```
其他功能正在陆续添加中，欢迎读者留下您的意见或与作者联系。





