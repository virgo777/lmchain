from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lmchain.embeddings import embeddings
from lmchain.vectorstores import laiss
from lmchain.agents import llmMultiAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,  # 用于构建聊天模板的类
    MessagesPlaceholder,  # 用于在模板中插入消息占位的类
    SystemMessagePromptTemplate,  # 用于构建系统消息模板的类
    HumanMessagePromptTemplate  # 用于构建人类消息模板的类
)
from langchain.chains import ConversationChain

class ConversationalRetrievalChain:
    def __init__(self,document,chunk_size = 1280,chunk_overlap = 50,file_name = "这是一份辅助材料"):
        """
        :param document: 输入的文本内容，只要一个text文本
        :param chunk_size: 切分后每段的字数
        :param chunk_overlap: 每个相隔段落重叠的字数
        :param file_name: 文本名称/文本地址
        """
        self.text_splitter = RecursiveCharacterTextSplitter(    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_tool = embeddings.GLMEmbedding()

        self.lmaiss = laiss.LMASS() #这里是用于将文本转化为vector，并且计算query相应的相似度的类
        self.llm = llmMultiAgent.AgentZhipuAI()
        self.memory = ConversationBufferMemory(return_messages=True)

        conversation_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template("你是一个最强大的人工智能程序，可以知无不答，但是你不懂的东西会直接回答不知道。"),
                MessagesPlaceholder(variable_name="history"),  # 历史消息占位符
                HumanMessagePromptTemplate.from_template("{input}")  # 人类消息输入模板
            ])

        self.qa_chain = ConversationChain(memory=self.memory, prompt=conversation_prompt, llm=self.llm)
        "---------------------------"
        document = [Document(page_content=document, metadata={"source": file_name})]    #对输入的document进行格式化处理
        self.documents = self.text_splitter.split_documents(document)    #根据
        self.vectorstore = self.lmaiss.from_documents(self.documents, embedding_class=self.embedding_tool)

    def __call__(self, query):
        query_embedding = self.embedding_tool.embed_query(query)

        #根据query查找最近的那个序列
        close_id = self.lmaiss.get_similarity_vector_indexs(query_embedding, self.vectorstore, k=1)[0]
        #查找最近的那个段落id
        doc = self.documents[close_id]

        #构建查询的query
        query = f"你现在要回答问题'{query}',你可以参考文献'{doc}',你如果找不到对应的内容,就从自己的记忆体中查找，就回答'请提供更为准确的查询内容'，注意你要一步步的思考再回答。"
        result = (self.qa_chain.predict(input=query))
        return result

    def predict(self,input):
        result = self.__call__(input)
        return result