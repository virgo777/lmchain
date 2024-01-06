import warnings
warnings.filterwarnings("ignore")

import asyncio
from abc import ABC, abstractmethod
from typing import List


class Embeddings(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_documents, texts
        )

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await asyncio.get_running_loop().run_in_executor(
            None, self.embed_query, text
        )


# class LMEmbedding(Embeddings):
#     from modelscope.pipelines import pipeline
#     from modelscope.utils.constant import Tasks
#     pipeline_se = pipeline(Tasks.sentence_embedding, model='thomas/text2vec-base-chinese', model_revision='v1.0.0',
#                            device="cuda")
#
#     def _costruct_inputs(self, texts):
#         inputs = {
#             "source_sentence": texts
#         }
#
#         return inputs
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         """Embed search docs."""
#
#         inputs = self._costruct_inputs(texts)
#         result_embeddings = self.pipeline_se(input=inputs)
#         return result_embeddings["text_embedding"]
#
#     def embed_query(self, text: str) -> List[float]:
#         """Embed query text."""
#         inputs = self._costruct_inputs([text])
#         result_embeddings = self.pipeline_se(input=inputs)
#         return result_embeddings["text_embedding"]


class GLMEmbedding(Embeddings):
    import zhipuai as zhipuai
    zhipuai.api_key = "1f565e40af1198e11ff1fd8a5b42771d.SjNfezc40YFsz2KC"  # 控制台中获取的 APIKey 信息

    def _costruct_inputs(self, texts):
        inputs = {
            "source_sentence": texts
        }

        return inputs

    aembeddings = []  # 这个是为了在并发获取embedding_value时候使用的存储embedding_list内容。
    atexts = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        result_embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            result_embeddings.append(embedding)
        return result_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        result_embeddings = self.zhipuai.model_api.invoke(
            model="text_embedding", prompt=text)
        return result_embeddings["data"]["embedding"]

    def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
        result_embeddings = self.zhipuai.model_api.invoke(
            model="text_embedding", prompt=text)
        emb = result_embeddings["data"]["embedding"]

        self.aembeddings.append(emb)
        self.atexts.append(text)

    # 这里实现了并发embedding获取
    def aembed_documents(self, texts: List[str], thread_num=5, wait_sec=0.3) -> List[List[float]]:
        import threading
        text_length = len(texts)
        thread_batch = text_length // thread_num

        for i in range(thread_batch):
            start = i * thread_num
            end = (i + 1) * thread_num

            # 创建线程列表
            threads = []
            # 创建并启动5个线程，每个线程调用一个模型
            for text in texts[start:end]:
                thread = threading.Thread(target=self.aembed_query, args=(text,))
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join(wait_sec)  # 设置超时时间为0.3秒
        return self.aembeddings, self.atexts


if __name__ == '__main__':
    import time

    inputs = ["不可以，早晨喝牛奶不科学", "今天早晨喝牛奶不科学", "早晨喝牛奶不科学"] * 50

    start_time = time.time()
    aembeddings = (GLMEmbedding().aembed_documents(inputs, thread_num=5, thread_sec=0.3))
    print(aembeddings)
    print(len(aembeddings))
    end_time = time.time()
    # 计算函数执行时间并打印结果
    execution_time = end_time - start_time
    print(f"函数执行时间: {execution_time} 秒")
    print("----------------------------------------------------------------------------------")
    start_time = time.time()
    aembeddings = (GLMEmbedding().embed_documents(inputs))
    print(len(aembeddings))
    end_time = time.time()
    # 计算函数执行时间并打印结果
    execution_time = end_time - start_time
    print(f"函数执行时间: {execution_time} 秒")
