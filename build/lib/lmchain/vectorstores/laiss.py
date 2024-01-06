# LM AI Similarity Search

import warnings
warnings.filterwarnings("ignore")

import torch
import math
import asyncio
import logging
import operator
import os
import pickle
import uuid
import warnings
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sized,
    Tuple,
    Union,
)
import numpy as np
from lmchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance

from langchain.docstore.base import AddableMixin, Docstore
from langchain.docstore.document import Document
from langchain.docstore.in_memory import InMemoryDocstore


def dependable_faiss_import(no_avx2: Optional[bool] = None) -> Any:
    """
    Import faiss if available, otherwise raise error.
    If FAISS_NO_AVX2 environment variable is set, it will be considered
    to load FAISS with no AVX2 optimization.

    Args:
        no_avx2: Load FAISS strictly with no AVX2 optimization
            so that the vectorstore is portable and compatible with other devices.
    """
    if no_avx2 is None and "FAISS_NO_AVX2" in os.environ:
        no_avx2 = bool(os.getenv("FAISS_NO_AVX2"))

    try:
        if no_avx2:
            from faiss import swigfaiss as faiss
        else:
            import faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss


def _len_check_if_sized(x: Any, y: Any, x_name: str, y_name: str) -> None:
    if isinstance(x, Sized) and isinstance(y, Sized) and len(x) != len(y):
        raise ValueError(
            f"{x_name} and {y_name} expected to be equal length but "
            f"len({x_name})={len(x)} and len({y_name})={len(y)}"
        )
    return



class LMASS:
    def __init__(self):
        super().__init__()
        self.texts = []
        self.vectors = []
        self.embedding_class = None

    def from_documents(self, docs, embedding_class):

        texts = [str(doc) for doc in docs]
        self.texts = texts
        self.docs = docs
        self.embedding_class = embedding_class

        self.vectors = embedding_class.embed_documents(texts)
        return self.vectors

    def from_adocuments(self, docs, embedding_class,thread_num = 5,wait_sec = 0.3):

        texts = [str(doc) for doc in docs]
        self.texts = texts
        self.docs = docs
        self.embedding_class = embedding_class

        self.vectors = embedding_class.aembed_documents(texts, thread_num=thread_num, wait_sec=wait_sec)
        return self.vectors

    def get_relevant_documents(self,query,k = 3):
        query_vector = self.embedding_class.embed_query(query)
        sorted_index = self.get_similarity_vector_indexs(query_vector,self.vectors,k = k)
        sorted_docs = [self.docs[id] for id in sorted_index]

        return sorted_docs


    def get_similarity_vector_indexs(self, query_vector ,vectors, k: int = 3,  ):
        similarity = self._cosine_similaritys(query_vector,vectors)

        #这里是按分值从大到小的进行排序
        # 使用sort()函数对tensor进行降序排序，并返回排序后的tensor和索引
        sorted_tensor, sorted_indices = torch.sort(similarity, descending=True)
        return sorted_indices[:k].numpy()


    def _cosine_similaritys(self, query_vector, vectors):
        query_vector = torch.tensor(query_vector)
        vectors = torch.tensor(vectors)
        similarity_matrix = torch.nn.functional.cosine_similarity(query_vector, vectors, dim=-1)
        return similarity_matrix

        #CosineSimilarity
    def cosine_similarity(self, query_vector, tensor_2):
        normalized_tensor_1 = query_vector / query_vector.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
        return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

    #DotProductSimilarity
    def dot_product_similarity(self, query_vector, tensor_2, scale_output = True):
        result = (query_vector * tensor_2).sum(dim=-1)
        if scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(query_vector.size(-1))
        return result

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding,
            metadatas: Optional[List[dict]] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
    ) :
        embeddings = embedding.embed_documents(texts)
        return cls.__from(
            texts,
            embeddings,
            embedding,
            metadatas=metadatas,
            ids=ids,
            **kwargs,
        )

    @classmethod
    def __from(
            cls,
            texts: Iterable[str],
            embeddings: List[List[float]],
            embedding,
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
            normalize_L2: bool = False,
            distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
            **kwargs: Any,
    ) :
        faiss = dependable_faiss_import()
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(embeddings[0]))
        vecstore = cls(
            embedding,
            index,
            InMemoryDocstore(),
            {},
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )
        vecstore.__add(texts, embeddings, metadatas=metadatas, ids=ids)
        return vecstore



if __name__ == '__main__':
    v1 = torch.randn(size=(1,768))
    v2 = torch.randn(size=(4,768))
    lmass = LMASS()
    #print(lmass.dot_product_similarity(v1, v2))
    #print(lmass.cosine_similarity(v1, v2))
    sim_index = lmass.get_similarity_vector_indexs(v1,v2)
    print(sim_index)

