"""Select examples based on length."""
import re
from typing import Callable, Dict, List,Any, Dict, List, Optional, Type

from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import BaseModel, validator, Extra


def _get_length_based(text: str) -> int:
    return len(re.split("\n| ", text))


class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    """Select examples based on length."""

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    get_text_length: Callable[[str], int] = _get_length_based
    """Function to measure prompt length. Defaults to word count."""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut."""

    example_text_lengths: List[int] = []  #: :meta private:

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)
        self.example_text_lengths.append(self.get_text_length(string_example))

    @validator("example_text_lengths", always=True)
    def calculate_example_text_lengths(cls, v: List[int], values: Dict) -> List[int]:
        """Calculate text lengths if they don't exist."""
        # Check if text lengths were passed in
        if v:
            return v
        # If they were not, calculate them
        example_prompt = values["example_prompt"]
        get_text_length = values["get_text_length"]
        string_examples = [example_prompt.format(**eg) for eg in values["examples"]]
        return [get_text_length(eg) for eg in string_examples]

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the input lengths."""
        inputs = " ".join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        i = 0
        examples = []
        while remaining_length > 0 and i < len(self.examples):
            new_length = remaining_length - self.example_text_lengths[i]
            if new_length < 0:
                break
            else:
                examples.append(self.examples[i])
                remaining_length = new_length
            i += 1
        return examples


from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
def sorted_values(values: Dict[str, str]) -> List[Any]:
    """Return a list of values in dict sorted by key."""
    return [values[val] for val in sorted(values)]
class SemanticSimilarityExampleSelector(BaseExampleSelector, BaseModel):
    """Example selector that selects examples based on SemanticSimilarity."""

    vectorstore: VectorStore
    """VectorStore than contains information about examples."""
    k: int = 4
    """Number of examples to select."""
    example_keys: Optional[List[str]] = None
    """Optional keys to filter examples to."""
    input_keys: Optional[List[str]] = None
    """Optional keys to filter input to. If provided, the search is based on
    the input variables instead of all variables."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def add_example(self, example: Dict[str, str]) -> str:
        """Add new example to vectorstore."""
        if self.input_keys:
            string_example = " ".join(
                sorted_values({key: example[key] for key in self.input_keys})
            )
        else:
            string_example = " ".join(sorted_values(example))
        ids = self.vectorstore.add_texts([string_example], metadatas=[example])
        return ids[0]

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        # Get the docs with the highest similarity.
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(sorted_values(input_variables))
        example_docs = self.vectorstore.similarity_search(query, k=self.k)
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [dict(e.metadata) for e in example_docs]
        # If example keys are provided, filter examples to those keys.
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        embeddings: Embeddings,
        vectorstore_cls: Type[VectorStore],
        k: int = 4,
        input_keys: Optional[List[str]] = None,
        **vectorstore_cls_kwargs: Any,
    ):
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        if input_keys:
            string_examples = [
                " ".join(sorted_values({k: eg[k] for k in input_keys}))
                for eg in examples
            ]
        else:
            string_examples = [" ".join(sorted_values(eg)) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(vectorstore=vectorstore, k=k, input_keys=input_keys)
class MaxMarginalRelevanceExampleSelector(SemanticSimilarityExampleSelector):
    """ExampleSelector that selects examples based on Max Marginal Relevance.

    This was shown to improve performance in this paper:
    https://arxiv.org/pdf/2211.13892.pdf
    """

    fetch_k: int = 20
    """Number of examples to fetch to rerank."""

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on semantic similarity."""
        # Get the docs with the highest similarity.
        if self.input_keys:
            input_variables = {key: input_variables[key] for key in self.input_keys}
        query = " ".join(sorted_values(input_variables))
        example_docs = self.vectorstore.max_marginal_relevance_search(
            query, k=self.k, fetch_k=self.fetch_k
        )
        # Get the examples from the metadata.
        # This assumes that examples are stored in metadata.
        examples = [dict(e.metadata) for e in example_docs]
        # If example keys are provided, filter examples to those keys.
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    @classmethod
    def from_examples(
        cls,
        examples: List[dict],
        embeddings: Embeddings,
        vectorstore_cls: Type[VectorStore],
        k: int = 4,
        input_keys: Optional[List[str]] = None,
        fetch_k: int = 20,
        **vectorstore_cls_kwargs: Any,
    ):
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An iniialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        """
        if input_keys:
            string_examples = [
                " ".join(sorted_values({k: eg[k] for k in input_keys}))
                for eg in examples
            ]
        else:
            string_examples = [" ".join(sorted_values(eg)) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(vectorstore=vectorstore, k=k, fetch_k=fetch_k, input_keys=input_keys)

import numpy as np
def ngram_overlap_score(source: List[str], example: List[str]) -> float:
    """Compute ngram overlap score of source and example as sentence_bleu score.

    Use sentence_bleu with method1 smoothing function and auto reweighting.
    Return float value between 0.0 and 1.0 inclusive.
    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """
    from nltk.translate.bleu_score import (
        SmoothingFunction,  # type: ignore
        sentence_bleu,
    )

    hypotheses = source[0].split()
    references = [s.split() for s in example]

    return float(
        sentence_bleu(
            references,
            hypotheses,
            smoothing_function=SmoothingFunction().method1,
            auto_reweigh=True,
        )
    )
class NGramOverlapExampleSelector(BaseExampleSelector, BaseModel):
    """Select and order examples based on ngram overlap score (sentence_bleu score).

    https://www.nltk.org/_modules/nltk/translate/bleu_score.html
    https://aclanthology.org/P02-1040.pdf
    """

    examples: List[dict]
    """A list of the examples that the prompt template expects."""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples."""

    threshold: float = -1.0
    """Threshold at which algorithm stops. Set to -1.0 by default.

    For negative threshold:
    select_examples sorts examples by ngram_overlap_score, but excludes none.
    For threshold greater than 1.0:
    select_examples excludes all examples, and returns an empty list.
    For threshold equal to 0.0:
    select_examples sorts examples by ngram_overlap_score,
    and excludes examples with no ngram overlap with input.
    """

    #@root_validator(pre=True)
    def check_dependencies(cls, values: Dict) -> Dict:
        """Check that valid dependencies exist."""
        try:
            from nltk.translate.bleu_score import (  # noqa: F401
                SmoothingFunction,
                sentence_bleu,
            )
        except ImportError as e:
            raise ImportError(
                "Not all the correct dependencies for this ExampleSelect exist."
                "Please install nltk with `pip install nltk`."
            ) from e

        return values

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Return list of examples sorted by ngram_overlap_score with input.

        Descending order.
        Excludes any examples with ngram_overlap_score less than or equal to threshold.
        """
        inputs = list(input_variables.values())
        examples = []
        k = len(self.examples)
        score = [0.0] * k
        first_prompt_template_key = self.example_prompt.input_variables[0]

        for i in range(k):
            score[i] = ngram_overlap_score(
                inputs, [self.examples[i][first_prompt_template_key]]
            )

        while True:
            arg_max = np.argmax(score)
            if (score[arg_max] < self.threshold) or abs(
                score[arg_max] - self.threshold
            ) < 1e-9:
                break

            examples.append(self.examples[arg_max])
            score[arg_max] = self.threshold - 1.0

        return examples



