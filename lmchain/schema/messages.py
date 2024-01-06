from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Union

from typing_extensions import Literal

#from langchain.load.serializable import Serializable
#from langchain.pydantic_v1 import Extra, Field

from lmchain.load.serializable import Serializable
from pydantic.v1 import Extra, Field,BaseModel, PrivateAttr  # 导入pydantic库的BaseModel和PrivateAttr，用于数据验证和私有属性定义


if TYPE_CHECKING:
    from langchain.prompts.chat import ChatPromptTemplate


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Convert sequence of Messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of HumanMessages.
        ai_prefix: THe prefix to prepend to contents of AIMessages.

    Returns:
        A single string concatenation of all input messages.

    Example:
        .. code-block:: python

            from langchain.schema import AIMessage, HumanMessage

            messages = [
                HumanMessage(content="Hi, how are you?"),
                AIMessage(content="Good, how are you?"),
            ]
            get_buffer_string(messages)
            # -> "Human: Hi, how are you?\nAI: Good, how are you?"
    """
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        message = f"{role}: {m.content}"
        if isinstance(m, AIMessage) and "function_call" in m.additional_kwargs:
            message += f"{m.additional_kwargs['function_call']}"
        string_messages.append(message)

    return "\n".join(string_messages)

#这段代码定义了一个名为 BaseMessage 的抽象基类，该类继承自 Serializable。这个类代表了一个基本的消息对象，该对象可以作为聊天模型的输入和输出。
#BaseMessage 类代表了一个基本的消息对象，具有 content（消息内容）、additional_kwargs（额外信息）和 type（消息类型）等属性。
# 该类还实现了 is_lc_serializable 方法用于判断类是否可序列化，并实现了 __add__ 方法以支持对象间的加法运算。

# 定义一个名为 BaseMessage 的类，继承自 Serializable 类。
class BaseMessage(Serializable):
    """The base abstract Message class.

    Messages are the inputs and outputs of ChatModels.
    """

    # 定义一个名为 content 的实例变量，类型为 str，表示消息的内容。
    content: str
    """The string contents of the message."""

    # 定义一个名为 additional_kwargs 的实例变量，类型为 dict，使用 Field 的 default_factory 参数为其设置默认值（一个空的字典）。
    # 这个变量用于存储任何额外的信息。
    additional_kwargs: dict = Field(default_factory=dict)
    """Any additional information."""

    # 定义一个名为 type 的实例变量，类型为 str，表示消息的类型。
    type: str

    # 定义一个类内部的 Config 配置类。
    class Config:
        # 设置 extra 为 Extra.allow，表示允许在反序列化时忽略额外的字段。
        extra = Extra.allow

    # 类方法，返回一个布尔值，表示这个类是否可序列化。在这个类中，始终返回 True。
    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    # 定义 __add__ 方法，使得 BaseMessage 对象可以通过 + 运算符与其他对象相加。
    # 这里使用了 Any 类型注解，表示 other 可以是任何类型的对象。
    def __add__(self, other: Any) -> ChatPromptTemplate:
        # 从 langchain.prompts.chat 模块中导入 ChatPromptTemplate 类。
        from lmchain.prompts.chat import ChatPromptTemplate

        # 创建一个 ChatPromptTemplate 对象，将当前的 BaseMessage 对象作为 messages 参数传入。
        prompt = ChatPromptTemplate(messages=[self])
        # 返回 prompt 与 other 相加的结果。这里假设 ChatPromptTemplate 类已经实现了 __add__ 方法。
        return prompt + other


class BaseMessageChunk(BaseMessage):
    """A Message chunk, which can be concatenated with other Message chunks."""

    def _merge_kwargs_dict(
        self, left: Dict[str, Any], right: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge additional_kwargs from another BaseMessageChunk into this one."""
        merged = left.copy()
        for k, v in right.items():
            if k not in merged:
                merged[k] = v
            elif type(merged[k]) != type(v):
                raise ValueError(
                    f'additional_kwargs["{k}"] already exists in this message,'
                    " but with a different type."
                )
            elif isinstance(merged[k], str):
                merged[k] += v
            elif isinstance(merged[k], dict):
                merged[k] = self._merge_kwargs_dict(merged[k], v)
            else:
                raise ValueError(
                    f"Additional kwargs key {k} already exists in this message."
                )
        return merged

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, BaseMessageChunk):
            # If both are (subclasses of) BaseMessageChunk,
            # concat into a single BaseMessageChunk

            if isinstance(self, ChatMessageChunk):
                return self.__class__(
                    role=self.role,
                    content=self.content + other.content,
                    additional_kwargs=self._merge_kwargs_dict(
                        self.additional_kwargs, other.additional_kwargs
                    ),
                )
            return self.__class__(
                content=self.content + other.content,
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )
        else:
            raise TypeError(
                'unsupported operand type(s) for +: "'
                f"{self.__class__.__name__}"
                f'" and "{other.__class__.__name__}"'
            )


class HumanMessage(BaseMessage):
    """A Message from a human."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["human"] = "human"


HumanMessage.update_forward_refs()


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """A Human Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment] # noqa: E501


class AIMessage(BaseMessage):
    """A Message from an AI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    type: Literal["ai"] = "ai"


AIMessage.update_forward_refs()


class AIMessageChunk(AIMessage, BaseMessageChunk):
    """A Message chunk from an AI."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["AIMessageChunk"] = "AIMessageChunk"  # type: ignore[assignment] # noqa: E501

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, AIMessageChunk):
            if self.example != other.example:
                raise ValueError(
                    "Cannot concatenate AIMessageChunks with different example values."
                )

            return self.__class__(
                example=self.example,
                content=self.content + other.content,
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class SystemMessage(BaseMessage):
    """A Message for priming AI behavior, usually passed in as the first of a sequence
    of input messages.
    """

    type: Literal["system"] = "system"


SystemMessage.update_forward_refs()


class SystemMessageChunk(SystemMessage, BaseMessageChunk):
    """A System Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["SystemMessageChunk"] = "SystemMessageChunk"  # type: ignore[assignment] # noqa: E501


class FunctionMessage(BaseMessage):
    """A Message for passing the result of executing a function back to a model."""

    name: str
    """The name of the function that was executed."""

    type: Literal["function"] = "function"


FunctionMessage.update_forward_refs()


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """A Function Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["FunctionMessageChunk"] = "FunctionMessageChunk"  # type: ignore[assignment]

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                raise ValueError(
                    "Cannot concatenate FunctionMessageChunks with different names."
                )

            return self.__class__(
                name=self.name,
                content=self.content + other.content,
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


class ChatMessage(BaseMessage):
    """A Message that can be assigned an arbitrary speaker (i.e. role)."""

    role: str
    """The speaker / role of the Message."""

    type: Literal["chat"] = "chat"


ChatMessage.update_forward_refs()


class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """A Chat Message chunk."""

    # Ignoring mypy re-assignment here since we're overriding the value
    # to make sure that the chunk variant can be discriminated from the
    # non-chunk variant.
    type: Literal["ChatMessageChunk"] = "ChatMessageChunk"  # type: ignore

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore
        if isinstance(other, ChatMessageChunk):
            if self.role != other.role:
                raise ValueError(
                    "Cannot concatenate ChatMessageChunks with different roles."
                )

            return self.__class__(
                role=self.role,
                content=self.content + other.content,
                additional_kwargs=self._merge_kwargs_dict(
                    self.additional_kwargs, other.additional_kwargs
                ),
            )

        return super().__add__(other)


AnyMessage = Union[AIMessage, HumanMessage, ChatMessage, SystemMessage, FunctionMessage]


def _message_to_dict(message: BaseMessage) -> dict:
    return {"type": message.type, "data": message.dict()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> List[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as BaseMessages) to convert.

    Returns:
        List of messages as dicts.
    """
    return [_message_to_dict(m) for m in messages]


def _message_from_dict(message: dict) -> BaseMessage:
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "chat":
        return ChatMessage(**message["data"])
    elif _type == "function":
        return FunctionMessage(**message["data"])
    else:
        raise ValueError(f"Got unexpected message type: {_type}")


def messages_from_dict(messages: List[dict]) -> List[BaseMessage]:
    """Convert a sequence of messages from dicts to Message objects.

    Args:
        messages: Sequence of messages (as dicts) to convert.

    Returns:
        List of messages (BaseMessages).
    """
    return [_message_from_dict(m) for m in messages]
