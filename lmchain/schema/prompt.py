# 这段代码定义了一个名为 PromptValue 的抽象基类，该类用于表示任何语言模型的输入。
# 这个类继承自 Serializable 和 ABC（Abstract Base Class），意味着它是一个可序列化的抽象基类。


# 导入 __future__ 模块中的 annotations 功能，使得在 Python 3.7 以下版本中也可以使用类型注解的延迟评估功能。
from __future__ import annotations

# 导入 abc 模块中的 ABC（抽象基类）和 abstractmethod（抽象方法）装饰器。
from abc import ABC, abstractmethod
# 导入 typing 模块中的 List 类型，用于类型注解。
from typing import List

# 从 lmchain.load.serializable 模块中导入 Serializable 类，用于序列化和反序列化对象。
from lmchain.load.serializable import Serializable
# 从 lmchain.schema.messages 模块中导入 BaseMessage 类，作为消息基类。
from lmchain.schema.messages import BaseMessage


# 定义一个名为 PromptValue 的抽象基类，继承自 Serializable 和 ABC。
class PromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
        ChatModel inputs.
    """

    # 类方法，返回一个布尔值，表示这个类是否可序列化。在这个类中，始终返回 True。
    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    # 抽象方法，需要子类实现。返回一个字符串，表示 prompt 的值。
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt value as string."""

    # 抽象方法，需要子类实现。返回一个 BaseMessage 对象的列表，表示 prompt。
    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as a list of Messages."""
