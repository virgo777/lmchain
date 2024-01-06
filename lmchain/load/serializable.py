# 导入必要的模块和类型
from abc import ABC  # 导入抽象基类模块
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast  # 导入类型注解相关的模块

from pydantic.v1 import BaseModel, PrivateAttr  # 导入pydantic库的BaseModel和PrivateAttr，用于数据验证和私有属性定义


# 定义一个TypedDict基类，用于类型化字典
class BaseSerialized(TypedDict):
    """Base class for serialized objects."""  # 基类用于序列化对象

    lc: int  # 字典必须包含一个名为'lc'的整数键
    id: List[str]  # 字典必须包含一个名为'id'的字符串列表键


# 定义几个特定的序列化类型
class SerializedConstructor(BaseSerialized):
    """Serialized constructor."""  # 序列化的构造函数类型

    type: Literal["constructor"]  # 必须有一个名为'type'的键，其值为"constructor"
    kwargs: Dict[str, Any]  # 必须有一个名为'kwargs'的键，其值为任意类型的字典


class SerializedSecret(BaseSerialized):
    """Serialized secret."""  # 序列化的秘密类型

    type: Literal["secret"]  # 必须有一个名为'type'的键，其值为"secret"


class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented."""  # 序列化的未实现类型

    type: Literal["not_implemented"]  # 必须有一个名为'type'的键，其值为"not_implemented"
    repr: Optional[str]  # 可以有一个可选的名为'repr'的字符串键


# 定义一个辅助函数，用于比较模型字段的默认值与给定值是否不相等
def try_neq_default(value: Any, key: str, model: BaseModel) -> bool:
    try:
        return model.__fields__[key].get_default() != value  # 比较默认值与给定值
    except Exception:  # 如果发生异常，则返回True，表示值不等于默认值
        return True

    # 定义一个可序列化的抽象基类，继承自pydantic的BaseModel和ABC


class Serializable(BaseModel, ABC):
    """Serializable（可串行化，可序列化）是一个接口，通常定义一个实体类的时候会实现它，  
    目的是为了告诉JVM（Java Virtual Machine）这个类的对象可以被序列化。  
    序列化是将对象状态转化为可保持或者传输的格式过程，与序列化相反的是反序列化。  
    完成序列化和反序列化，可以存储或传输数据。 base class."""  # 类说明文档字符串

    @classmethod  # 类方法，用于判断该类是否可序列化
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?"""  # 方法说明文档字符串
        return False  # 默认返回False，表示不可序列化，子类可以重写此方法以改变行为

    @classmethod  # 类方法，用于获取对象的命名空间
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the lmchain object."""  # 方法说明文档字符串
        return cls.__module__.split(".")  # 返回模块的命名空间，以点分割的模块名列表

    @property  # 属性方法，用于获取构造函数的秘密参数映射
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids."""  # 属性说明文档字符串
        return dict()  # 默认返回空字典，子类可以重写此属性以提供秘密参数映射

    @property  # 属性方法，用于获取应包含在序列化kwargs中的属性名列表
    def lc_attributes(self) -> Dict:
        """List of attribute names that should be included in the serialized kwargs.  
        These attributes must be accepted by the constructor."""  # 属性说明文档字符串
        return {}  # 默认返回空字典，子类可以重写此属性以提供属性名列表

    @classmethod  # 类方法，用于获取类的唯一标识符，用于序列化目的
    def lc_id(cls) -> List[str]:
        """A unique identifier for this class for serialization purposes.  
        The unique identifier is a list of strings that describes the path to the object."""  # 方法说明文档字符串
        return [*cls.get_lc_namespace(), cls.__name__]  # 返回包含命名空间和类名的唯一标识符列表

    class Config:  # pydantic配置类，用于配置模型的额外行为
        extra = "ignore"  # 设置额外属性为忽略，即模型不会包含未在字段中定义的属性

    def __repr_args__(self) -> Any:  # 定义对象的字符串表示所需的参数列表
        return [
            (k, v)  # 返回包含键值对的列表，其中键是字段名，值是字段值（如果字段值不等于默认值）
            for k, v in super().__repr_args__()  # 调用父类的__repr_args__方法获取原始参数列表
            if (k not in self.__fields__ or try_neq_default(v, k, self))  # 如果字段不在模型中或字段值不等于默认值则保留该字段
        ]

    _lc_kwargs = PrivateAttr(default_factory=dict)  # 定义一个私有属性，用于存储构造函数的参数（默认为空字典）

    def __init__(self, **kwargs: Any) -> None:  # 定义构造函数，接受任意关键字参数并存储在私有属性中
        super().__init__(**kwargs)  # 调用父类的构造函数进行初始化（包括数据验证）
        self._lc_kwargs = kwargs  # 将传入的参数存储在私有属性中，供后续序列化使用

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        if not self.is_lc_serializable():
            return self.to_json_not_implemented()

        secrets = dict()
        # Get latest values for kwargs if there is an attribute with same name
        lc_kwargs = {
            k: getattr(self, k, v)
            for k, v in self._lc_kwargs.items()
            if not (self.__exclude_fields__ or {}).get(k, False)  # type: ignore
        }

        # Merge the lc_secrets and lc_attributes from every class in the MRO
        for cls in [None, *self.__class__.mro()]:
            # Once we get to Serializable, we're done
            if cls is Serializable:
                break

            if cls:
                deprecated_attributes = [
                    "lc_namespace",
                    "lc_serializable",
                ]

                for attr in deprecated_attributes:
                    if hasattr(cls, attr):
                        raise ValueError(
                            f"Class {self.__class__} has a deprecated "
                            f"attribute {attr}. Please use the corresponding "
                            f"classmethod instead."
                        )

            # Get a reference to self bound to each class in the MRO
            this = cast(Serializable, self if cls is None else super(cls, self))

            secrets.update(this.lc_secrets)
            lc_kwargs.update(this.lc_attributes)

        # include all secrets, even if not specified in kwargs
        # as these secrets may be passed as an environment variable instead
        for key in secrets.keys():
            secret_value = getattr(self, key, None) or lc_kwargs.get(key)
            if secret_value is not None:
                lc_kwargs.update({key: secret_value})

        return {
            "lc": 1,
            "type": "constructor",
            "id": self.lc_id(),
            "kwargs": lc_kwargs
            if not secrets
            else _replace_secrets(lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        return to_json_not_implemented(self)


def _replace_secrets(
    root: Dict[Any, Any], secrets_map: Dict[str, str]
) -> Dict[Any, Any]:
    result = root.copy()
    for path, secret_id in secrets_map.items():
        [*parts, last] = path.split(".")
        current = result
        for part in parts:
            if part not in current:
                break
            current[part] = current[part].copy()
            current = current[part]
        if last in current:
            current[last] = {
                "lc": 1,
                "type": "secret",
                "id": [secret_id],
            }
    return result


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """Serialize a "not implemented" object.

    Args:
        obj: object to serialize

    Returns:
        SerializedNotImplemented
    """
    _id: List[str] = []
    try:
        if hasattr(obj, "__name__"):
            _id = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            _id = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        pass

    result: SerializedNotImplemented = {
        "lc": 1,
        "type": "not_implemented",
        "id": _id,
        "repr": None,
    }
    try:
        result["repr"] = repr(obj)
    except Exception:
        pass
    return result

