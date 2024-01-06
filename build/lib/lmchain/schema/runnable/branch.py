from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain.load.dump import dumpd
from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain.schema.runnable.config import (
    RunnableConfig,
    ensure_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain.schema.runnable.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)


class RunnableBranch(RunnableSerializable[Input, Output]):
    """A Runnable that selects which branch to run based on a condition.

    The runnable is initialized with a list of (condition, runnable) pairs and
    a default branch.

    When operating on an input, the first condition that evaluates to True is
    selected, and the corresponding runnable is run on the input.

    If no condition evaluates to True, the default branch is run on the input.

    Examples:

        .. code-block:: python

            from langchain.schema.runnable import RunnableBranch

            branch = RunnableBranch(
                (lambda x: isinstance(x, str), lambda x: x.upper()),
                (lambda x: isinstance(x, int), lambda x: x + 1),
                (lambda x: isinstance(x, float), lambda x: x * 2),
                lambda x: "goodbye",
            )

            branch.invoke("hello") # "HELLO"
            branch.invoke(None) # "goodbye"
    """

    branches: Sequence[Tuple[Runnable[Input, bool], Runnable[Input, Output]]]
    default: Runnable[Input, Output]

    def __init__(
        self,
        *branches: Union[
            Tuple[
                Union[
                    Runnable[Input, bool],
                    Callable[[Input], bool],
                    Callable[[Input], Awaitable[bool]],
                ],
                RunnableLike,
            ],
            RunnableLike,  # To accommodate the default branch
        ],
    ) -> None:
        """A Runnable that runs one of two branches based on a condition."""
        if len(branches) < 2:
            raise ValueError("RunnableBranch requires at least two branches")

        default = branches[-1]

        if not isinstance(
            default,
            (Runnable, Callable, Mapping),  # type: ignore[arg-type]
        ):
            raise TypeError(
                "RunnableBranch default must be runnable, callable or mapping."
            )

        default_ = cast(
            Runnable[Input, Output], coerce_to_runnable(cast(RunnableLike, default))
        )

        _branches = []

        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):  # type: ignore[arg-type]
                raise TypeError(
                    f"RunnableBranch branches must be "
                    f"tuples or lists, not {type(branch)}"
                )

            if not len(branch) == 2:
                raise ValueError(
                    f"RunnableBranch branches must be "
                    f"tuples or lists of length 2, not {len(branch)}"
                )
            condition, runnable = branch
            condition = cast(Runnable[Input, bool], coerce_to_runnable(condition))
            runnable = coerce_to_runnable(runnable)
            _branches.append((condition, runnable))

        super().__init__(branches=_branches, default=default_)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """RunnableBranch is serializable if all its branches are serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """The namespace of a RunnableBranch is the namespace of its default branch."""
        return cls.__module__.split(".")[:-1]

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        runnables = (
            [self.default]
            + [r for _, r in self.branches]
            + [r for r, _ in self.branches]
        )

        for runnable in runnables:
            if runnable.get_input_schema(config).schema().get("type") is not None:
                return runnable.get_input_schema(config)

        return super().get_input_schema(config)

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec
            for step in (
                [self.default]
                + [r for _, r in self.branches]
                + [r for r, _ in self.branches]
            )
            for spec in step.config_specs
        )

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """First evaluates the condition, then delegate to true or false branch."""
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
        )

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = runnable.invoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = self.default.invoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except Exception as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(dumpd(output))
        return output

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Async version of invoke."""
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
        )
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = await runnable.ainvoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = await self.default.ainvoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except Exception as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(dumpd(output))
        return output
