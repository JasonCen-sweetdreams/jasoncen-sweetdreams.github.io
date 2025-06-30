# 基于Placeholder的阻塞

## placeholder的阻塞行为

1. 初始化
   - 使用一个 `RpcAgentClient`来创建 `PlaceholderMessage` 时，会通过 `call_in_thread` **立即在后台线程中发起一个RPC请求**
   - 构造函数本身**几乎立即返回**一个 `PlaceholderMessage` 实例，这个实例的 `_is_placeholder` 标记为 `True`，并且它持有一个 `_stub`，该存根（stub）代表了后台操作的未来结果。
   - 这个创建过程对于主调用线程是非阻塞的。
2. **获取占位符的属性 (`__getattr__` 或 `__getitem__`)**：
   - 当尝试访问 `PlaceholderMessage` 实例的某个属性时（例如 `msg.content` 或 `msg['content']`）：
     - 它首先通过 `__is_local()` 检查这个属性是否是“本地的”（例如 `name`, `timestamp`, `id` 或内部的 `_host`, `_port`, `_is_placeholder` 等），或者这个消息是否已经不再是占位符（`_is_placeholder` 为 `False`）。
     - 如果属性**不是本地的**，并且 `_is_placeholder` **仍然为 `True`**，那么它就会调用 `self.update_value()` 方法。
3. **`update_value()` 方法 (潜在的第一个阻塞点)**：
   - 这个方法负责从远程服务器获取真实的消息数据。
   - 首先调用 `self.__update_task_id()`。
4. **`__update_task_id()` 方法 (第一个实际的阻塞点)**：
   - 如果 `self._stub`存在（意味着初始的RPC调用已在后台启动）：
     - 会调用 `self._stub.get_response()`——**阻塞执行**，直到后台线程完成其任务并返回一个包含 `task_id` 的响应。
   - 获取到 `task_id` 后，`_stub` 通常会被置为 `None`，表示其使命已完成。
5. **`update_value()` 方法续 (第二个实际的阻塞点)**：
   - 拿到 `task_id` 后，`update_value()` 会创建一个新的 `RpcAgentClient` (或者使用已有的，根据具体实现)，然后**发起一个同步的、阻塞的RPC调用**来根据 `task_id` 获取完整的消息数据。
   - 获取到数据后，它会用真实数据更新 `PlaceholderMessage` 实例自身的属性，并将 `_is_placeholder` 设置为 `False`。

## 阻塞与非阻塞的对比

- 非阻塞
  - **创建 `PlaceholderMessage` 实例时**：当提供了 `client` 参数进行初始化，实际的耗时操作在后台线程启动，主线程几乎立即获得 `PlaceholderMessage` 对象。
  - **访问本地属性时**：访问 `PlaceholderMessage.LOCAL_ATTRS` 中定义的属性（如 `name`, `timestamp`, `id`, `_is_placeholder`, `_host`, `_port`）不会触发 `update_value()`，因此是非阻塞的。
- 阻塞
  - **首次访问非本地属性时**：当 `PlaceholderMessage` 仍是占位符 (`_is_placeholder is True`)，并且你尝试访问一个不在 `LOCAL_ATTRS` 中的属性（最典型的就是 `content`，也可能是 `role` 或其他需要从远程获取的业务特定属性），会==触发== `update_value()`。
  - **在 `update_value()` 内部**:
    - `__update_task_id()` 中的 `_stub.get_response()` 会阻塞，等待后台线程返回初步结果（如 `task_id`）。
    - 之后 `client.call_func(func_name="_get", ...)` 会阻塞，等待从RPC服务器获取完整的消息数据。
  - **显式调用 `update_value()` 时**: 如果代码直接调用这个方法，它自然会执行上述阻塞逻辑（如果消息仍是占位符）。
  - **调用 `serialize()` 时 (如果需要 `task_id`)**: 如果 `PlaceholderMessage` 仍是占位符且 `_task_id` 未知，但 `_stub` 存在，`serialize()` 内部会调用 `__update_task_id()`，这也可能导致阻塞。







# 源码

## MessageBase

```python
# -*- coding: utf-8 -*-
"""The base class for message unit"""

from typing import Any, Optional, Union, Sequence, Literal
from uuid import uuid4
import json

from loguru import logger

from .rpc import RpcAgentClient, ResponseStub, call_in_thread
from .utils.tools import _get_timestamp


class MessageBase(dict):
    """Base Message class, which is used to maintain information for dialog,
    memory and used to construct prompt.
    """

    def __init__(
        self,
        name: str,
        content: Any,
        role: Literal["user", "system", "assistant"] = "assistant",
        url: Optional[Union[Sequence[str], str]] = None,
        timestamp: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the message object

        Args:
            name (`str`):
                The name of who send the message. It's often used in
                role-playing scenario to tell the name of the sender.
            content (`Any`):
                The content of the message.
            role (`Literal["system", "user", "assistant"]`, defaults to "assistant"):
                The role of who send the message. It can be one of the
                `"system"`, `"user"`, or `"assistant"`. Default to
                `"assistant"`.
            url (`Optional[Union[list[str], str]]`, defaults to None):
                A url to file, image, video, audio or website.
            timestamp (`Optional[str]`, defaults to None):
                The timestamp of the message, if None, it will be set to
                current time.
            **kwargs (`Any`):
                Other attributes of the message.
        """  # noqa
        # id and timestamp will be added to the object as its attributes
        # rather than items in dict
        self.id = uuid4().hex
        if timestamp is None:
            self.timestamp = _get_timestamp()
        else:
            self.timestamp = timestamp

        self.name = name
        self.content = content
        self.role = role

        if url:
            self.url = url
        else:
            self.url = None

        self.update(kwargs)

    def __getattr__(self, key: Any) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"no attribute '{key}'") from e

    def __setattr__(self, key: Any, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: Any) -> None:
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(f"no attribute '{key}'") from e

    def to_str(self) -> str:
        """Return the string representation of the message"""
        raise NotImplementedError

    def serialize(self) -> str:
        """Return the serialized message."""
        raise NotImplementedError
```

## Msg

```python
class Msg(MessageBase):
    """The Message class."""

    def __init__(
        self,
        name: str,
        content: Any,
        role: Literal["system", "user", "assistant"] = None,
        url: Optional[Union[Sequence[str], str]] = None,
        timestamp: Optional[str] = None,
        echo: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the message object

        Args:
            name (`str`):
                The name of who send the message.
            content (`Any`):
                The content of the message.
            role (`Literal["system", "user", "assistant"]`):
                Used to identify the source of the message, e.g. the system
                information, the user input, or the model response. This
                argument is used to accommodate most Chat API formats.
            url (`Optional[Union[list[str], str]]`, defaults to `None`):
                A url to file, image, video, audio or website.
            timestamp (`Optional[str]`, defaults to `None`):
                The timestamp of the message, if None, it will be set to
                current time.
            **kwargs (`Any`):
                Other attributes of the message.
        """

        if role is None:
            logger.warning(
                "A new field `role` is newly added to the message. "
                "Please specify the role of the message. Currently we use "
                'a default "assistant" value.',
            )

        super().__init__(
            name=name,
            content=content,
            role=role or "assistant",
            url=url,
            timestamp=timestamp,
            **kwargs,
        )
        if echo:
            logger.chat(self)

    def to_str(self) -> str:
        """Return the string representation of the message"""
        return f"{self.name}: {self.content}"

    def serialize(self) -> str:
        return json.dumps({"__type": "Msg", **self})


class Tht(MessageBase):
    """The Thought message is used to record the thought of the agent to
    help them make decisions and responses. Generally, it shouldn't be
    passed to or seen by the other agents.

    In our framework, we formulate the thought in prompt as follows:
    - For OpenAI API calling:

    .. code-block:: python

        [
            ...
            {
                "role": "assistant",
                "name": "thought",
                "content": "I should ..."
            },
            ...
        ]

    - For open-source models that accepts string as input:

    .. code-block:: python

        ...
        {self.name} thought: I should ...
        ...

    We admit that there maybe better ways to formulate the thought. Users
    are encouraged to create their own thought formulation methods by
    inheriting `MessageBase` class and rewrite the `__init__` and `to_str`
    function.

    .. code-block:: python

        class MyThought(MessageBase):
            def to_str(self) -> str:
                # implement your own thought formulation method
                pass
    """

    def __init__(
        self,
        content: Any,
        timestamp: Optional[str] = None,
    ) -> None:
        super().__init__(
            name="thought",
            content=content,
            role="assistant",
            timestamp=timestamp,
        )

    def to_str(self) -> str:
        """Return the string representation of the message"""
        return f"{self.name} thought: {self.content}"

    def serialize(self) -> str:
        return json.dumps({"__type": "Tht", **self})
```

## Placeholder

```python
class PlaceholderMessage(MessageBase):
    """A placeholder for the return message of RpcAgent."""

    PLACEHOLDER_ATTRS = {
        "_host",
        "_port",
        "_client",
        "_task_id",
        "_stub",
        "_is_placeholder",
    }

    LOCAL_ATTRS = {
        "name",
        "timestamp",
        *PLACEHOLDER_ATTRS,
    }

    def __init__(
        self,
        name: str,
        content: Any,
        url: Optional[Union[Sequence[str], str]] = None,
        timestamp: Optional[str] = None,
        host: str = None,
        port: int = None,
        task_id: int = None,
        client: Optional[RpcAgentClient] = None,
        x: dict = None,
        **kwargs: Any,
    ) -> None:
        """A placeholder message, records the address of the real message.

        Args:
            name (`str`):
                The name of who send the message. It's often used in
                role-playing scenario to tell the name of the sender.
                However, you can also only use `role` when calling openai api.
                The usage of `name` refers to
                https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models.
            content (`Any`):
                The content of the message.
            role (`Literal["system", "user", "assistant"]`, defaults to "assistant"):
                The role of the message, which can be one of the `"system"`,
                `"user"`, or `"assistant"`.
            url (`Optional[Union[list[str], str]]`, defaults to None):
                A url to file, image, video, audio or website.
            timestamp (`Optional[str]`, defaults to None):
                The timestamp of the message, if None, it will be set to
                current time.
            host (`str`, defaults to `None`):
                The hostname of the rpc server where the real message is
                located.
            port (`int`, defaults to `None`):
                The port of the rpc server where the real message is located.
            task_id (`int`, defaults to `None`):
                The task id of the real message in the rpc server.
            client (`RpcAgentClient`, defaults to `None`):
                An RpcAgentClient instance used to connect to the generator of
                this placeholder.
            x (`dict`, defaults to `None`):
                Input parameters used to call rpc methods on the client.
        """  # noqa
        super().__init__(
            name=name,
            content=content,
            url=url,
            timestamp=timestamp,
            **kwargs,
        )
        # placeholder indicates whether the real message is still in rpc server
        self._is_placeholder = True
        if client is None:
            self._stub: ResponseStub = None
            self._host: str = host
            self._port: int = port
            self._task_id: int = task_id
        else:
            self._stub = call_in_thread(
                client,
                x.serialize() if x is not None else "",
                "_reply",
            )
            self._host = client.host
            self._port = client.port
            self._task_id = None

    def __is_local(self, key: Any) -> bool:
        return (
            key in PlaceholderMessage.LOCAL_ATTRS or not self._is_placeholder
        )

    def __getattr__(self, __name: str) -> Any:
        """Get attribute value from PlaceholderMessage. Get value from rpc
        agent server if necessary.

        Args:
            __name (`str`):
                Attribute name.
        """
        if not self.__is_local(__name):
            self.update_value()
        return MessageBase.__getattr__(self, __name)

    def __getitem__(self, __key: Any) -> Any:
        """Get item value from PlaceholderMessage. Get value from rpc
        agent server if necessary.

        Args:
            __key (`Any`):
                Item name.
        """
        if not self.__is_local(__key):
            self.update_value()
        return MessageBase.__getitem__(self, __key)

    def to_str(self) -> str:
        return f"{self.name}: {self.content}"

    def update_value(self) -> MessageBase:
        """Get attribute values from rpc agent server immediately"""
        if self._is_placeholder:
            # retrieve real message from rpc agent server
            self.__update_task_id()
            client = RpcAgentClient(self._host, self._port)
            result = client.call_func(
                func_name="_get",
                value=json.dumps({"task_id": self._task_id}),
            )
            msg = deserialize(result)
            status = msg.pop("__status", "OK")
            if status == "ERROR":
                raise RuntimeError(msg.content)
            self.update(msg)
            # the actual value has been updated, not a placeholder any more
            self._is_placeholder = False
        return self

    def __update_task_id(self) -> None:
        if self._stub is not None:
            try:
                resp = deserialize(self._stub.get_response())
            except Exception as e:
                logger.error(
                    f"Failed to get task_id: {self._stub.get_response()}",
                )
                raise ValueError(
                    f"Failed to get task_id: {self._stub.get_response()}",
                ) from e
            self._task_id = resp["task_id"]  # type: ignore[call-overload]
            self._stub = None

    def serialize(self) -> str:
        if self._is_placeholder:
            self.__update_task_id()
            return json.dumps(
                {
                    "__type": "PlaceholderMessage",
                    "name": self.name,
                    "content": None,
                    "timestamp": self.timestamp,
                    "host": self._host,
                    "port": self._port,
                    "task_id": self._task_id,
                },
            )
        else:
            states = {
                k: v
                for k, v in self.items()
                if k not in PlaceholderMessage.PLACEHOLDER_ATTRS
            }
            states["__type"] = "Msg"
            return json.dumps(states)


_MSGS = {
    "Msg": Msg,
    "Tht": Tht,
    "PlaceholderMessage": PlaceholderMessage,
}


def deserialize(s: str) -> Union[MessageBase, Sequence]:
    """Deserialize json string into MessageBase"""
    js_msg = json.loads(s)
    msg_type = js_msg.pop("__type")
    if msg_type == "List":
        return [deserialize(s) for s in js_msg["__value"]]
    elif msg_type not in _MSGS:
        raise NotImplementedError(
            "Deserialization of {msg_type} is not supported.",
        )
    return _MSGS[msg_type](**js_msg)


def serialize(messages: Union[Sequence[MessageBase], MessageBase]) -> str:
    """Serialize multiple MessageBase instance"""
    if isinstance(messages, MessageBase):
        return messages.serialize()
    seq = [msg.serialize() for msg in messages]
    return json.dumps({"__type": "List", "__value": seq})
```

## response stub & call in thread function

`rpc_agent_client.py`

```python
class ResponseStub:
    """A stub used to save the response of an rpc call in a sub-thread."""

    def __init__(self) -> None:
        self.response = None
        self.condition = threading.Condition()

    def set_response(self, response: str) -> None:
        """Set the message."""
        with self.condition:
            self.response = response
            self.condition.notify_all()

    def get_response(self) -> str:
        """Get the message."""
        with self.condition:
            while self.response is None:
                self.condition.wait()
            return self.response


def call_in_thread(
    client: RpcAgentClient,
    value: str,
    func_name: str,
) -> ResponseStub:
    """Call rpc function in a sub-thread.

    Args:
        client (`RpcAgentClient`): the rpc client.
        x (`str`): the value of the reqeust.
        func_name (`str`): the name of the function being called.

    Returns:
        `ResponseStub`: a stub to get the response.
    """
    stub = ResponseStub()

    def wrapper() -> None:
        try:
            resp = client.call_func(
                func_name=func_name,
                value=value,
            )
            stub.set_response(resp)  # type: ignore[arg-type]
        except RpcError as e:
            logger.error(f"Fail to call {func_name} in thread: {e}")
            stub.set_response(str(e))

    thread = threading.Thread(target=wrapper)
    thread.start()
    return stub
```

