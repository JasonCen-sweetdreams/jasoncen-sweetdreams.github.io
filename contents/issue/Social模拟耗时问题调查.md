# 耗时问题

## 问题描述

- 在加入了Group群体行为后，整个模拟任务的单次模拟迭代耗时为原来的5-10倍（1~2小时——10小时）

## 计时工具

- 利用Decorator进行打点计时

  - 在utils中定义一个timing工具

    ```python
    import time
    from functools import wraps
    
    def timed(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            print(f"[TIMER] {func.__name__} took {elapsed:.4f} s")
            return result
        return wrapper
    ```


## 问题分析

### 初步分析

- 通过小规模测试对比得到的消耗时间统计（actor agent规模为200左右）

| Function                       | Origin Time | New Method Time |
| ------------------------------ | ----------- | --------------- |
| update_agents                  | 20.3982s    | 18.6799s        |
| collect_agent_plans            | 53.6487s    | 54.1081s        |
| social（包含social_one_agent） | 107.0372s   | 1197.3274s      |
| update_social_manager          | 0.1623s     | 2.4222s         |
| step                           | 222.6959s   | 1253.9943s      |

- 基本将问题定位在Group改写的部分

### 分析1

- 按照目前的group架构，理论上串行的部分应该Group内的leader+follower方法
  - 将非必要串行的部分并行化
    - follower的自身行为
    - 不同follower的行为
  - 结果：social模拟耗时降低至：995.8990s
  - 一轮完整模拟的耗时
    - 降低至8小时
    - 比原来要低2~3小时

| Function                       | Origin Time | New Method Time | After  Opti. |
| ------------------------------ | ----------- | --------------- | ------------ |
| update_agents                  | 20.3982s    | 18.6799s        | 64.8902s     |
| collect_agent_plans            | 53.6487s    | 54.1081s        | 24.6517s     |
| social（包含social_one_agent） | 107.0372s   | 1197.3274s      | 931.6033s    |
| update_social_manager          | 0.1623s     | 2.4222s         | 1.1830s      |
| step                           | 222.6959s   | 1253.9943s      | 1022.3491s   |

### 分析2

- group agent的在调用agent的function时主动阻塞了通信

  ```python
      def call_agent_func(self,
                          agent:ArticleAgentWrapper,
                          func_name:str,
                          kwargs:dict={}) -> Msg:
          msg = Msg("user",
                  content="call_function",
                  role="assistant",
                  kwargs=kwargs,
                  func=func_name
                  )
          return_msg = agent(msg)
          ### 阻塞
          if isinstance(return_msg,PlaceholderMessage):
              return_msg.update_value()
          return return_msg
  ```

- 尝试：注释掉阻塞的代码

- 结果：无效

### 分析3

- 分析`wrapper`并行的实现：传入了`to_dist`参数
  - 作为基类的`AgentBase`及其`metaclass` `_AgentMeta`处理了这个参数
  - 在`_AgentMeta`中处理了
- Group agent基于`AgentBase`进行开发并传入`to_dist`参数
  - 完整的一轮模拟耗时为约2小时，在预期的范围内
  - 问题解决！

| Function                       | Origin Time | New Method Time | After  Opti.1 | After  Opti.2 |
| ------------------------------ | ----------- | --------------- | ------------- | ------------- |
| update_agents                  | 20.3982s    | 18.6799s        | 24.6517s      | 19.3416s      |
| collect_agent_plans            | 53.6487s    | 54.1081s        | 64.8902s      | 61.3883s      |
| social（包含social_one_agent） | 107.0372s   | 1197.3274s      | 931.6033s     | 196.5567s     |
| update_social_manager          | 0.1623s     | 2.4222s         | 1.1830s       | 0.5044s       |
| step                           | 222.6959s   | 1253.9943s      | 1022.3491s    | 258.5630s     |