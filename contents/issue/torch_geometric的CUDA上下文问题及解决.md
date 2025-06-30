# 一、加入torch geometric后出现的CUDA问题

CUDA initialize错误的代码定位：`embeddings = HuggingFaceEmbeddings(model_name="/XYFS01/nsccgz_ywang_wzh/cenjc/all-MiniLM-L6-v2")`

## 基本排查

- 新的环境

  - import torch_geometric后导致CUDA出问题

  - 相关库版本

    ```shell
    torch                             2.5.1
    torch-geometric                   2.6.1
    ```

  - requirement中的torch版本：torch==2.2.2

- 运行校验（新环境下）

  - 老版本代码（modified）
    - 正常运行
  - python脚本独立检验
    - 正常运行
  - 结论：应该与库版本没有关系

## 进一步调查

### 多进程情况下

> https://blog.csdn.net/qq_39779233/article/details/144546144

- 基本确定是torch_geometric 多进程下的CUDA问题

#### 解决方法

`start`

- 采用spawn的方式，并将agent的实例化移动到setup_server中——这样每个进程有自己独占的上下文

```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def setup_server(server_launcher:RpcAgentServerLauncher) -> None:
    """Setup rpc server for participant agent"""
    agentscope.init(
        project="llmgraph",
        name="server",
        model_configs="LLMGraph/llms/default_model_configs.json",
        use_monitor=False,
        save_code=False,
        save_api_invoke=False,
    )
    server_launcher.launch(in_subprocess=False)
    server_launcher.wait_until_terminate()
```

# 二、linux多进程的kill

从`fork`切换为`spawn`方式，并将agent实例化移动到子进程中后，无法使用`pkill -f "python start_launchers.py" 2>/dev/null`来清除服务端的进程

## 基本排查

- 切换到 `spawn` 启动模式后，真正跑在子进程里的不是 `python start_launchers.py`，而是一串由 Python 在内存里执行的命令：

	```python
	python -c from multiprocessing.spawn import spawn_main; spawn_main(…)
	```
	
- `spawn` 模式启动子进程，每个子进程都会重新执行解释器启动流程（而不是简单地复制父进程内存），这样它们的命令行就指向了 Python 内置的 `-c spawn_main`，而不再指向你的脚本名。



## 方法

### 踩雷：

- 干掉PID值范围内的进程：可能不连号，有误伤

```bash
#!/bin/bash
ps -u $USER -o pid= \
  | awk '$1 >= 1105897' \
  | xargs -r kill
```

- 杀掉多进程：误伤VLLM的多进程

```bash
#!/bin/bash
ps -u $USER -o pid,cmd \
  | awk '/multiprocessing\.spawn/ && !/vllm\.entrypoints/ {print $1}' \
  | xargs -r kill
```

### 可用方法1：进程组

- `setsid python … &`：把 `start_launchers.py` 放入一个新 session，并且它自己为 leader
- `kill -TERM -- -$launcher_sid`：向该 PGID（前面加负号）广播 `SIGTERM`，会终结所有在这个 session/组里的进程

```bash
#!/bin/bash
cleanup() {
    echo "*** 执行清理操作 ***"
    # 如果 launcher_sid 有值，就杀掉其整个进程组（负号表示进程组）
    if [ -n "$launcher_sid" ]; then
        kill -TERM -- -"$launcher_sid" 2>/dev/null
    fi
}
trap cleanup EXIT

setsid python start_launchers.py > server.log 2>&1 &
launcher_sid=$!

python main.py
client_exit=$?
exit $client_exit
```

### 可用方法2：pkill标记

- 需要在 Python 里把 `--tag` 传给每个子进程

```bash
#!/bin/bash
export LAUNCHER_TAG="GRAG_$(date +%s)"
python start_launchers.py --tag $LAUNCHER_TAG

pkill -f "$LAUNCHER_TAG"
```

