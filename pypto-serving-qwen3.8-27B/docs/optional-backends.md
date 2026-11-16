# 可选的推理后端

OneServing 通过 Python 包元数据 `oneserving.backends` 发现后端。没有设置 `ONESERVING_BACKEND` 时使用入口自己的引擎。显式指定名字时要求恰好匹配一个 provider。

```toml
[project.entry-points."oneserving.backends"]
redknot = "redknot_oneserving.backend:Engine"
```

provider 的构造参数为 `(model_root, device_id, max_active=..., prefix_cache=...)`，实现原有 engine 的启动、退出、请求和快照接口。后端可以用 `runner_class`、`scheduler_class` 安装自己的实现。completion 和 chat 收到的可选 `extensions` 在分词后、入队前由 `request_context(prompt_ids, max_tokens, extensions)` 校验并绑定到请求。请求的区间和快照生命周期由后端管理。无扩展请求走原有提交流程。

`serving.long_context.api:app` 提供 Qwen3.8-27B 的单槽 32K 原生模型和运行时。它公开图注册、额外共享输入注册和 mask 就绪钩子，输出缓冲按已注册的最大图分配。原生和外部插件均可用同一 HTTP 入口。宿主代码拥有模型权重、基础图、统一调度器和设备生命周期。RedKnot 包负责标定、文档匹配、页分配、稀疏图及策略。

```bash
# 进入 pypto-serving-qwen3.8-27B，使用 PyPTO 运行环境。
export MODEL_ROOT=/path/to/full/Qwen3.8-27B MAX_ACTIVE=1 PREFIX_CACHE=0
./run.sh --backend long-context --model-root "$MODEL_ROOT" --devices 0 --port 18082
# 安装外部 wheel 后，在启动命令前加 ONESERVING_BACKEND=redknot。
```

`serving.optimized.api:app` 的默认配置仍来自原 PR。选用长上下文是一次显式的模型与图 ABI 选择。移植到已有长上下文实现的宿主时，可以单独查看 provider 接口和 runtime 钩子的提交。
