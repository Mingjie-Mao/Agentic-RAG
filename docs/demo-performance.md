# 公开演示性能与运行

## 运行方式

公开演示使用 `make demo`，或 `.venv/bin/python -m scripts.run_public_demo`。复用固定 Ollama 0.11.10 与已安装模型，独立请求队列为 `127.0.0.1:11438`；实验默认保持 `11436`。现有 API / worker 监督脚本管理自己启动的进程，Ctrl-C 只停止这些子进程，不杀其他实验。

启动先检查生成与 embedding 模型 digest，再用官方空 chat 和 embedding 请求预热，完成后才启动 API / worker。预热不建立问答、不占访客提问额度。预热记录写入新的 `.runtime/public-model-warmup-<UTC时间>.json`。

独立端口隔离的是排队，不是物理 GPU。当前仍在同一台 M4 / 32GB 上；后台构建、测试和模型实验仍会影响吞吐。禁止将本机的缓存延迟表述为 GPU 推理性能。

## 已测量的效果

本轮实际支持测试账号连续两次 HTTP 请求：首次真实生成 **57.430s**，第二次精确缓存 **0.135s**，答案与引用一致，缓存请求模型调用数为 0。首次请求期间还有前端构建与全量 pytest，不能把它与此前 39.2s 当作受控性能对比。模型加载为 56.7ms，输入处理 40.4s、解码 15.8s，首次推理仍然慢。

记录：`.runtime/public-demo-captures/cache-live-20261009T145516Z.json`。原支持测试账号请求计数由 3 增至 5，两次都计请求额度；没有重置额度，没有消耗共享 visitor 账号的剩余请求。

## 无损压缩 Dev 门槛

完整 Dev47 的真实授权检索 / 准备后输入已采集，34 题到达首次生成，13 题未产生生成输入。本次没有执行生成调用，也没有给质量或延迟打分（均为 null）。所有 JSON 值原样保留；cl100k_base 估计平均输入缩减 **10.983%**，低于事先登记的 15% 输入收益门槛。

因此停止在输入筛选阶段，公开启动明确保持 `compact_prompt_json=false`。不使用未经质量验证的裁剪；不修改 Core 或历史文件。报告为 `artifacts/demo-prompt-dev-720a2e02-bd3d-492e-87ca-9f2f5255ac5f.json`；原始第三方文本只在忽略的 `.runtime/demo-performance/` 下。

## 精确缓存的业务边界

缓存复用已有 Answer 存储，不引入第二套持久化框架。默认关闭，公开启动只为只读试用身份打开；只有无上文、已返回 answered 且有核验引用的精确相同问题可以复用。

键绑定当前用户 / 租户 / 角色 / 组、完整可见文档集合、版本内容哈希和处理状态、文档修订、当前代码 / 配置以及进程 epoch。新增可见资料（包括可能产生冲突的资料）、撤权、删除、版本切换、模型或提示配置变化都失效；启动后重新建立缓存，默认最多 1 小时。命中仍逐个检查原回答的全部证据依赖，再核对语料快照，重新写本次问答记录。缓存命中不会延长原条目的寿命。

失败、拒答、追问、Agent 任务和带 benchmark_run_token 的运行不进入缓存。当前日限额是请求数，所以缓存命中仍占一次；不绕过限额。页面标明缓存、原生成日期、当前再鉴权和本次零模型调用。近期典型模型耗时排除缓存，不将两者混算。

正常追问保留上文；“新问题”清空本页上下文与回答显示，已保存历史仍可查看。独立问题才可命中上述缓存，不隐式丢弃用户的追问语义。

## 独立硬件后端：配置已准备，未开通

`deploy/compose.demo-inference.yaml` 为独立 NVIDIA 主机准备固定版本服务，端口只绑定 localhost。需提供实际 GPU 主机，不会在当前 Mac 上伪造独立算力。

在用户自己的 GPU 主机运行 Docker Compose，安装两份固定模型，随后经 SSH 把该机 localhost:11438 转发到本机一个空闲端口，例如 11440。运行：

```sh
.venv/bin/python -m scripts.run_public_demo --inference-url http://127.0.0.1:11440 --warm-only
.venv/bin/python -m scripts.run_public_demo --inference-url http://127.0.0.1:11440
```

独立服务仍要通过相同模型 / runtime digest 验证和 Dev 实测；不能未经测量称为更快。外部直接连接必须为 HTTPS；拒绝明文外部服务、URL 内嵌凭据和实验 11436 队列。目前没有提供服务器地址或云服务预算，本轮没有创建服务器、读取云密钥或调用计费模型。

## 发布与复查

前端已发布到 `https://9f8ceb38.agentic-rag.pages.dev`，正式入口为 `https://agentic-rag.pages.dev/live`。构建时间为 2026-10-10 悉尼 02:07:25。9 项公开资源 HTTP 200 且 SHA-256 与发布构建一致；记录保存在 `.runtime/public-demo-captures/demo-latency-release-20261009T151936Z.json`。8 项 Node 测试、前端构建与相关检查通过；当时完整 pytest 快照为 1286 passed / 36 skipped。

2026-10-10 悉尼 06:19 复查发现 API 未运行，现已重新预热并恢复。现有 `scripts/run.py` 监督进程在后台运行，日志为 `.runtime/public-demo-supervisor.log`，所属进程记录仍由 `.runtime/processes.json` 管理；原实验进程与已有 Tunnel 未停止或改动。公开状态接口返回 200，浏览器重新进入只读工作空间，未发送问答、未重置额度，访客仍剩 1/10。恢复记录为 `.runtime/public-demo-captures/demo-latency-restored-20261009T191909Z.json`；线上截图为 `/private/tmp/demo-latency-live-20261010.png`。复查时重新执行缓存与独立运行相关测试，24 passed；Compose 配置校验和 diff 检查通过。

重启后缓存按设计重新建立，此前的 0.135s 是历史命中实测，不能保证重启后的首次问题也命中。后台运行避免依赖当前终端会话，但电脑休眠、断网或服务退出仍会影响 Live；不等同于独立托管。第 5 步迁移、速度和质量对照仍待实际后端信息。
