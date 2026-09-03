# OmniVoice 配音工具（模板化 v2）

基于 GGUF 权重的本地文本转语音工具：语音克隆（零样本）/ 声音设计 / 自动音色，
附 SenseVoice GGUF ASR 参考转写。Python 负责编排，推理在 native 引擎
（omnivoice-tts C++/GGML、llama-funasr-sensevoice）一次性子进程中完成，
无 torch、无后台常驻进程。

本项目同时是「生成模型参考模板」：`ov/` 核心层与具体模型无关，加新生成模型
只需在 `ov/models/` 新增一个子包（见下文「加新模型」）。

## 运行方式

```bash
uv sync        # 安装依赖；首次运行自动 clone+编译引擎、下载 GGUF 权重到 HF 缓存

# CLI：合成（<text.txt> 必选；[<ref.wav>] 提供即语音克隆）
uv run python cli.py <text.txt> [<ref.wav>]

# CLI：ASR 转写参考音频（校对用）
uv run python cli.py --asr <ref.wav>

# Web：http://localhost:38001
uv run python web.py

# 测试（stub 引擎，不加载真实模型 / 不联网）
uv run pytest
```

所有设置（语言、设备、输出目录、生成参数、Web 端口等）都是
`ov/settings.py` 顶部变量，同名环境变量可覆盖；CLI 不设额外启动参数，
只接受「引用哪个文件」。输出 wav 命名 `<文件名>.<unix时间戳>.wav`
（同秒冲突自动递增），默认落盘在文本文件所在目录（CLI）或项目 `out/`（Web）。

## 架构

```
cli.py / web.py（薄入口）
   -> tools/cli_main.py / tools/web_main.py（界面适配，只依赖 ov.api）
        -> ov.api（统一门面：模型解析 / ASR 预转写 / 默认参数）
             -> ov.pipeline（编排：命名 / 落盘 / 长文本策略）
                  -> ov.models/<name>/engine.py（Engine 适配器）
                       -> native 引擎（GGUF 一次性子进程）
```

- `ov/`         模型无关核心：settings（顶部变量）、types（数据契约）、
                model（ModelSpec + Engine 注册表）、pipeline、assets（HF 下载）、
                audio/text（工具）、logs（纯文本日志）。
- `ov/models/`  具体模型包：`omnivoice`（TTS）、`sensevoice`（ASR）。
- `runtime/`    引擎产物目录（gitignore）：C++ 源码+编译、funasr 二进制。
- `patches/`    需要维护的 C++ 定制（重 clone runtime 后 `git apply`）。
- `tests/`      stub 引擎单测（注册表 / 编排 / 长文本 / 音频），不加载模型。

数据契约集中在 `ov/types.py`：SynthesizeRequest / TranscribeRequest /
GenParams / EngineResult / Segment / SynthesisOutcome。引擎能力由
ModelSpec.capabilities 声明（clone/design/auto/native_longform/transcribe…），
长文本：有 native_longform 走引擎原生分块，否则 `ov/pipeline` 用
`ov/text` 兜底逐段合成拼接（未来模型的模板路径）。

模型管理规则：GGUF 权重一律经 HuggingFace（本地优先 + hf-mirror 兜底，
`HF_LOCAL_FIRST=0` / `HF_NO_MIRROR_FALLBACK=1` 可调）；不硬编码缓存路径。

## 设备

`settings.DEVICE` 留空 = 引擎自选（darwin 优先 Metal/MTL0），backend
初始化失败自动回退 CPU 重试一次；可显式设 cuda / mps / cpu。不再依赖
torch 做设备探测。

## 加新模型（模板用法）

1. `ov/models/<name>/` 下写 `spec.py`：ModelSpec(id/kind/capabilities/
   supported_params/fallback_chunk_chars) + register()；声明资产与默认参数
   （放 `ov/settings.py` 顶部变量）。
2. 写 `engine.py`：实现 Engine 协议（provision / synthesize / transcribe），
   内部可用 subprocess 调 native 引擎，也可接任意 Python 推理库。
3. 在 `ov/models/__init__.py` import 一行完成注册。
4. 无特殊需求的模型自动获得：CLI 模式、Web 参数面板、长文本兜底、
   单测框架 —— 都不需要改核心层。

## 常见问题

- 引擎二进制缺失：首次调用自动 clone+cmake 编译到 `runtime/`（约 10-20 分钟）；
  已备好的机器可用 `OMNIVOICE_CPP_BIN` / `FUNASR_LLAMACPP_BIN` 直接指定。
- 断句硬切分等 C++ 定制见 `patches/omnivoice-cpp.patch`；重 clone 后 git apply。
- macOS 登录自启见 `omni-web.plist`（launchctl bootstrap 部署）。
