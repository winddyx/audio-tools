# tts-omnivoice — OmniVoice 配音工具

OmniVoice 文本转语音工具，支持语音克隆（参考音频）和音色设计（属性描述）两种模式。基于 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 模型。

## 项目

- **语言**: Python ≥ 3.10，使用 `uv` 管理依赖
- **依赖管理**: `pyproject.toml` + `uv.lock`（`uv sync` 安装）
- **入口文件**:
  - `omni.py` — CLI 长文本配音入口（`uv run python omni.py <ref_audio> <text_file>`）
  - `web.py` — Gradio Web 界面（`uv run python web.py`）

## 命令

| 用途 | 命令 |
|------|------|
| 安装依赖 | `uv sync` |
| CLI 长文本配音 | `uv run python omni.py <ref_audio.wav> <text.txt>` |
| 启动 Web 界面 | `uv run python web.py` |
| 指定端口 | `uv run python web.py --port 8000` |
| 创建公开链接 | `uv run python web.py --share` |
| 指定模型路径 | `uv run python web.py --model-path /path/to/OmniVoice` |

CLI 参数和环境变量覆盖优先级：CLI 参数 > 环境变量 > Config 默认值。关键环境变量：`LANGUAGE`、`DRAW_COUNT`、`REF_TEXT`、`DEVICE`、`MODEL_PATH`、`OMNI_MODEL_ID`。

## 架构

- **`omni.py` (CLI)** — `Config` dataclass 集中管理默认配置；`main()` 负责参数解析 → 配置合并 → 输入校验 → 音频转换 → 自动转录（SenseVoiceSmall）→ 多轮生成。全局缓存 OmniVoice 和 SenseVoiceSmall 模型（单例模式）。
- **`web.py` (Gradio)** — `build_demo()` 构建双 Tab 界面（语音克隆 + 音色设计），共享 `_gen_core()` 生成核心。引用 `omni.py` 的 `Config`、`resolve_path`、`convert_audio` 作为唯一数据源。模型加载全程 float32。
- **数据传输** — 参考音频经 `convert_audio()` 转为 24kHz 单声道 WAV，OmniVoice 生成后 `soundfile` 写出；临时文件自动清理。

## 约定

- **import**: 标准库 → 第三方 → 本地，空行分隔；`from __future__ import annotations` 在第一行（有 `#!` 则在其后）。
- **类型注解**: 全程使用 type hints（`Optional`, `|`, dataclass）。
- **错误处理**: CLI 入口捕获 Exception 并 `logger.exception` + `sys.exit(1)`；工具函数抛出 `RuntimeError`；Gradio 内部捕获异常返回 UI 错误信息。
- **日志**: `logging.getLogger(__name__)`，CLI 格式 `%(message)s`，Web 格式 `%(asctime)s %(name)s %(levelname)s: %(message)s`。
- **模型缓存**: 全局模块级变量（`_OMNIVOICE_MODEL`, `_SENSEVOICE_MODEL`），from_pretrained 只加载一次。Web 用 `load_model()` 单例，CLI 用 `_load_omnivoice()`/`_load_sensevoice()`。
- **设备**: `get_best_device()` 自动检测 CUDA > MPS > CPU。CLI 默认 MPS，Web 默认自动检测。
- **工具依赖**: 需要 `ffmpeg` 在 PATH 中用于音频格式转换。
- **git**: 当前 4 个 commit，无分支策略记录。fork 项目不向 upstream 提 PR。

## 备注
