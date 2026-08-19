# OmniVoice 配音工具

基于 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 的文本转语音工具，支持 **600+ 语言**：

- **语音克隆**：传参考音频（+ 可选参考文本），零样本克隆音色
- **声音设计**：用指令（如 `female, low pitch, british accent`）合成指定音色
- **自动音色**：不传参考音频与指令，模型自动选择音色

提供 CLI（`cli.py`）与 Web 界面（`web.py`），两者共用同一套模型逻辑（`omni.py` 核心库：设备检测 / 模型下载与加载 / ASR 转写 / 生成参数）。

## 安装

```bash
uv sync        # 安装依赖（含 omnivoice 包）
```

> **设备加速**:`uv sync` 自动按平台选择 PyTorch 构建——
> - **Windows / Linux**:torch 固定为 PyTorch 官方 `xpu` 构建（2.11.0+xpu），
>   有 Intel Arc 显卡时自动用 GPU 加速（`DEVICE=xpu` 或自动检测）；无 Intel GPU
>   的机器删除 `pyproject.toml` 中 `[tool.uv]` 与 `[[tool.uv.index]]` 两段、
>   依赖列表里的 `torch==2.11.0`/`torchaudio==2.11.0` 四行，即回退 PyPI 的
>   CUDA/CPU 构建。
> - **macOS (Apple Silicon)**:自动从 PyPI 安装带 `macosx` wheel 的 torch，
>   用 MPS 加速（`DEVICE=mps` 或自动检测）。

## CLI 用法

```bash
# 语音克隆（ref_text 省略时自动用 FunASR/SenseVoiceSmall 转写参考音频）
uv run python cli.py <ref_audio.wav> <text.txt> -l yue

# 声音设计
uv run python cli.py --text <text.txt> --instruct "female, low pitch, british accent"

# 自动音色
uv run python cli.py --text <text.txt>

# ASR 转写参考音频（FunASR/SenseVoiceSmall，用于校对/数据集）
uv run python cli.py --transcribe <ref_audio.wav>
```

生成结果输出到文本文件所在目录，文件名 `<文本名>.<unix时间戳>.wav`。

## Web 界面

```bash
uv run python web.py                 # 启动后自动打开浏览器 → http://localhost:38001
uv run python web.py --port 8000     # 自定义端口
uv run python web.py --share         # 创建公开链接
uv run python web.py --no-browser    # 不自动打开浏览器
```

- 监听 `0.0.0.0:38001`（`--ip`/`--port` 可改）
- 语音克隆页支持**抽卡**：设置次数 N，一次生成 N 个结果供挑选
- 生成参数（推理步数、引导强度、语速、时长等）与参考文本/附加指令在"生成参数"折叠面板中
- 生成文件在项目内 `.tmp/`，退出时自动清理，每次启动清扫残留

## 模型管理

- 模型统一由 HuggingFace 管理，落在默认缓存（`~/.cache/huggingface`，或 `HF_HOME` / `HF_HUB_CACHE`）
- **本地优先**：模型已完整缓存时直接复用，跳过一切联网；`HF_LOCAL_FIRST=0` 可关闭并强制联网校验更新
- 直连 huggingface.co 失败时自动改用 hf-mirror.com 镜像重试（`HF_NO_MIRROR_FALLBACK=1` 关闭；此兜底仅对 OmniVoice 主模型生效，ASR 模型由 funasr 经 huggingface_hub 下载，遵循 `HF_ENDPOINT`）

## 常用环境变量

| 变量 | 作用 |
|---|---|
| `LANGUAGE` / `--language` | 合成语言（如 `yue`/`en`） |
| `DRAW_COUNT` | CLI 抽卡次数（默认 2） |
| `DEVICE` / `--device` | `cuda` / `xpu` / `mps` / `cpu`（默认自动检测） |
| `THREADS` | CPU 线程数（默认 `os.cpu_count()`，用满所有逻辑核心；可设小值如 `4` 留出核心给其他任务） |
| `DTYPE` | 覆盖默认精度（CUDA fp16，XPU bfloat16，MPS/CPU fp32） |
| `MODEL_PATH` / `OMNIVOICE_MODEL_ID` | 本地模型目录 / 模型 ID |
| `ASR_MODEL` | SenseVoice 模型 ID/本地目录（默认 `FunAudioLLM/SenseVoiceSmall`） |
| `ASR_HUB` | SenseVoice 下载源：`hf`（默认，HuggingFace）/ `ms`（ModelScope） |
| `ASR_VAD` | VAD 切分模型（默认 `fsmn-vad`；设 `0` 关闭，仅适合短音频） |
| `OMNI_PORT` | web 端口（默认 38001） |
| `HF_ENDPOINT` / `HF_NO_MIRROR_FALLBACK` | HuggingFace 镜像 / 关闭镜像兜底 |
| `HF_LOCAL_FIRST` | `0` 关闭本地优先（强制联网校验） |
