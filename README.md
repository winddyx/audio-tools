# OmniVoice 配音工具

基于 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 的文本转语音工具，支持 **600+ 语言**：

- **语音克隆**：传参考音频（+ 可选参考文本），零样本克隆音色
- **声音设计**：用指令（如 `female, low pitch, british accent`）合成指定音色
- **自动音色**：不传参考音频与指令，模型自动选择音色

提供 CLI（`cli.py`）与 Web 界面（`web.py`），两者共用同一套核心（`src/` 包：设备检测 / 模型下载与加载 / ASR 转写 / 生成参数），推理后端与全部可调设置统一在项目根 `settings.py`：

| 后端 | 说明 | 模型 |
|---|---|---|
| `gguf`（默认） | C++/GGML 推理（[omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp)），输出 24 kHz | [Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF) **BF16** |
| `transformers` | 原 transformers 实现（进程内推理） | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) |

## 安装

```bash
uv sync        # 安装依赖（含 omnivoice 包，transformers 后端需要）
```

> **GGUF 后端（默认）首次运行**：自动完成两件事（均只需一次，之后离线复用）——
> 1. `git clone` + 编译 `omnivoice.cpp` 到项目内 `vendor/omnivoice.cpp/`（gitignore，约 10-20 分钟）；
> 2. 经 HuggingFace 下载 `omnivoice-base-BF16.gguf`（1.23 GB）+ `omnivoice-tokenizer-BF16.gguf`（373 MB）到 HF 默认缓存（有进度条；默认 BF16，可用 `OMNIVOICE_GGUF_BASE` / `OMNIVOICE_GGUF_CODEC` 切回 Q8_0 等变体）。
>
> 已编译的机器可用 `OMNIVOICE_CPP_BIN` 直接指定二进制、`OMNIVOICE_CPP_SRC` 指向已有源码，跳过自动 clone/编译。

> **设备加速**:`uv sync` 自动按平台选择 PyTorch 构建——
> - **Windows / Linux**:torch 固定为 PyTorch 官方 `xpu` 构建（2.11.0+xpu），
>   有 Intel Arc 显卡时自动用 GPU 加速（`DEVICE=xpu` 或自动检测）；无 Intel GPU
>   的机器删除 `pyproject.toml` 中 `[tool.uv]` 与 `[[tool.uv.index]]` 两段、
>   依赖列表里的 `torch==2.11.0`/`torchaudio==2.11.0` 四行，即回退 PyPI 的
>   CUDA/CPU 构建。
> - **macOS (Apple Silicon)**:自动从 PyPI 安装带 `macosx` wheel 的 torch，
>   用 MPS 加速（`DEVICE=mps` 或自动检测）。MPS 设备下自动解除 PyTorch
>   MPS 分配器内存上限（`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`，避免系统
>   其它进程占用高时连几十 MiB 都申请不到而报 `MPS backend out of memory`）；
>   若加载/生成阶段仍触发 MPS OOM，自动改用 CPU 重试并提示（可用
>   `DEVICE=cpu` 或显式设置该环境变量关闭自动解除）。
>
> GGUF 后端的设备映射（`GGML_BACKEND`）：`mps → MTL0`（ggml 的 Metal 设备名）、`cuda → CUDA0`、`cpu → CPU`；
> `xpu`/未指定交给运行时自动选择。设备推理失败自动回退 CPU 重试一次。

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
uv run python web.py                 # 启动后访问 http://localhost:38001
uv run python web.py --port 8000     # 自定义端口
uv run python web.py --share         # 创建公开链接
```

- 监听 `0.0.0.0:38001`（`--ip`/`--port` 可改）
- **自动打开浏览器**：由 `settings.py` 的 `WEB_AUTO_OPEN_BROWSER` 控制（默认 `False`，改为 `True` 或设 `OMNIVOICE_WEB_OPEN_BROWSER=1` 则启动后自动用默认浏览器打开），无需命令行参数
- 语音克隆页支持**抽卡**：设置次数 N，一次生成 N 个结果供挑选
- 生成参数（推理步数、引导强度、语速、时长等）与参考文本/附加指令在"生成参数"折叠面板中（GGUF 后端仅支持 `steps` / `denoise` / `chunk-duration` / `chunk-threshold` / `duration` 子集，其余忽略）

## 切换推理后端与设置

所有可调设置统一在项目根 `settings.py`（改文件即生效；同名环境变量可运行时覆盖）：

```python
# settings.py
BACKEND = "gguf"            # "gguf"（默认，C++/GGML，BF16）| "transformers"
GGUF_BASE = "omnivoice-base-BF16.gguf"    # 可换 Q8_0 / F32 / Q4_K_M（与 CODEC 配套）
CPP_BIN = ""               # 指定已编译的 omnivoice-tts（留空 = 自动编译）
WEB_PORT = 38001           # Web 端口
WEB_AUTO_OPEN_BROWSER = False
```

临时切换（不改文件）：`OMNIVOICE_BACKEND=transformers uv run python cli.py ...`
共享核心与另一后端代码均保留（`src/backends/`），随时可切回。

## 模型管理

- 模型统一由 HuggingFace 管理，落在默认缓存（`~/.cache/huggingface`，或 `HF_HOME` / `HF_HUB_CACHE`）
- **本地优先**：模型已完整缓存时直接复用，跳过一切联网；`HF_LOCAL_FIRST=0` 可关闭并强制联网校验更新
- 直连 huggingface.co 失败时自动改用 hf-mirror.com 镜像重试（`HF_NO_MIRROR_FALLBACK=1` 关闭；覆盖 GGUF 权重、transformers 权重与 ASR 模型，均遵循 `HF_ENDPOINT`）

## 常用环境变量

| 变量 | 作用 |
|---|---|
| `OMNIVOICE_BACKEND` | 推理后端：`gguf`（默认）/ `transformers`（对应 settings.BACKEND） |
| `LANGUAGE` / `--language` | 合成语言（如 `yue`/`en`） |
| `DRAW_COUNT` | CLI 抽卡次数（默认 2） |
| `DEVICE` / `--device` | `cuda` / `xpu` / `mps` / `cpu`（默认自动检测） |
| `THREADS` | CPU 线程数（默认 `os.cpu_count()`，用满所有逻辑核心；可设小值如 `4` 留出核心给其他任务） |
| `DTYPE` | 仅 transformers 后端：覆盖默认精度（CUDA/XPU bfloat16，MPS/CPU fp32）；GGUF 量化随文件而定（默认 BF16） |
| `MODEL_PATH` / `OMNIVOICE_MODEL_ID` | transformers 后端：本地模型目录 / 模型 ID |
| `OMNIVOICE_CPP_BIN` | GGUF 后端：指定已编译的 `omnivoice-tts` 二进制（跳过自动编译） |
| `OMNIVOICE_CPP_SRC` | GGUF 后端：指定 omnivoice.cpp 源码目录（默认项目内 `vendor/omnivoice.cpp`） |
| `OMNIVOICE_CPP_BUILD_ARGS` | GGUF 后端：追加 cmake 参数（如 `-DGGML_CUDA=ON`） |
| `OMNIVOICE_GGUF_REPO` / `OMNIVOICE_GGUF_BASE` / `OMNIVOICE_GGUF_CODEC` | GGUF 后端：权重仓库 ID / base 文件 / codec 文件（默认 BF16 双文件，可切 Q8_0 等） |
| `OMNIVOICE_GGUF_DEBUG` | `1` 透传 omnivoice-tts 全部 stderr（ggml 内核编译 / MaskGIT 步进，默认静默、失败时打印尾部） |
| `ASR_MODEL` | SenseVoice 模型 ID/本地目录（默认 `FunAudioLLM/SenseVoiceSmall`） |
| `ASR_HUB` | SenseVoice 下载源：`hf`（默认，HuggingFace）/ `ms`（ModelScope） |
| `ASR_VAD` | VAD 切分模型（默认 `fsmn-vad`；设 `0` 关闭，仅适合短音频） |
| `OMNI_PORT` / `OMNIVOICE_WEB_IP` | web 端口（默认 38001）/ 监听地址（默认 0.0.0.0） |
| `OMNIVOICE_WEB_OPEN_BROWSER` | `1` 启动后自动打开浏览器（对应 settings.WEB_AUTO_OPEN_BROWSER） |
| `HF_ENDPOINT` / `HF_NO_MIRROR_FALLBACK` | HuggingFace 镜像 / 关闭镜像兜底 |
| `HF_LOCAL_FIRST` | `0` 关闭本地优先（强制联网校验） |
