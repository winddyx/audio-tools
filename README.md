# OmniVoice 配音工具

基于 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 的文本转语音工具，支持 **600+ 语言**：

- **语音克隆**：传参考音频（+ 可选参考文本），零样本克隆音色
- **声音设计**：用指令（如 `female, low pitch, british accent`）合成指定音色
- **自动音色**：不传参考音频与指令，模型自动选择音色

提供 CLI（`cli.py`）与 Web 界面（`web.py`），共用同一套核心（`src/` 包），全部模型与可调设置统一在项目根 `settings.py`：

| 模块 | 实现 |
|---|---|
| **TTS 推理**（唯一后端） | C++/GGML：[omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp) + [Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF) 权重（默认 **BF16**，24 kHz 输出） |
| **ASR 参考文本** | [FunAudioLLM/SenseVoiceSmall-GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF) **Q8_0**（FunASR llama.cpp runtime，CPU 上 ~20× 实时） |

## 安装

```bash
uv sync        # 安装依赖（Python 侧仅 soundfile/huggingface_hub/gradio/torch）
```

> **首次运行自动完成**（均只需一次，之后离线复用）——
> 1. `git clone` + 编译 `omnivoice.cpp` 到项目内 `vendor/omnivoice.cpp/`（gitignore，约 10-20 分钟）；
> 2. 经 HuggingFace 下载 GGUF 权重到 HF 默认缓存（有进度条）：
>    - TTS：`omnivoice-base-BF16.gguf`（1.23 GB）+ `omnivoice-tokenizer-BF16.gguf`（373 MB）；
>    - ASR：`sensevoice-small-q8.gguf`（~235 MB）+ `fsmn-vad.gguf`（1.7 MB）。
> 3. 下载 FunASR llama.cpp runtime 预编译二进制（`llama-funasr-sensevoice`，~3 MB）到 `vendor/funasr-llamacpp/`。
>
> 已备好的机器可用 `OMNIVOICE_CPP_BIN` / `FUNASR_LLAMACPP_BIN` 直接指定二进制、`OMNIVOICE_CPP_SRC` 指向已有源码，跳过自动获取。

> **设备加速**：
> - **GGUF TTS**（C++/ggml）：`GGML_BACKEND` 映射 `mps → MTL0`（ggml 的 Metal 设备名）、`cuda → CUDA0`、`cpu → CPU`；`xpu`/未指定交给运行时自动选择。设备初始化失败自动回退 CPU 重试一次。
> - **ASR**（llama.cpp）：CPU 推理（q8 小模型，~20× 实时）。
> - **macOS (Apple Silicon)**：M4 系列硬件支持 bfloat16，BF16 GGUF 在 Metal 上接近实时（RTF < 1）。

## CLI 用法

```bash
# 语音克隆（ref_text 省略时自动用 SenseVoiceSmall-GGUF 转写参考音频）
uv run python cli.py <ref_audio.wav> <text.txt> -l yue

# 声音设计
uv run python cli.py --text <text.txt> --instruct "female, low pitch, british accent"

# 自动音色
uv run python cli.py --text <text.txt>

# ASR 转写参考音频（SenseVoiceSmall-GGUF，用于校对/数据集）
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
- 生成参数（推理步数、时长等）与参考文本/附加指令在"生成参数"折叠面板中（GGUF 后端仅支持 `steps` / `denoise` / `chunk-duration` / `chunk-threshold` / `duration` 子集，其余忽略）
- ASR 识别的参考文本会同步打印到终端（与 cli.py 一致）

## 模型管理

- 模型统一由 HuggingFace 管理，落在默认缓存（`~/.cache/huggingface`，或 `HF_HOME` / `HF_HUB_CACHE`）
- **本地优先**：模型已完整缓存时直接复用，跳过一切联网；`HF_LOCAL_FIRST=0` 可关闭并强制联网校验更新
- 直连 huggingface.co 失败时自动改用 hf-mirror.com 镜像重试（`HF_NO_MIRROR_FALLBACK=1` 关闭；覆盖 TTS 与 ASR 权重，均遵循 `HF_ENDPOINT`）

## 常用环境变量

| 变量 | 作用 |
|---|---|
| `LANGUAGE` / `--language` | 合成语言（如 `yue`/`en`） |
| `DRAW_COUNT` | CLI 抽卡次数（默认 2） |
| `DEVICE` / `--device` | `cuda` / `xpu` / `mps` / `cpu`（默认自动检测） |
| `THREADS` | CPU 线程数（默认 `os.cpu_count()`，用满所有逻辑核心；可设小值如 `4` 留出核心给其他任务） |
| `OMNIVOICE_CPP_BIN` | 指定已编译的 `omnivoice-tts` 二进制（跳过自动编译） |
| `OMNIVOICE_CPP_SRC` | 指定 omnivoice.cpp 源码目录（默认项目内 `vendor/omnivoice.cpp`） |
| `OMNIVOICE_CPP_BUILD_ARGS` | 追加 cmake 参数（如 `-DGGML_CUDA=ON`） |
| `OMNIVOICE_GGUF_REPO` / `OMNIVOICE_GGUF_BASE` / `OMNIVOICE_GGUF_CODEC` | TTS GGUF 权重仓库 ID / base 文件 / codec 文件（默认 BF16 双文件，可切 Q8_0 等） |
| `OMNIVOICE_GGUF_DEBUG` | `1` 透传 omnivoice-tts 全部 stderr（ggml 内核编译 / MaskGIT 步进，默认静默、失败时打印尾部） |
| `ASR_GGUF_REPO` / `ASR_GGUF_BASE` | ASR 权重仓库 ID / GGUF 文件（默认 `FunAudioLLM/SenseVoiceSmall-GGUF` / `sensevoice-small-q8.gguf`，可换 f16） |
| `ASR_VAD_REPO` / `ASR_VAD_BASE` | VAD 权重仓库 ID / GGUF 文件（默认 `FunAudioLLM/fsmn-vad-GGUF` / `fsmn-vad.gguf`） |
| `FUNASR_LLAMACPP_BIN` | 指定已下载的 `llama-funasr-sensevoice` 二进制（留空自动下载到 `vendor/funasr-llamacpp/`） |
| `OMNI_PORT` / `OMNIVOICE_WEB_IP` | web 端口（默认 38001）/ 监听地址（默认 0.0.0.0） |
| `OMNIVOICE_WEB_OPEN_BROWSER` | `1` 启动后自动打开浏览器（对应 settings.WEB_AUTO_OPEN_BROWSER） |
| `HF_ENDPOINT` / `HF_NO_MIRROR_FALLBACK` | HuggingFace 镜像 / 关闭镜像兜底 |
| `HF_LOCAL_FIRST` | `0` 关闭本地优先（强制联网校验） |

## 目录结构

```
cli.py / web.py     业务入口（CLI / Gradio Web）
settings.py         所有可调设置（改文件或环境变量覆盖）
src/
├── config.py       Config 数据类
├── logs.py         HF 相关第三方库日志降噪
├── device.py       设备检测与容错（cuda > xpu > mps > cpu）
├── hf.py           HuggingFace 下载与缓存管理
├── params.py       生成参数环境变量映射
├── lang_map.py     Web 语言下拉（内置常用语言表）
├── asr.py          SenseVoiceSmall-GGUF 参考音频转写（llama.cpp runtime）
└── backends/gguf.py  GGUF 推理后端（omnivoice.cpp）
omni.py             兼容 shim（聚合导出共享核心 + GGUF 后端）
vendor/             omnivoice.cpp 源码/编译产物 + FunASR llama.cpp runtime（gitignore）
```
