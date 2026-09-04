# audio-tools

语音克隆工具：参考音频 + 文本 → 克隆音色朗读（多 TTS 模型可切换）。
推理由 [audio.cpp](https://github.com/0xShug0/audio.cpp)（ggml C++ 引擎，
`audiocpp_cli`）完成，Python 侧只做编排；ASR 用 SenseVoice-Small（audiocpp
`sense_asr` 族）自动转写参考音频。

## 目录结构

```
├── vc.py                  # CLI 入口（语音克隆 / --transcribe）
├── web.py                 # Gradio 入口（单页语音克隆）
└── src/
    ├── config.py          # 全局设置（唯一设置源，顶部变量 + 环境覆盖）
    ├── audiocpp.py        # 推理引擎运行器（audio.cpp，模型无关）
    ├── omnivoice.py       # OmniVoice 模型核心（TTS 语音克隆）
    ├── indextts2.py       # IndexTTS-2.5 模型核心（TTS 语音克隆）
    ├── sensevoice.py      # SenseVoice-Small ASR 核心（参考音频转写）
    ├── hf.py              # HuggingFace 下载（本地优先 + hf-mirror 兜底）
    └── pipeline.py        # 统一编排 synthesize()/draw()
```

## 快速开始

```bash
uv sync   # 安装依赖（首次运行自动 clone + 构建 audiocpp_cli、下载 GGUF 权重）

# CLI：语音克隆（ref_text 省略时自动用 SenseVoice 转写参考音频）
uv run python vc.py <ref_audio.wav> <text.txt>

# CLI：ASR 转写参考音频（校对用）
uv run python vc.py --transcribe <ref_audio.wav>

# Web：http://localhost:38001
uv run python web.py
```

## 设置（src/config.py 顶部变量，同名环境变量可覆盖）

| 变量 | 默认 | 说明 |
|---|---|---|
| `TTS_MODEL` | `omnivoice` | TTS 模型：`omnivoice` / `indextts2` |
| `LANGUAGE` | 空 | 合成语言（如 `zh` / `en` / `yue`）；空 = 自动 |
| `DRAW_COUNT` | `2` | 抽卡次数 |
| `OUTPUT_DIR` | 文本所在目录 | CLI 输出目录 |
| `DEVICE` | 自动 | `cuda` / `mps` / `cpu`（audiocpp 后端映射） |
| `OMNI_INFERENCE_STEPS` | `32` | OmniVoice 去噪步数（0 = 引擎默认） |
| `OMNI_GUIDANCE_SCALE` | `2.0` | OmniVoice CFG 引导尺度（空 = 引擎默认） |
| `INDEXTTS_TOP_K` / `INDEXTTS_TOP_P` / `INDEXTTS_TEMPERATURE` | `30` / `0.8` / `0.8` | IndexTTS-2.5 gpt 层采样参数（官方基准） |
| `GEN_SEED` | `-1` | 固定随机种子（`-1` = 随机；设同值可复现结果） |
| `AUDIOCPP_BIN` / `AUDIOCPP_SRC` | 空 | 已编译二进制 / 已有源码（留空自动构建到 vendor/） |
| `ASR_MODEL` | 空 | 本地 SenseVoice GGUF 路径（默认经 HF 下载） |
| `WEB_IP` / `WEB_PORT` | `0.0.0.0` / `38001` | Web 监听 |
| `HF_ENDPOINT` | 空 | 直连失败自动切 hf-mirror（`HF_NO_MIRROR_FALLBACK=1` 关闭） |

## 模型

- TTS 权重默认走本地 `models/`（gitignore），缺失时从 HuggingFace 下载到 HF
  默认缓存（本地优先 + 镜像兜底）：
  - OmniVoice: `audio-cpp/audio.cpp-gguf` → `OmniVoice-GGUF/omnivoice-bf16.gguf`
  - IndexTTS-2.5: `audio-cpp/audio.cpp-gguf` → `IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf`
- ASR: `FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp` →
  `sensevoice-small-q8-audiocpp-v1.gguf`
- 引擎：audio.cpp 首次运行自动 clone + cmake 构建到 `vendor/audiocpp/`
  （custom 模型集：omnivoice / index_tts2 / sense_asr）

## 阶段化生成（CLI 与 Web 同构）

环境准备 → 模型准备 → 输入文件检查 → ASR → VOICECLONE → 输出文件规范，
终端以 `[i/6]` 显示，长合成每 10s 报一次进度。
