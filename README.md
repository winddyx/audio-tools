# audio-tools

语音克隆工具：参考音频 + 文本 → 克隆音色朗读。支持多 TTS 模型切换
（OmniVoice / IndexTTS-2.5 / FireRedTTS-3），参考音频缺文本时用 SenseVoice
自动转写。推理由 [audio.cpp](https://github.com/0xShug0/audio.cpp)
（ggml C++ 引擎，`audiocpp_cli`）子进程完成，Python 只做编排，无 torch 依赖。

## 项目逻辑（架构与调用链）

```
vc.py (CLI) ─┐
             ├─→ src/pipeline.synthesize() ──→ ASR（缺 ref_text 时 SenseVoice）
web.py (Web) ┘          │                     → 按 TTS_MODEL 分发模型核心
                        │                     → 写 WAV（<out_name>.<秒>.wav）
                        └─→ 模型核心 generate() ──→ audiocpp_cli 子进程（C++ 推理）
                        （每次生成独立子进程：模型随进程加载/退出即卸载）
```

- **入口与编排**：`vc.py`（CLI）、`web.py`（Gradio 双 Tab）共用 `src/` 包，
  统一走 `pipeline.synthesize()`；Web 生成页的"上传即 ASR"直连
  `_transcribe_ref()`。
- **引擎/模型按需加载**：web 启动只启动 UI（不预热）。首次上传音频（ASR）或
  点击生成时才触发 `audiocpp._ensure_binary()`（定位二进制，缺失自动
  clone + cmake 构建 custom 集）与各模型核心 `_ensure_model()`（定位/下载
  GGUF）；模型在 `audiocpp_cli` 子进程内按次加载、进程退出即卸载，任务结束
  调 `pipeline.release()` 清进程内引擎缓存，长时间运行不留存资源。
- **模型放置（只在 HF 默认缓存）**：权重经 HF 下载后一律留在默认缓存
  （`~/.cache/huggingface/hub`，遵循 `HF_HOME` / `HF_HUB_CACHE`），不写入
  工程目录。audio.cpp 按真实文件扩展名识别权重：缓存 blobs/ 是哈希文件名
  （无扩展名）、snapshots/ 软链会被引擎 canonical 解析还原，均不能直接喂
  引擎；`hf._ensure_gguf_file()` 在缓存仓库目录内生成带 `.gguf` 的硬链接别名
  （同 inode，不占额外空间；跨文件系统退化为复制），把别名路径交给引擎。
  工程 `models/` 仅供用户手工放置（可选），自动下载绝不写入。
- **生成参数**：默认 = 官方基准（OmniVoice 32 步/CFG 2.0、IndexTTS-2.5
  top-k 30/top-p 0.8/temperature 0.8、FireRedTTS-3 4 步/CFG 2.0/停止阈值
  0.5），常量在 `src/config.py`（同名 env 覆盖，0/空 = 不传回引擎默认）；
  Web「配置」页可运行期覆盖，持久化仍以 config.py / env 为准。

## 目录结构

```
├── vc.py                  # CLI 入口（语音克隆 / --transcribe）
├── web.py                 # Gradio 入口（双 Tab：生成 / 配置）
└── src/
    ├── config.py          # 全局设置（唯一设置源：顶部变量 + env 覆盖）
    ├── audiocpp.py        # 推理引擎运行器（audio.cpp，模型无关，按需构建/释放）
    ├── omnivoice.py       # OmniVoice 模型核心（TTS 语音克隆）
    ├── indextts2.py       # IndexTTS-2.5 模型核心（TTS 语音克隆）
    ├── fireredtts3.py     # FireRedTTS-3 Base 模型核心（零样本语音克隆）
    ├── sensevoice.py      # SenseVoice-Small ASR 核心（参考音频转写）
    ├── hf.py              # HuggingFace 下载（本地优先 + hf-mirror 兜底 + .gguf 别名）
    └── pipeline.py        # 统一编排 synthesize()/draw()/release()
```

## 快速开始

```bash
uv sync   # 安装依赖（首次运行自动 clone + 构建 audiocpp_cli、下载 GGUF 权重）

# CLI：语音克隆（ref_text 省略时自动用 SenseVoice 转写参考音频）
uv run python vc.py <ref_audio.wav> <text.txt>

# CLI：ASR 转写参考音频（校对用）
uv run python vc.py --transcribe <ref_audio.wav>

# Web：http://localhost:38001（模型/设备等在页面「配置」Tab 选择）
uv run python web.py
```

## Web 界面（双 Tab）

- **生成页**：左栏自上而下＝参考音频（上传后立即用 SenseVoice 自动转写并
  回填）→ 参考文本（可修改）→ txt 文件（按钮式上传，读入文本框）→ 待合成
  文本；右栏＝状态 + 按抽卡次数展示的生成音频槽。
- **配置页**：置顶模型选择（omnivoice / indextts2 / fireredtts3，生成参数组
  随模型联动显示），下方基本设置＝推理设备 / 语言 / 抽卡次数 / 当前模型生成
  参数。页面设置为进程内运行期覆盖（空值回 config.py 默认或引擎默认）；
  持久化修改请编辑 `src/config.py` 顶部变量或设置同名环境变量。
- 引擎/模型按需加载：启动即用；首次 ASR 或生成自动构建/下载，任务结束立即
  释放，长时间运行无需重启。

## 设置（src/config.py 顶部变量，同名环境变量可覆盖）

| 变量 | 默认 | 说明 |
|---|---|---|
| `TTS_MODEL` | `omnivoice` | TTS 模型：`omnivoice` / `indextts2` / `fireredtts3` |
| `LANGUAGE` | 空 | 合成语言（如 `zh` / `en` / `yue`）；空 = 自动 |
| `DRAW_COUNT` | `2` | 抽卡次数 |
| `OUTPUT_DIR` | 文本所在目录 | CLI 输出目录 |
| `DEVICE` | 自动 | `cuda` / `xpu` / `mps` / `cpu`（audiocpp 后端映射；web 配置页可选） |
| `OMNI_INFERENCE_STEPS` | `32` | OmniVoice 去噪步数（0 = 引擎默认） |
| `OMNI_GUIDANCE_SCALE` | `2.0` | OmniVoice CFG 引导尺度（空 = 引擎默认） |
| `INDEXTTS_TOP_K` / `INDEXTTS_TOP_P` / `INDEXTTS_TEMPERATURE` | `30` / `0.8` / `0.8` | IndexTTS-2.5 gpt 层采样参数（官方基准） |
| `FIREREDTTS3_INFERENCE_STEPS` | `4` | FireRedTTS-3 flow 步数（0 = 引擎默认） |
| `FIREREDTTS3_GUIDANCE_SCALE` | `2.0` | FireRedTTS-3 CFG 引导（空 = 引擎默认） |
| `FIREREDTTS3_STOP_THRESHOLD` | `0.5` | FireRedTTS-3 AR 停止阈值（空 = 引擎默认） |
| `GEN_SEED` | `-1` | 固定随机种子（`-1` = 随机；设同值可复现结果） |
| `AUDIOCPP_BIN` / `AUDIOCPP_SRC` | 空 | 已编译二进制 / 已有源码（留空自动构建到 vendor/） |
| `ASR_MODEL` | 空 | 本地 SenseVoice GGUF 路径（默认经 HF 下载） |
| `WEB_IP` / `WEB_PORT` | `0.0.0.0` / `38001` | Web 监听 |
| `HF_ENDPOINT` | 空 | 直连失败自动切 hf-mirror（`HF_NO_MIRROR_FALLBACK=1` 关闭） |

## 模型

- TTS 权重从 HuggingFace 下载，文件留在 **HF 默认缓存**（本地优先 + 镜像
  兜底；下载后在缓存内生成引擎可用的 `.gguf` 硬链接别名，见上）：
  - OmniVoice（bf16）：`audio-cpp/audio.cpp-gguf` →
    `OmniVoice-GGUF/omnivoice-bf16.gguf`
  - IndexTTS-2.5（q8_0）：`audio-cpp/audio.cpp-gguf` →
    `IndexTTS2.5-GGUF/index-tts2_5-q8_0.gguf`
  - FireRedTTS-3 Base（q8_0）：`audio-cpp/audio.cpp-gguf` →
    `FireRedTTS3-Base-GGUF/fireredtts3-base-q8_0.gguf`（零样本语音克隆）
- ASR：`FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp` →
  `sensevoice-small-q8-audiocpp-v1.gguf`
- 引擎：audio.cpp 首次运行自动 clone + cmake 构建到 `vendor/audiocpp/`
  （custom 模型集：omnivoice / index_tts2 / sense_asr / fireredtts3；
  引擎/源码/权重均 gitignore，删除后首跑会重新构建下载；macOS 全新机器需
  brew libomp，audiocpp.py 已注入 include/flag）

## CLI 阶段化流程（`[i/6]`）

环境准备 → 模型准备 → 输入文件检查 → ASR → VOICECLONE → 输出文件规范；
长合成每 10s 报一次进度。Web 与 CLI 共用同一条 `pipeline.synthesize()`
链路（不显示阶段标签，改以页面事件驱动）。
