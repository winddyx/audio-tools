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
  统一走 `pipeline.synthesize()`；语音克隆页的"上传即 ASR"直连
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
- **生成参数**：默认 = 最高质量档（步数按官方基准翻倍：OmniVoice 64 步/CFG
  2.0、FireRedTTS-3 10 步/CFG 2.0/停止阈值 0.5、CosyVoice-3 25 步/top-k 25；
  采样类维持官方基准：IndexTTS-2.5 top-k 30/top-p 0.8/temperature 0.8 等），
  常量在 `src/config.py`（同名 env 覆盖，0/空 = 不传回引擎默认）；
  **Web 不在界面暴露生成参数**，调整即改 config.py 顶部变量或同名 env。

## 目录结构

```
├── vc.py                  # CLI 入口（语音克隆 / --transcribe）
├── web.py                 # Gradio 入口（三 Tab：语音克隆 / 粤语翻译 / SRT 字幕）
├── tools/
│   └── srt_eval.py        # SRT 断句评测（孤行/行宽/语速/断点 F1，离线只读）
└── src/
    ├── config.py          # 全局设置（唯一设置源：顶部变量 + env 覆盖）
    ├── audiocpp.py        # 推理引擎运行器（audio.cpp，模型无关，按需构建/释放）
    ├── omnivoice.py       # OmniVoice 模型核心（TTS 语音克隆）
    ├── indextts2.py       # IndexTTS-2.5 模型核心（TTS 语音克隆）
    ├── fireredtts3.py     # FireRedTTS-3 Base 模型核心（零样本语音克隆）
    ├── cosyvoice3.py      # CosyVoice-3 模型核心（零样本语音克隆）
    ├── moss_tts_local.py  # MOSS-TTS-Local v1.5 模型核心（零样本语音克隆）
    ├── qwen3_tts.py       # Qwen3-TTS 12Hz 1.7B Base 模型核心（零样本语音克隆）
    ├── fish_audio.py      # Fish Audio S2-Pro 模型核心（零样本语音克隆）
    ├── auk.py             # AuK / AuK-Flash 模型核心（零样本语音克隆，组件目录）
    ├── sensevoice.py      # SenseVoice-Small ASR 核心（参考音频转写）
    ├── subtitle.py        # SRT 字幕核心（VAD+SenseVoice 段级 / Qwen3-ASR 词级）
    ├── hf.py              # HuggingFace 下载（本地优先 + hf-mirror 兜底 + .gguf 别名）
    ├── llamarun.py        # llama.cpp 单次补全运行器（hymt2 / segment_llm 共用）
    ├── segment_llm.py     # 小 LLM 辅助断句（补停顿分隔 / 断条分组，默认关闭）
    └── pipeline.py        # 统一编排 synthesize()/draw()/release()
```

## 快速开始

```bash
uv sync   # 安装依赖（首次运行自动 clone + 构建 audiocpp_cli、下载 GGUF 权重）

# CLI：语音克隆（ref_text 省略时自动用 SenseVoice 转写参考音频）
uv run python vc.py <ref_audio.wav> <text.txt>

# CLI：ASR 转写参考音频（校对用）
uv run python vc.py --transcribe <ref_audio.wav>

# Web：http://localhost:38001（页面底部「模型与运行设置」选模型/设备/语言）
uv run python web.py
```

## 字幕断条（SRT）

断条与折行共用同一套"从句"逻辑，任何断法都不切词组：

- **从句**：以标点为界把文本切成从句（标点之间的整段），从句是断条与折行的原子单位，整体成条/成行；只有从句自己就宽过一行（标点之间无处可断）时才在该从句内部拆。
- **断条**：全篇取代价最小的断法（动态规划）而不是逐条贪心。每条字幕按"过短、语速过快"计罚，每个断点按标点级别计代价——句末标点与逗号倾向在此断、顿号/分号/冒号不倾向、从句内硬拆代价最大、明显停顿小幅倾向；停顿超过 `SRT_MAX_GAP_SECONDS` 的位置必须断，每屏容量与 `SRT_MAX_BLOCK_SECONDS` 是硬约束。
- **折行**：行数取最少行数，理想行宽 = 总宽 / 行数，再按"偏离理想行宽² + 孤行罚 + 行首虚词/行尾开引号禁则"取最优，因此长从句被拆成两行大致等宽，而不是填满首行、末行只剩一两个字。
- **可选小 LLM 辅助**（`SRT_LLM`，默认关闭）：`punct` 给无标点转写补标点（SenseVoice 关 ITN 时有用），`breaks` 让模型判断每条字幕该含哪几个从句。模型**不产生时间轴**（时间永远来自 ForcedAligner / VAD），输出严格校验（补标点要"去标点后逐字一致"、分组要"每组 ≥ 1 且总和不变"），不通过就退回规则结果，结果按内容哈希缓存。0.6B 模型补标点偏稀，正式用建议 `SRT_LLM_REPO=Qwen/Qwen3-1.7B-GGUF`、`SRT_LLM_FILE=Qwen3-1.7B-Q8_0.gguf`。
- **评测**：`uv run python tools/srt_eval.py 字幕.srt [人工校对.srt]` 输出条数/行数/孤行/行首虚词/超宽行/语速，给了校对版再算断点 F1 与"需要改动的条数"。

## Web 界面（三 Tab）

- **语音克隆页（VoiceClone）**：左栏自上而下＝参考音频（上传后立即用
  SenseVoice 自动转写并回填）→ 参考文本（可修改）→ txt 文件（按钮式上传，
  读入文本框）→ 待合成文本；右栏＝状态 + 按抽卡次数展示的生成音频槽。
- **模型与运行设置**（语音克隆页底部折叠区）：模型选择（omnivoice /
  indextts2 / fireredtts3 / cosyvoice3 / moss_tts_local / qwen3_tts /
  fish_audio / auk / auk_flash）+ 推理设备 / 语言 / 抽卡次数。设置为进程内运行期覆盖；
  持久化修改请编辑 `src/config.py` 顶部变量或设置同名环境变量。
  生成参数（步数 / 采样等）不在界面暴露，统一在 config.py 顶部调整
  （默认最高质量档）。
- **粤语翻译页**：普通话文案 → 香港粤语译文（Hy-MT2-1.8B，可编辑提示词）。
- **SRT 字幕生成页**：左栏上传音频（wav）+ 生成按钮；右栏＝生成的 SRT 文件
  （点击下载）+ SRT 预览。底部「模型与 ASR 设置」＝ASR 模型（`sensevoice`
  ＝silero VAD 分段 + SenseVoice 段级时间轴，复用已缓存权重；`qwen3_asr`
  ＝Qwen3-ASR + Qwen3-ForcedAligner 词级时间轴，首次约 2.3 GB 下载）+
  设备 / 语种 / ITN + 分段与排版参数（合并间隙、最短语音段、每行宽度、
  每屏行数、单条最长秒数）。
- 引擎/模型按需加载：启动即用；首次 ASR 或生成自动构建/下载，任务结束立即
  释放，长时间运行无需重启。
- 终端日志在标题下方、Tab 栏上方（页面级共享终端）：三个 Tab 的事件处理器
  都把日志写进同一个会话缓冲（gradio 会话隔离，刷新即清空）。

## 设置（src/config.py 顶部变量，同名环境变量可覆盖）

| 变量 | 默认 | 说明 |
|---|---|---|
| `TTS_MODEL` | `omnivoice` | TTS 模型：`omnivoice` / `indextts2` / `fireredtts3` / `cosyvoice3` / `moss_tts_local` / `qwen3_tts` / `fish_audio` / `auk`（AuK Base）/ `auk_flash`（AuK-Flash 四步档） |
| `LANGUAGE` | 空 | 合成语言（如 `zh` / `en` / `yue`）；空 = 自动 |
| `DRAW_COUNT` | `2` | 抽卡次数 |
| `OUTPUT_DIR` | 文本所在目录 | CLI 输出目录 |
| `DEVICE` | 自动 | `cuda` / `xpu` / `mps` / `cpu`（audiocpp 后端映射；web 底部「模型与运行设置」可选） |
| `OMNI_INFERENCE_STEPS` | `64` | OmniVoice 去噪步数（最高质量档，官方基准 32；0 = 引擎默认） |
| `OMNI_GUIDANCE_SCALE` | `2.0` | OmniVoice CFG 引导尺度（空 = 引擎默认） |
| `INDEXTTS_TOP_K` / `INDEXTTS_TOP_P` / `INDEXTTS_TEMPERATURE` | `30` / `0.8` / `0.8` | IndexTTS-2.5 gpt 层采样参数（官方基准） |
| `FIREREDTTS3_INFERENCE_STEPS` | `10` | FireRedTTS-3 flow 步数（最高质量档，官方基准 4；0 = 引擎默认） |
| `FIREREDTTS3_GUIDANCE_SCALE` | `2.0` | FireRedTTS-3 CFG 引导（空 = 引擎默认） |
| `FIREREDTTS3_STOP_THRESHOLD` | `0.5` | FireRedTTS-3 AR 停止阈值（空 = 引擎默认） |
| `COSYVOICE3_TOP_K` | `25` | CosyVoice-3 AR top-k（0 = 引擎默认） |
| `COSYVOICE3_INFERENCE_STEPS` | `25` | CosyVoice-3 flow 步数（最高质量档，官方基准 10；0 = 引擎默认） |
| `MOSS_TEMPERATURE` / `MOSS_TOP_P` / `MOSS_TOP_K` / `MOSS_REPETITION_PENALTY` | `1.7` / `0.8` / `25` / `1.0` | MOSS-TTS-Local 音频 token 采样参数（空/0 = 引擎默认；该族未暴露 seed） |
| `QWEN3TTS_TEMPERATURE` / `QWEN3TTS_TOP_P` / `QWEN3TTS_TOP_K` / `QWEN3TTS_REPETITION_PENALTY` | `0.9` / `1.0` / `50` / `1.05` | Qwen3-TTS 主 talker 采样参数（空/0 = 引擎默认；种子用 `GEN_SEED`） |
| `FISH_AUDIO_TEMPERATURE` / `FISH_AUDIO_TOP_P` / `FISH_AUDIO_TOP_K` / `FISH_AUDIO_MAX_NEW_TOKENS` | `0.8` / `0.8` / `30` / `1024` | Fish Audio S2-Pro 采样参数（空/0 = 引擎默认；种子用 `GEN_SEED`） |
| `GEN_SEED` | `-1` | 固定随机种子（`-1` = 随机；设同值可复现结果） |
| `AUK_GGUF_DTYPE` | `q8_0` | AuK 生成器精度档：`f32`（6.1 GB）/ `f16`（3.1 GB）/ `q8_0`（1.6 GB） |
| `AUK_QWEN_GGUF` / `AUK_VAE_GGUF` | `qwen2.5-omni-3b-q8_0.gguf` / `auk-vae-f32.gguf` | AuK 条件编码器（Qwen2.5-Omni-3B，可换 `qwen2.5-omni-3b-bf16.gguf`）/ VAE（固定 F32） |
| `AUK_ZERO_SHOT_TEMPLATE` | `Say the following with the same voice: "{text}"` | AuK 零样本克隆指令模板（含 `{text}` 占位；AuK 需要自然语言指令而非裸文本） |
| `AUK_INSTRUCT` | 空 | 声音描述（引擎 `instruct`）：非空时待朗读文本按普通文本传入并由引擎包成 instruct TTS 指令；空 = 纯零样本克隆 |
| `AUK_DURATION_SEC` / `AUK_DURATION_SCALE` | 空 / `1.0` | AuK 输出时长（秒）：空 = 自动估算（参考音频时长 × 目标/参考文本 UTF-8 字节比，缺参考文本时按 `AUK_CHARS_PER_SECOND`），再乘系数 |
| `AUK_CHARS_PER_SECOND` / `AUK_MIN_DURATION` / `AUK_MAX_DURATION` | `4.5` / `1.0` / `60.0` | 无参考文本时的语速估算（字/秒）/ 输出时长下限 / 上限（估得超上限则截到上限并告警） |
| `AUK_INFERENCE_STEPS` / `AUK_GUIDANCE_SCALE` | `32` / `2.0` | AuK Base 采样步数 / 引导强度（官方基准；Flash 固定 4 步且关闭引导，两项无效；0/空 = 引擎默认） |
| `AUK_MEM_SAVER` / `AUK_ATTENTION` | `false` / 空 | 分阶段释放权重（省显存/内存但明显变慢）/ Flow 注意力（空 = 引擎 `auto`） |
| `TEXT_CHUNK_SIZE` | `160` | 长文本分块每块上限（`0` = 不分块；修复长文吞字/乱码，实测 OmniVoice 相似度 0.877→0.982） |
| `TEXT_CHUNK_MODE` | 空 | 分块模式：空 = 自动（输入分段且每段 ≤ 上限用 `endline` 按换行，否则 `default` 按标点断句）；可设 `endline` / `tag_aware` / `japanese` / `default` |
| `AUDIOCPP_BIN` / `AUDIOCPP_SRC` | 空 | 已编译二进制 / 已有源码（留空自动构建到 vendor/） |
| `AUDIOCPP_REF` | `main` | 引擎 clone/构建分支（AuK 只在 main 实现；dev 是 main 的历史提交） |
| `ASR_MODEL` | 空 | 本地 SenseVoice GGUF 路径（默认经 HF 下载） |
| `SRT_ASR` | `qwen3_asr` | SRT 字幕的 ASR 路径：`qwen3_asr`（Qwen3-ASR + ForcedAligner，词级时间轴 + 带标点转写，默认）/ `sensevoice`（VAD 分段 + SenseVoice，段级时间轴、无下载） |
| `SRT_VAD_MERGE_GAP` / `SRT_VAD_MIN_SPEECH` | `0.5` / `0.3` | VAD 相邻段合并间隙 / 丢弃的最短语音段（秒；仅 sensevoice 路径） |
| `SRT_ITN` | `true` | SenseVoice 反向文本规范化（数字/标点） |
| `SRT_QWEN3_PUNCTUATION` | `true` | Qwen3-ASR 保留标点（`preserve_punctuation`）：带标点转写用于按标点断句与输出 |
| `SRT_PUNCTUATION` | `false` | 字幕文本是否输出标点（关掉只留正文；断句仍按标点判断） |
| `SRT_ENUM_COMMA_AS_SPACE` | `true` | 顿号（、）输出为空格：`赣州、贵阳` → `赣州 贵阳` |
| `SRT_MAX_LINE_WIDTH` / `SRT_MAX_LINES` | `32` / `2` | 字幕每行宽度（CJK 按 2 计，32 ≈ 16 汉字）/ 每屏最大行数（断条以标点为界，一屏 2 行才能让多数整句落在同一条里） |
| `SRT_MAX_BLOCK_SECONDS` / `SRT_MAX_GAP_SECONDS` / `SRT_MIN_BLOCK_SECONDS` | `6.0` / `1.0` / `0.8` | 单条字幕最长秒数（超过则在下一个从句边界断条）/ 句间断句间隔 / 单条最短秒数 |
| `SRT_MIN_CUE_WIDTH` | `8` | 碎条阈值（宽度，8 = 4 汉字）：低于此宽度的条目并入相邻条（上一条停在句末时优先并入下一条；0 = 关闭） |
| `SRT_CPS_MAX` | `17` | 单条字幕语速上限（CJK 字数/秒）：按超出比例计罚，倾向另起一条或延长显示（0 = 不检查） |
| `SRT_BREAK_ON_COMMA` | `true` | 逗号（，）处即收条：一条字幕停在逗号上（顿号、分号、冒号不受影响，仍只作从句边界，容量不够时才在这里断） |
| `SRT_MIN_LINE_WIDTH` / `SRT_LINE_BALANCE` | `8` / `1.0` | 折行：孤行阈值（宽度低于它算孤行）/ 均衡权重（从句宽过一行时拆行取"偏离理想行宽² + 孤行罚 + 行首行尾禁则"最优；0 = 行内尽量填满） |
| `SRT_LLM` / `SRT_LLM_MODE` | `false` / `both` | 小 LLM 辅助断句总开关与形态（`punct` 补标点 / `breaks` 判断每条含哪几个从句 / `both` / `off`） |
| `SRT_LLM_REPO` / `SRT_LLM_FILE` | `Qwen/Qwen3-0.6B-GGUF` / `Qwen3-0.6B-Q8_0.gguf` | 断句辅助模型（`SRT_LLM_LOCAL` 可指本地 .gguf；补标点建议换 1.7B 级模型） |
| `SRT_LLM_TEMPERATURE` / `SRT_LLM_SEED` / `SRT_LLM_BATCH_CLAUSES` / `SRT_LLM_PUNCT_CHARS` / `SRT_LLM_CACHE` | `0.0` / `1234` / `24` / `60` / `true` | 采样固定可复现 / 固定种子 / 每次请求的从句数 / 补标点分块字数 / 按内容哈希缓存结果 |
| `SRT_QWEN3_ASR_FILE` | `Qwen3-ASR-0.6B-GGUF/qwen3-asr-0.6b-q8_0.gguf` | Qwen3-ASR 权重（HF 仓库 `audio-cpp/audio.cpp-gguf`；`SRT_QWEN3_ASR_LOCAL` 可指本地文件） |
| `SRT_QWEN3_ALIGNER_FILE` | `Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf` | Qwen3-ForcedAligner 权重（词级时间戳；`SRT_QWEN3_ALIGNER_LOCAL` 可指本地文件） |
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
  - CosyVoice-3（q8_0）：`audio-cpp/audio.cpp-gguf` →
    `CosyVoice3-GGUF/cosyvoice3-q8_0.gguf`（零样本语音克隆）
  - MOSS-TTS-Local v1.5（q8_0）：`audio-cpp/audio.cpp-gguf` →
    `MOSS-TTS-Local-v1.5-GGUF/moss-tts-local-v1.5-q8_0.gguf`
    （零样本语音克隆，48 kHz）
  - Qwen3-TTS 12Hz 1.7B Base（q8_0）：`audio-cpp/audio.cpp-gguf` →
    `Qwen3-TTS-12Hz-1.7B-Base-GGUF/qwen3-tts-12hz-1.7b-base-q8_0_v2.gguf`
    （零样本语音克隆）
  - Fish Audio S2-Pro（q8_0）：`audio-cpp/audio.cpp-gguf` →
    `Fish-Audio-S2-Pro-GGUF/fish-audio-s2-pro-q8_0.gguf`
    （零样本语音克隆）
  - AuK / AuK-Flash（组件目录，默认 q8_0）：`audio-cpp/AuK-Base-and-Flash-GGUF`
    → 生成器 `auk-base-q8_0.gguf` / `auk-flash-q8_0.gguf` + 条件编码器
    `qwen2.5-omni-3b-q8_0.gguf`（Qwen2.5-Omni-3B）+ `auk-vae-f32.gguf` +
    `config/auk-base.yaml` + `config/auk-flash.yaml` + `tokenizer/`
    （零样本语音克隆，24 kHz；引擎的 AuK session 硬性要求 CUDA 后端，
    即需要 NVIDIA GPU，macOS/Metal 与 CPU 上会直接报错）
- ASR：`FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp` →
  `sensevoice-small-q8-audiocpp-v1.gguf`
- 引擎：audio.cpp 首次运行自动 clone + cmake 构建到 `vendor/audiocpp/`
  （custom 模型集：omnivoice / index_tts2 / sense_asr / fireredtts3 /
  cosyvoice3 / moss / qwen3_tts / fish_audio / qwen3_asr /
  qwen3_forced_aligner / auk；moss 目标覆盖 moss_tts_local 与 moss_tts_nano
  两族；clone 分支由 `AUDIOCPP_REF` 决定，默认 `main`——AuK 仅在 main 实现；
  引擎/源码/权重均 gitignore，删除后首跑会重新构建下载；macOS 全新机器需
  brew libomp，audiocpp.py 已注入 include/flag）

## CLI 阶段化流程（`[i/6]`）

环境准备 → 模型准备 → 输入文件检查 → ASR → VOICECLONE → 输出文件规范；
长合成每 10s 报一次进度。Web 与 CLI 共用同一条 `pipeline.synthesize()`
链路（不显示阶段标签，改以页面事件驱动）。
