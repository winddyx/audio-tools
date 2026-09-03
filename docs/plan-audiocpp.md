# 迁移方案：tts-omnivoice v1 -> audio.cpp（audiocpp）通用引擎版

> 状态：已实施（2026-09-04，commit 于 main；旧引擎基线保留在本地分支 v1）
> 基线：738f94f 已存为本地分支 `v1`（GGUF/omnivoice.cpp 版，可随时回退）
> 更名：pyproject name / README 标题已更新为 audio-tools

> 注：本文为迁移前方案稿，结构描述与最终实现基本一致；差异见 commit 说明。

## 1. 目标

1. 推理引擎从 ServeurpersoCom/omnivoice.cpp 更换为 0xShug0/audio.cpp（`audiocpp_cli`）。
2. 弱化 OmniVoice 的"唯一模型"地位：同一引擎下可挂多种 TTS 模型（先 OmniVoice，
   预留 IndexTTS-2.5），模型由 `src/config.py` 顶部变量切换，业务代码不感知。
3. 本阶段只做语音克隆（voice clone）；删除声音设计（design）与自动音色入口。
4. 保留分阶段生成逻辑，阶段固定为：
   环境准备 -> 模型准备 -> 输入文件检查 -> ASR -> VOICECLONE -> 输出文件规范。
5. 终端与 Gradio 去多余修饰、去自定义 theme，但保持规范化、可读性。
6. ASR 从独立 llama-funasr 二进制迁到 audiocpp 的 SenseVoice 族（同源模型），
   收敛为"一个 C++ 引擎二进制 + 按 family 切换"，减少外部下载件。

## 2. 约束（沿用项目既有规则）

- Python 由 uv 管理；依赖最小化（推理全在 C++ 子进程，Python 侧只做编排）。
- 所有设置/参数/配置放 `src/config.py` 顶部变量；CLI 不新增业务启动参数
  （保留 vc.py 的"引用哪个文件"类位置参数与 `--transcribe` 等业务参数即可）。
- 输出文本（日志/终端/Web/文档）禁止 emoji 与特殊字符；允许 `…`、`[1/4]`、`──`。
- 设备优先级 cuda > xpu > mps > cpu（audiocpp 后端名映射见 5.2）。
- HF 下载走默认缓存，本地优先 + hf-mirror 回退（config 顶部变量控制开关）。

## 3. 目录结构（目标形态）

```
audio-tools/
|-- vc.py                  # CLI 业务入口：仅语音克隆 + ASR 子命令（转写参考音频）
|-- web.py                 # Web 业务入口：单页语音克隆
|-- pyproject.toml         # name = "audio-tools"（迁移完成时改名）
|-- README.md
|-- .gitignore             # 追加 vendor/audiocpp、.tmp/
|-- vendor/                # gitignore；audiocpp 源码与构建产物
|   `-- audiocpp/          #   （clone 0xShug0/audio.cpp + 本地构建）
|-- .tmp/                  # gitignore；运行期临时 wav/文本
`-- src/
    |-- __init__.py        # 公开 re-export（保持现状风格）
    |-- config.py          # 全局设置：输入输出、设备、ASR/TTS 模型切换、HF 代理、路径
    |-- hf.py              # HF 下载：本地优先 + hf-mirror 回退（沿用）
    |-- audiocpp.py        # 推理引擎通用运行器（新核心，见 5）
    |-- omnivoice.py       # OmniVoice 模型核心：语言表/参数表/GGUF 包定义（弱化为可插拔模型之一）
    |-- indextts2.py       # IndexTTS-2.5 模型核心（预留骨架；本期可只留占位 + 未启用开关）
    |-- sensevoice.py      # SenseVoice-Small ASR 核心（audiocpp sense_asr 族封装）
    `-- pipeline.py        # 业务流水线：阶段编排（环境/模型/输入/ASR/clone/输出）+ 心跳 + 抽卡
```

删除/收敛的现文件：`src/gguf.py`（并入 audiocpp.py + omnivoice.py）、`src/funasr.py`
（并入 sensevoice.py）、`src/omni.py` 的生成参数表（并入各模型核心）。

## 4. 分阶段生成逻辑（vc.py / web.py 共用）

阶段固定为 6 段，编号输出统一 `[i/6]`；CLI 与 Web 状态条同构：

| # | 阶段 | 内容 | 现有映射 |
|---|---|---|---|
| 1 | 环境准备 | 定位/构建 `audiocpp_cli`（vendor 缓存命中则跳过），确保 .tmp 目录 | gguf._ensure_binary |
| 2 | 模型准备 | 按 config 的 TTS_MODEL 下载/定位 GGUF 包（含 SenseVoice GGUF 若 ASR 需要） | gguf._ensure_gguf / funasr |
| 3 | 输入文件检查 | 校验 ref_audio/text 文件、draw_count、设备可用性；早失败早退出 | vc._validate_inputs |
| 4 | ASR | ref_text 缺失时用 SenseVoice 转写参考音频（进程内缓存，抽卡只转一次） | _transcribe_ref |
| 5 | VOICECLONE | 多轮抽卡：逐轮调 audiocpp_cli，心跳线程报时，失败 CPU 回退 | vc 阶段 3 |
| 6 | 输出文件规范 | 结果落盘 `<out_name>.<ts>.wav`，汇总清单与耗时 | vc 阶段 4 |

CLI 终端格式（规范化、可读、无装饰）：
```
[1/6] 环境准备
  audiocpp_cli 就绪（…）
[2/6] 模型准备
  TTS 模型: omnivoice (…gguf)
...
[5/6] VOICECLONE
  [第 1/2 次] 生成中 …
    已用时 12 s，继续生成中 …
  [第 2/2 次] 生成中 …
 生成完成: 耗时 35.2 s
[6/6] 输出文件规范
  .../out_001.wav
共 2 个文件，写入 …
```

## 5. 推理引擎层设计

### 5.1 audiocpp.py（通用运行器，模型无关）

职责：
- `ensure_engine()`：vendor/audiocpp 不存在则 clone 0xShug0/audio.cpp + cmake 构建
  `audiocpp_cli`；构建参数走 config 顶部变量（含后端开关）；构建进度仅 GGUF_DEBUG
  时透传，否则吞掉并保留尾部报错（沿用现有 `_run_quiet/_run_progress` 思路）。
- `run(family, model_path, args...) -> AudioResult`：构造
  `audiocpp_cli --task tts --family <family> --model <pkg> [--backend] --text …
   [--voice-ref] [--reference-text] [--language] [-o out.wav]`；一次性子进程，
  捕获 stderr，失败尾部 60 行报错；GGUF_DEBUG（改名 AUDIOCPP_DEBUG）时先 dump 全量再抛。
- 临时文件用 `.tmp/` mkstemp，finally 清理；文本传参（audiocpp 示例为 `--text`，
  不依赖 stdin，迁移时以实测为准）。
- 后端映射（config.device -> --backend）：cuda->cuda、xpu->（无，回退 cpu 并告警）、
  mps->metal、cpu->cpu；backend 初始化失败自动 CPU 重试一次（沿用现逻辑）。
- 抽卡同一 handle/session：本期保持"每轮一次性子进程"不动（与 v1 行为一致），
  server/常驻列为二期候选。

### 5.2 模型核心（可插拔）

每个模型核心导出一致接口，供 pipeline 无差别调用：

```
MODEL = "omnivoice"            # 或 "indextts2"，在 config.py 顶部切换
```

- omnivoice.py：HF 包 audio-cpp/audio.cpp-gguf 的 `omnivoice-{bf16,f16,q8_0}.gguf`；
  语言表沿用现有 19 语言名 -> audiocpp 语言提示；参数表
  （num_step->--num-inference-steps 等按 audiocpp 实测校准）。
- indextts2.py：本期仅建骨架 + `ENABLE=False`，字段齐（HF 包/语言/参数映射），
  待实测后打开。
- sensevoice.py：ASR 核心，封装 audiocpp `sense_asr` 族（SenseVoice-Small q8），
  输出参考音频文本；替代 funasr.py 与 llama-funasr 二进制下载。

config.py 顶部新增/改名示例（方向，最终以实施为准）：

```python
# ── 引擎 ──
AUDIOCPP_SRC = ""               # 已有源码目录（默认 vendor/audiocpp）
AUDIOCPP_BIN = ""               # 已编译二进制（留空自动构建）
AUDIOCPP_BUILD_ARGS = ""
AUDIOCPP_DEBUG = False

# ── TTS 模型（弱化 omnivoice：改这里即换模型，业务代码不动）──
TTS_MODEL = "omnivoice"         # "omnivoice" | "indextts2"
TTS_GGUF_REPO = "audio-cpp/audio.cpp-gguf"
TTS_GGUF_FILE = "OmniVoice-GGUF/omnivoice-bf16.gguf"   # 按 TTS_MODEL 由核心提供默认值
TTS_LANG = ""                   # 默认语言（留空自动）

# ── ASR（SenseVoice，audiocpp sense_asr）──
ASR_MODEL = "sensevoice"
ASR_GGUF_REPO = "..."           # sensevoice q8 GGUF 来源
```

## 6. Web（web.py）与 CLI（vc.py）收整

- 只保留语音克隆：删除 声音设计 Tab、design/_instruct/_gen_settings 分支、
  自动音色（无 ref_audio 无 instruct）路径；页头描述去掉 design 宣传。
- 去掉自定义 theme：删除 `_THEME = gr.themes.Soft.from_hub(...)` 与字体覆盖，
  `demo.launch(theme=默认)`；保留排版结构（块标题、说明文字、间距）但不用装饰元素。
- 界面参数与 config 顶部变量一致（不做 argparse 扩展）；服务地址/端口/自动开浏览器
  沿用 config 顶部变量。
- vc.py 仅留：语音克隆（ref_audio + text 必填）与 `--transcribe` 子命令；阶段输出按 4 节。

## 7. 迁移步骤与验收

| 步 | 内容 | 验收 |
|---|---|---|
| 0 | 已建分支 v1 存基线 | 738f94f 可回退（已完成） |
| 1 | vendor 内 clone + 构建 audiocpp，验证 `audiocpp_cli` 可跑 | 本机产出可执行文件 |
| 2 | 用官方 omnivoice GGUF 手动跑通一句克隆，与 v1 输出 A/B 听感 | 听感可接受（漂移确认） |
| 3 | 落地 src/audiocpp.py + omnivoice.py + sensevoice.py，pipeline 切到新引擎 | vc.py 语音克隆端到端通过 |
| 4 | 收整 config.py 顶部变量（TTS_MODEL/ASR/引擎/代理） | 设置只经 config 顶部变量 |
| 5 | web.py 单页克隆 + 去 theme；vc.py 六阶段输出 | UI 无多余装饰、阶段可读 |
| 6 | 删除 gguf.py/funasr.py/omni.py 残留与 design/自动音色代码 | compileall + 全流程冒烟通过 |
| 7 | pyproject/README 更名 audio-tools | uv sync + 双入口可用 |
| 8 | commit 到 main；分支 v1 保留 | 工作区干净 |

## 8. 待实测/开放问题

1. audiocpp 的 omnivoice 是否仍走 k2-fsa/OmniVoice 同源权重、音质漂移幅度（文档标
   `Pass (drift)`，需实听）。
2. `audiocpp_cli` 文本输入形态（--text 参数 vs stdin）与 `--language` 是否接受中文名。
3. sense_asr 的 GGUF 包来源与参数（--task asr --family sense_asr），VAD 是否需要独立
   加载（现 llama-funasr 需 fsmn-vad）。
4. audio.cpp 整体构建体积/耗时（62+ 模型族框架）是否接受；是否可裁剪仅编 omnivoice +
   sense_asr 相关目标。
5. IndexTTS-2.5 本期仅骨架，实测后另开任务启用。
