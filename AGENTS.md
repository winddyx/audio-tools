# audio-tools（repo 目录名仍是 tts-omnivoice）

语音克隆工具：参考音频 + 文本 → 克隆音色朗读。推理由 [audio.cpp](https://github.com/0xShug0/audio.cpp)
（ggml C++ 引擎，`audiocpp_cli`）子进程完成，Python 只做编排。多 TTS 模型可切换
（`TTS_MODEL`：omnivoice / indextts2 / fireredtts3 / cosyvoice3 / moss_tts_local / qwen3_tts / fish_audio /
auk / auk_flash），ASR 用
SenseVoice-Small 自动转写参考音频。
只做语音克隆，不做声音设计/自动音色。

## Project

- Python >=3.10，仅 uv 管理（`uv sync` / `uv run`，禁 pip/venv/poetry）
- 入口：`vc.py`（CLI）、`web.py`（Gradio 三 Tab：语音克隆 / 粤语翻译 / SRT 字幕生成），共用 `src/` 包
- 推理全在 C++ 侧；Python 无 torch/torchaudio 依赖
- 仓库是个人 fork，push 走 `origin main`；绝不向上游（audio.cpp 等）提交 PR/issue
- 本地分支 `v1` = 旧 omnivoice.cpp 引擎基线（738f94f），main 为 audiocpp 版

## Commands

```bash
uv sync                              # 装依赖（首次运行自动 clone+构建引擎、下载 GGUF）
uv run python vc.py <ref.wav> <text.txt>          # 语音克隆（自动 ASR）
uv run python vc.py --transcribe <ref.wav>        # 只转写参考音频（校对用）
uv run python web.py                              # Web：http://localhost:38001
uv run python -m compileall -q src vc.py web.py   # 语法检查
uv run python tools/srt_eval.py a.srt [b.srt]     # SRT 断句评测（孤行/行宽/语速/断点 F1）
```
无测试套件；验证方式 = compileall + import 冒烟 + 用真实素材端到端跑 vc.py。
注意：引擎与模型均 gitignore，删除 `vendor/ models/ .venv/` 后首跑会重新 clone+编译+下载
（macOS 全新机器需 brew libomp；audiocpp.py 已注入 include/flag，勿回退）。

## Architecture（src/ 平铺，无子包/注册机制）

- `config.py` — 唯一设置源：`Config` dataclass + 文件顶部常量（`_PROJECT_ROOT`/`VENDOR_DIR`/
  `MODELS_DIR`/`TMP_DIR`/`AUDIOCPP_*`/`TTS_MODEL`/`ASR_GGUF_*`/`WEB_*`）+ `get_best_device()`
  （纯平台探测 cuda>xpu>mps>cpu，darwin arm64→mps）+ `_quiet_hf_logs()`。所有常量 `_env(...)`
  可覆盖；**无 torch**。
- `audiocpp.py` — 模型无关引擎运行器：`_ensure_binary()`（AUDIOCPP_BIN→glob vendor/build/*
  →自动 `_clone_and_build()` custom 十一族 omnivoice,index_tts2,sense_asr,fireredtts3,
  cosyvoice3,moss,qwen3_tts,fish_audio,qwen3_asr,qwen3_forced_aligner,auk（moss 为 build
  目标名，覆盖 moss_tts_local/moss_tts_nano 两族；qwen3_asr + qwen3_forced_aligner
  供 SRT 字幕词级时间轴；auk 为社区模型目标）、clone 分支取 config.AUDIOCPP_REF（默认
  main：本项目各族都已在 main，AuK 仅 main 实现）、device→`--backend`
  映射（cuda/metal/cpu，""→best，xpu→cpu）、`run_cli()`（GPU 初始化失败自动 CPU 重试、
  `AUDIOCPP_DEBUG` 透传 stdout/stderr）、`_run_quiet()`（须传 env）。
- `omnivoice.py` / `indextts2.py` / `fireredtts3.py` / `cosyvoice3.py` /
  `moss_tts_local.py` / `qwen3_tts.py` / `fish_audio.py` — TTS 模型核心，
  各含 `_ensure_model(logger)`（本地
  手工放置优先，缺失经 HF 下载 `audio-cpp/audio.cpp-gguf` 且文件留在 HF 默认缓存）与
  `generate(cfg, logger, **kwargs)` →
  `AudioResult(audio: np.ndarray, sampling_rate, chunks)`。GGUF/族常量在各自文件。
- `auk.py` — AuK / AuK-Flash 模型核心（audiocpp `--family auk`，社区模型）：与其它族
  不同，权重是**组件目录**（`config/auk-base.yaml` + `config/auk-flash.yaml` +
  `tokenizer/` + 生成器 / Qwen2.5-Omni-3B 条件编码器 / VAE 三组 GGUF），
  `--model` 传目录、组件由
  `--session-option auk.*` 指定，下载经 `hf._ensure_gguf_tree` 在 HF 缓存内生成整棵
  别名目录（`AUK_LOCAL_DIR` 手工放置整个目录优先）；零样本克隆用法是「参考音频
  （`--voice-ref`）+ 上游 zero-shot 指令模板」（AuK 要自然语言指令而非裸文本，
  模板见 `AUK_ZERO_SHOT_TEMPLATE`，`AUK_INSTRUCT` 非空时改走 instruct TTS 指令），
  且**必须显式给输出时长**（不给就按参考音频长度输出）：由 `_estimate_duration`
  按参考音频时长 × 目标/参考文本 UTF-8 字节比估算并夹在 `AUK_MIN/MAX_DURATION`
  之间；变体由 TTS_MODEL 决定（`auk` = Base 32 步 / `auk_flash` = 固定 4 步）；
  该族不支持引擎文本分块（`--text-chunk-size` 无效），长文本受时长上限约束；
  且引擎的 AuK session 构造函数硬性要求 CUDA 后端（非 CUDA 直接抛
  `AuK native session currently requires CUDA`），故仅在 NVIDIA GPU 环境可用
  （generate 里对非 cuda/best 后端提前告警）。
- `sensevoice.py` — ASR 核心：`_transcribe_ref(cfg, logger)`（16 kHz mono 自动重采样；
  audiocpp sense_asr 族；**须以 cwd=audiocpp 仓库根运行**，silero_vad 相对路径）；解析
  stdout `text_output=` 行。
- `subtitle.py` — SRT 字幕核心：`subtitles(cfg, logger, **kwargs)`（音频 → SRT）。
  两条 ASR 路径（默认 `SRT_ASR=qwen3_asr`）：`qwen3_asr`（Qwen3-ASR +
  Qwen3-ForcedAligner，`--words-out` 词级时间轴 + `--text-out` 带标点转写，
  须给 `--session-option qwen3_asr.forced_aligner_model_path=`；词条本身无标点，
  标点由 `_punct_flags` 与词条序列按归一化字符流对齐后回填并作为断句依据）
  与 `sensevoice`（silero VAD `--task vad` 分段 → 逐段切 wav →
  `--batch-audio-dir` 一次批量转写 → 段级时间轴）；VAD 用引擎自带
  `assets/framework/models/silero_vad`（绝对路径，随 vendor 一同 clone）。
  SRT 由本模块排版：断条与折行共用同一套"从句"逻辑——以标点为界把文本
  切成从句（标点之间的整段），从句是断条与换行的原子单位，整体成条、不从
  中间切开；断条用整篇代价最优的动态规划（`_dp_cues`：每条按过短
  `SRT_MIN_CUE_WIDTH` 与语速 `SRT_CPS_MAX` 计罚，断点按标点级别计代价，
  句末/逗号倾向断、顿号等不倾向、从句内硬拆最贵、明显停顿小幅倾向；停顿超
  `SRT_MAX_GAP_SECONDS` 必断，每屏容量与 `SRT_MAX_BLOCK_SECONDS` 是硬约束）。
  折行按"最少行数 + 偏离理想行宽² + 孤行罚 + 行首虚词/行尾开引号禁则"取最优
  （`SRT_MIN_LINE_WIDTH` / `SRT_LINE_BALANCE`），长从句拆成两行大致等宽而不是
  末行只剩一两个字；`SRT_BREAK_ON_COMMA` 控制逗号是否直接收条。
  行宽按 CJK=2 计、中文不插空格。推理全在引擎侧。
  可选小 LLM 辅助（`SRT_LLM`，默认关）：`segment_llm.py` 给无标点转写补标点
  （punct）或判断每条字幕含哪几个从句（breaks），经 `--jinja` 走模型 chat 模板；
  模型不产生时间轴，输出严格校验（补标点逐字一致、分组总和不变），不过即退回
  规则结果，结果按内容哈希缓存。
  热词/上下文（`SRT_HOTWORDS` 或 kwargs `hotwords`，**仅 qwen3_asr 生效**）
  经引擎 `--text` 作为 Qwen3-ASR 的系统提示词注入（引擎侧把它当
  `request.context` 拼进 chat 模板），按 `SRT_HOTWORDS_PROMPT` 模板拼接
  （留空 = 原样用热词文本）；SenseVoice 无上下文接口，忽略并告警。取值顺序
  kwargs → `cfg.srt_hotwords` → `SRT_HOTWORDS` → 根目录 `hotword.txt`。
  文件由 web 在每次页面加载（含刷新）时重读填入热词框、每次带音频点击生成时写回
  一次，故改 `hotword.txt` 后刷新网页即生效，无需重启 web 服务。
- `llamarun.py` — llama.cpp 单次补全的共用运行器（引擎定位/自动 clone+编译
  `llama-completion`、模型定位、`run_once`）；`LLAMA_CLI` 留空即自动构建
  `vendor/llama.cpp`，模型只在 HF 默认缓存。
- `segment_llm.py` — 小 LLM 辅助断句（B1 补标点 / B2 断条分组）：提示词、
  严格校验、内容哈希缓存、分批请求；由 `subtitle.subtitles(llm_mode=...)`
  接入，默认关闭（`SRT_LLM`）。
- `hf.py` — HF 下载：本地优先 + hf-mirror 兜底（`HF_NO_MIRROR_FALLBACK=1` 关闭）；
  `_ensure_gguf_file`（单文件 .gguf 别名）/ `_ensure_gguf_tree`（多文件组件目录，
  如 AuK）都在 HF 缓存仓库目录内按真实文件名生成硬链接别名。
- `hymt2.py` — LLM 文案翻译（普通话→粤语，TTS 输入前处理）：Hy-MT2-1.8B
  （腾讯开源，官方支持粤语 yue）+ `llama-completion`（llama.cpp 子进程；
  `LLAMA_CLI` 留空时自动 clone + cmake 编译 `vendor/llama.cpp`，不依赖本机
  brew 安装，产物 `vendor/llama.cpp/build/bin/llama-completion`），
  模型走 HF 默认缓存（`HYMT2_REPO/HYMT2_FILE`，本地 `HYMT2_LOCAL` 优先）；
  `translate(text, prompt=None, logger=None)` 按 `HYMT2_CHUNK_CHARS` 长文分块
  逐段翻译；采样/设备参数在 config 顶部常量（腾讯官方推荐值）。
- `pipeline.py` — 唯一编排入口：`synthesize()`（ASR 转写→按 `cfg.tts_model` 分发
  omnivoice/indextts2/fireredtts3/cosyvoice3/moss_tts_local/qwen3_tts/fish_audio/auk
  →写盘）、`draw()`（抽卡 N 次）。
  vc/web 不直接调模型/ASR。

## Conventions

- **设置规范**：一切可调参数放 `src/config.py` 顶部变量（或 Config 字段），同名 env 覆盖；
  CLI/web 不加 `--language` 之类参数，只收"引用哪个文件"类数据参数（vc.py 仅 ref_audio/
  text_file/`--transcribe`）。
- **模型核心接口**：每个 TTS/ASR 核心实现 `_ensure_model(logger) -> str` 与
  `generate(cfg, logger, **kwargs)`；pipeline 按模型名分发，不要旁路。
  生成参数（steps/guidance/top-k/seed 等）在 config.py 顶部常量维护（env 覆盖），
  默认值 = 官方基准（omni 32 步/2.0、indextts2 30/0.8/0.8），设 0/空 回到引擎默认；
  各核心拼 CLI 时消费自己的子集；`GEN_SEED` 固定可复现（-1 = 随机）。
- **采样率不写死**：generate 后用 soundfile 从产出 wav 读实际 sr（omnivoice 24 kHz /
  indextts2 22.05 kHz）。omnivoice 克隆用 `--task tts`（不是 clon）+ `--voice-ref` +
  `--reference-text`；indextts2 用 `--task clon` + `--voice-ref`。
- **输出命名唯一事实来源**在 pipeline：`<out_dir>/<out_name>.<unix秒>.wav`，同秒冲突递增；
  CLI 传文本文件名，Web 传参考音频文件名（去扩展名，如 `voice.wav` → `voice.<秒>.wav`）。
  临时 wav 放 `TMP_DIR`（.tmp/，gitignore），用 mkstemp + finally 删除。
- **输出文本禁 emoji/特殊字符**（含带圈数字）；允许 `…`、`[1/6]`、`──`、ASCII 树。
- 模型/引擎走本地优先 + HF 兜底；device 优先级 cuda > xpu > mps > cpu，mps→metal 后端。
- 长文本自动分块：文本超 `TEXT_CHUNK_SIZE`（默认 160）时由
  `src/audiocpp._chunk_flags` 统一给引擎追加 `--text-chunk-size`，模式自动——
  输入按换行分段且每段 ≤ 上限用 `endline`（段落完整、以换行为界），否则
  `default`（按标点/CJK 断句）；`TEXT_CHUNK_MODE` 可显式覆盖。vc/web 同源，
  无需各自处理。终端/CLI 输出保持规范可读、无多余装饰（web 用默认 gradio
  主题，不引外部 CSS/theme）。
- 六阶段流程（CLI 与 Web 同构）：环境准备→模型准备→输入文件检查→ASR→VOICECLONE→
  输出文件规范；vc.py 终端以 `[i/6]` 显示，长合成每 10s 心跳报进度。
- web 三 Tab：1) 语音克隆页（VoiceClone）：左栏＝参考音频（上传即 SenseVoice
  自动转写并回填）→参考文本框→txt 文件（读入文本框）→待合成文本，右栏＝状态+
  按抽卡次数展示结果（`_clone_fn` 是生成器事件处理器：每合成出一个结果就
  yield 上屏并刷新终端，不等全部抽卡结束）；页面底部折叠区
  「模型与运行设置」＝模型选择（omnivoice/
  indextts2/fireredtts3/cosyvoice3/moss_tts_local/qwen3_tts/fish_audio/
  auk（AuK Base）/auk_flash（AuK-Flash 四步档））+ 设备/
  语言/抽卡次数，为进程内运行期设置，持久化修改仍以 src/config.py 顶部变量
  （或同名 env）为准；**生成参数（步数/采样）不在界面暴露**，统一由
  config.py 顶部常量控制（默认最高质量档）；
  2) 粤语翻译页：左栏＝普通话文案输入+执行翻译按钮，右栏＝可编辑提示词模板
  （默认提示词在根目录 `hymt2_prompt.txt` 纯文本，`config.HYMT2_PROMPT_FILE`
  指向，含 `{text}` 占位，可直接手改）+ 粤语译文输出；推理用 Hy-MT2
  （`llama-completion` 子进程，`LLAMA_CLI` 留空自动构建 `vendor/llama.cpp`），
  模型首用自动经 HF 下载，长文案自动分块（HYMT2_CHUNK_CHARS）；
  3) SRT 字幕生成页：左栏＝ASR 热词/上下文输入框（仅 Qwen3-ASR 生效；值在每次
  页面加载/刷新时重读根目录 `hotword.txt`，改文件后刷新即生效、无需重启服务，
  每次带音频点击生成时写回一次）+ 音频 wav + 生成按钮，右栏＝
  SRT 文件下载 + 预览；
  底部折叠区「模型与 ASR 设置」＝ASR 模型（`SRT_ASR` 默认 qwen3_asr：Qwen3-ASR +
  ForcedAligner 词级时间轴 + 带标点转写，按标点断句；sensevoice：VAD 分段 +
  SenseVoice 段级时间轴、无下载）+ 设备/语种 + VAD（合并间隙、最短语音段、ITN）
  + 输出开关（`SRT_PUNCTUATION` 是否输出标点、`SRT_ENUM_COMMA_AS_SPACE`
  顿号转空格、`SRT_BREAK_ON_COMMA` 逗号断条）与排版（每行宽度、每屏行数、
  单行最小宽度、单条最小宽度、单条最长秒数、句间间隙、最短显示秒数）参数，
  以及「断句辅助」（off/punct/breaks/both：小 LLM 补标点或判断分组），
  走 `src/subtitle.subtitles()`：词级路径的字幕文本取自带标点转写的逐词片段
  （`_align_tokens`，对齐器漏词也不丢字），断条取代价最优、碎条并入相邻条
  （`SRT_MIN_CUE_WIDTH`），渲染不裁行（宁可多一行也不丢文本）。
- 终端日志在标题下方、Tab 栏上方（页面级共享终端）：三个 Tab 的事件处理器
  共用同一个会话缓冲（gradio 会话隔离，刷新即清空）。
- `_run_quiet`/`run_cli` 失败抛 `RuntimeError` 带 stderr 尾部诊断（≈60 行），不在入口裸奔。
- **web 引擎/模型按需加载**：web.py 启动只启动 UI（无预热）；引擎/模型在点击"生成"时才由
  synthesize 内部定位/自动构建/下载，点击结束（finally）调 `pipeline.release()`（清
  audiocpp `_BINARY_CACHE`）——模型本就在 audiocpp_cli 子进程内按次加载、退出即卸载，
  Python 侧不留常驻资源，长时间运行无需重启。
- **模型只在 HF 默认缓存，不落工程目录**：模型经 HF 下载后一律留在默认缓存
  （~/.cache/huggingface/hub，遵循 HF_HOME/HF_HUB_CACHE）；工程 models/（GGUF_LOCAL）
  仅支持用户手工放置，自动下载绝不写入。audio.cpp 按真实文件扩展名识别权重（内部
  canonical 解析）：blobs/ 哈希名无扩展名、snapshots/ 软链被还原成 blob，都会报
  `unsupported tensor source format`；`hf._ensure_gguf_file` 在 HF 缓存仓库目录内
  硬链接生成带 .gguf 的别名（同 inode 不占空间，跨盘退复制）再返回给引擎。
- **改动流程**：每次改动完成后先审阅（通读自己的 diff，检查是否只含预期改动、无残留
  调试代码或临时文件），再 debug 验证（`compileall` + 相关冒烟/真实素材端到端），
  确认无问题才 commit 并推送到 GitHub（`git push origin main`，origin 为本仓库
  `git@github.com:winddyx/audio-tools.git`）；验证不通过或尚有疑问时不推送。

## Notes

-（可在此追加快速笔记）
