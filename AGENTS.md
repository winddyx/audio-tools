# audio-tools（repo 目录名仍是 tts-omnivoice）

语音克隆工具：参考音频 + 文本 → 克隆音色朗读。推理由 [audio.cpp](https://github.com/0xShug0/audio.cpp)
（ggml C++ 引擎，`audiocpp_cli`）子进程完成，Python 只做编排。多 TTS 模型可切换
（`TTS_MODEL`：omnivoice / indextts2 / fireredtts3 / cosyvoice3），ASR 用
SenseVoice-Small 自动转写参考音频。
只做语音克隆，不做声音设计/自动音色。

## Project

- Python >=3.10，仅 uv 管理（`uv sync` / `uv run`，禁 pip/venv/poetry）
- 入口：`vc.py`（CLI）、`web.py`（Gradio 双 Tab：生成 / 配置），共用 `src/` 包
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
  →自动 `_clone_and_build()` custom 五族 omnivoice,index_tts2,sense_asr,fireredtts3,
  cosyvoice3；clone 分支取 config.AUDIOCPP_REF，默认 dev——cosyvoice3 目前只在
  引擎 dev 分支实现）、device→`--backend`
  映射（cuda/metal/cpu，""→best，xpu→cpu）、`run_cli()`（GPU 初始化失败自动 CPU 重试、
  `AUDIOCPP_DEBUG` 透传 stdout/stderr）、`_run_quiet()`（须传 env）。
- `omnivoice.py` / `indextts2.py` / `fireredtts3.py` / `cosyvoice3.py` — TTS 模型核心，
  各含 `_ensure_model(logger)`（本地
  手工放置优先，缺失经 HF 下载 `audio-cpp/audio.cpp-gguf` 且文件留在 HF 默认缓存）与
  `generate(cfg, logger, **kwargs)` →
  `AudioResult(audio: np.ndarray, sampling_rate, chunks)`。GGUF/族常量在各自文件。
- `sensevoice.py` — ASR 核心：`_transcribe_ref(cfg, logger)`（16 kHz mono 自动重采样；
  audiocpp sense_asr 族；**须以 cwd=audiocpp 仓库根运行**，silero_vad 相对路径）；解析
  stdout `text_output=` 行。
- `hf.py` — HF 下载：本地优先 + hf-mirror 兜底（`HF_NO_MIRROR_FALLBACK=1` 关闭）。
- `pipeline.py` — 唯一编排入口：`synthesize()`（ASR 转写→按 `cfg.tts_model` 分发
  omnivoice/indextts2/fireredtts3/cosyvoice3→写盘）、`draw()`（抽卡 N 次）。
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
- web 双 Tab（生成/配置）：生成页左栏＝参考音频（上传即 SenseVoice 自动转写并回填）→
  参考文本框→txt 文件（读入文本框）→待合成文本，右栏＝状态+按抽卡次数展示结果；配置页
  提供模型（omnivoice/indextts2/fireredtts3）、设备、语言、抽卡次数与当前模型生成参数，
  为进程内运行期设置（事件回调里经 gen_kwargs 覆盖 config 常量；空值回常量/引擎默认），
  持久化修改仍以 src/config.py 顶部变量（或同名 env）为准。
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

## Notes

-（可在此追加快速笔记）
