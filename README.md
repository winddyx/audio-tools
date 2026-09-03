# OmniVoice 配音工具

基于 [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) 的文本转语音工具，支持语音克隆（参考音频零样本克隆音色）、声音设计（指令合成音色）与自动音色。提供 CLI（`vc.py`）与 Web（`web.py`）两个入口，共用 `src/` 核心。

## 运行方式

```bash
uv sync        # 安装依赖（首次运行自动 clone+编译 omnivoice.cpp、下载 GGUF 权重）

# CLI：语音克隆（ref_text 省略时自动用 SenseVoiceSmall-GGUF 转写参考音频）
uv run python vc.py <ref_audio.wav> <text.txt> -l yue

# CLI：声音设计 / 自动音色
uv run python vc.py --text <text.txt> --instruct "female, low pitch, british accent"
uv run python vc.py --text <text.txt>

# CLI：ASR 转写参考音频（校对用）
uv run python vc.py --transcribe <ref_audio.wav>

# Web：http://localhost:38001
uv run python web.py
```

生成结果输出到文本文件所在目录，文件名 `<文本名>.<unix时间戳>.wav`。设置统一在 `src/config.py` 顶部变量（同名环境变量可覆盖）；CLI 参数只负责"引用哪个文件 / 语言 / 次数 / 设备"。

## 运行逻辑

```
vc.py / web.py ──> src/pipeline.py ──> src/gguf.py ──> omnivoice-tts（C++ 一次性子进程）
                      │                    │
                      └ ASR 转写(src/funasr.py)  └ 读回 WAV + 分块 sidecar → AudioResult
```

- **进程模型**：无后台常驻进程。每次生成启动一次 `omnivoice-tts` 子进程（加载 GGUF → 合成 → 退出）；ASR 也是独立子进程。
- **编排**：`pipeline.synthesize()` 统一处理 ASR 转写（克隆模式缺 ref_text 时）→ 后端生成 → 时间戳命名防覆盖 → 写 WAV；CLI/Web 只做参数与界面适配。
- **断句与分块**（C++ `text-chunker.h`）：按句末标点切句，估时长低于阈值走整篇 single-shot，否则分块逐段生成再交叉淡化拼接；**句末强标点（。！？；：）后的换行是硬切分符**，行与行之间产生真实停顿。分块文本经 `--chunks-out` sidecar 返回，落在 `AudioResult.chunks`，可按段校对 / 只重生成某段。
- **设备**：自动检测 CUDA > XPU > MPS > CPU（`GGML_BACKEND` 映射 mps→MTL0、cuda→CUDA0）；后端初始化失败自动回退 CPU 重试一次。
- **模型**：TTS 与 ASR 权重经 HuggingFace 下载到默认缓存，本地优先 + hf-mirror 兜底；已备好的机器可用 `OMNIVOICE_CPP_BIN` / `FUNASR_LLAMACPP_BIN` 直接指定二进制。