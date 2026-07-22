# OmniVoice Toolkit

基于 [OmniVoice](https://github.com/k2-fsa/OmniVoice) 的配音工具集，支持参考音频克隆音色、长文本转语音、批量生成。

## 环境要求

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) 包管理器

## 安装

```bash
uv sync
```

首次运行会自动下载模型（~2GB），保存在 `models/` 目录下。

## 用法

### 单文件配音（`txt.py`）

```bash
# 最简形式：指定参考音频和文本文件
uv run python txt.py /path/to/ref.wav /path/to/text.txt

# 自定义生成次数（默认 2）
uv run python txt.py /path/to/ref.wav /path/to/text.txt -n 3

# 仍支持环境变量方式
REF_AUDIO=/path/to/ref.wav TEXT_PATH=/path/to/text.txt uv run python txt.py
```

每轮生成一个 `.wav` 文件，输出到文本文件所在目录。

### 批量配音（`batch.py`）

按行读取文本文件，每行生成语音。参考环境变量方式配置：

```bash
REF_AUDIO=/path/to/ref.wav TEXT_FILE=/path/to/lines.txt uv run python batch.py
```

## 项目结构

```
├── omni.py        — 通用封装（模型加载、音频转换、转录）
├── txt.py         — 单文件长文本配音（支持 CLI 参数）
├── batch.py       — 批量逐行配音
├── pyproject.toml — 项目元数据与依赖
└── models/        — 模型缓存（gitignore）
```
