# Qwen3-TTS

Qwen3-TTS 本地模型测试项目，基于 Qwen3-TTS 0.6B Base 模型进行语音合成测试。

## 目录

- [项目简介](#项目简介)
- [测试环境](#测试环境)
- [支持的语言](#支持的语言)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [运行测试](#运行测试)
- [测试结果](#测试结果)
- [技术要点](#技术要点)
- [注意事项](#注意事项)

## 项目简介

本项目用于在本地运行 Qwen3-TTS 语音合成模型，支持 Base 模型和 CustomVoice 模型两种模式：

- **Base 模型**: 使用内置 speaker 进行语音合成
- **CustomVoice 模型**: 使用预设的多个中文 Speaker 进行语音合成

## 测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 2060 |
| 显存 | 6.0 GB |
| Base 模型 | Qwen3-TTS-12Hz-0.6B-Base |
| CustomVoice 模型 | Qwen3-TTS-12Hz-0.6B-CustomVoice |
| 模型路径 | `D:\model\` |

## 支持的语言

| 语言 | 参数值 |
|------|--------|
| 自动检测 | auto |
| 中文 | chinese |
| 英文 | english |
| 法语 | french |
| 德语 | german |
| 意大利语 | italian |
| 日语 | japanese |
| 韩语 | korean |
| 葡萄牙语 | portuguese |
| 俄语 | russian |
| 西班牙语 | spanish |

## 项目结构

```
qwen3-tts/
├── .gitignore             # Git 忽略文件
├── .python-version        # Python 版本配置
├── README.md              # 项目说明
├── cuda_check.txt         # CUDA 检查结果
├── install.ps1            # 安装脚本
├── main.py                # 主程序
├── pyproject.toml         # 项目配置
├── requirements.txt        # Python 依赖
├── test_base.py           # Base 模型测试脚本
├── test_custom_voice.py   # CustomVoice 测试脚本
└── test_local.py          # 本地模型测试脚本
```

## 环境配置

### 1. 安装依赖

```powershell
uv venv
uv pip install torch torchaudio transformers soundfile librosa numpy
```

### 2. 克隆 Qwen3-TTS 源码

```powershell
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
uv pip install -e .
cd ..
```

### 3. 下载模型

使用 ModelScope 下载模型到本地：

```powershell
# Base 模型
modelscope download qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir D:\model\Qwen3-TTS-12Hz-0.6B-Base

# CustomVoice 模型
modelscope download qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir D:\model\Qwen3-TTS-12Hz-0.6B-CustomVoice
```

### 4. 设置环境变量

```powershell
$env:HF_HOME = "D:\download\huggingface"
$env:TRANSFORMERS_CACHE = "D:\download\huggingface\hub"
$env:MODELSCOPE_CACHE = "D:\download\modelscope"
```

## 运行测试

### Base 模型测试

```powershell
cd D:\workspace\tts\qwen3-tts
.\.venv\Scripts\python.exe test_local.py
```

测试结果保存到 `test_local_result.txt`，生成的音频文件为 `test_output.wav`。

### CustomVoice 模型测试

```powershell
cd D:\workspace\tts\qwen3-tts
.\.venv\Scripts\python.exe test_custom_voice.py
```

测试结果保存到 `custom_voice_test.txt`，生成的音频文件为 `test_{Speaker}.wav`。

## 测试结果

### Base 模型测试

| 测试项 | 状态 |
|--------|------|
| 导入测试 | ✅ 通过 |
| 模型加载 | ✅ 通过 |
| 语音生成 | ✅ 通过 |
| 音频播放 | ✅ 通过 |

**音频生成详情**:

| 项目 | 值 |
|------|-----|
| 测试文本 | 你好,欢迎使用 Qwen3-TTS 语音合成系统。这是测试语音。 |
| 生成耗时 | 12.01 秒 |
| 采样率 | 24000 Hz |
| 音频长度 | 6.32 秒 |
| 文件大小 | 296.3 KB |

### CustomVoice 模型测试

| Speaker | 描述 | 生成耗时 | 音频长度 | 文件大小 | 状态 |
|---------|------|---------|---------|---------|------|
| Vivian | 明亮，略带锋芒的年轻女声 | 14.17 秒 | 7.36 秒 | 345.0 KB | ✅ |
| Serena | 温暖，温柔的年轻女声 | 11.65 秒 | 6.32 秒 | 296.3 KB | ✅ |
| Uncle_Fu | 成熟的男声，低沉醇厚 | 15.81 秒 | 8.16 秒 | 382.5 KB | ✅ |
| Dylan | 年轻的北京男声，音色清晰自然 | 12.76 秒 | 6.88 秒 | 322.5 KB | ✅ |
| Eric | 活泼的成都男声，略带沙哑的明亮 | 10.67 秒 | 5.44 秒 | 255.0 KB | ✅ |

**平均生成耗时**: 13.01 秒

## 技术要点

1. **模型加载**: 使用 `Qwen3TTSModel.from_pretrained()` 加载本地模型
2. **语言参数**: 使用正确的语言参数 `chinese` 而非 `zh`
3. **音频播放**: 使用 Windows 内置的 `winsound` 模块直接播放音频
4. **API 使用**: Base 模型使用 `model.generate()` 方法，CustomVoice 模型使用 `model.generate_custom_voice()` 方法
5. **音频格式**: CustomVoice 返回的音频格式为 `(1, N)` tensor，需要转换为 numpy 数组后保存
6. **instruct 参数限制**: `instruct` 参数用于控制语音情感/风格，但**仅支持 1.7B 模型**，0.6B 模型的 `instruct` 参数会被自动忽略

## 注意事项

- 模型必须先下载到 `D:\model` 目录
- CustomVoice 模型需要约 6GB 显存，建议使用 RTX 3060 或更高配置的显卡
- 测试脚本会自动跳过参考音频下载，使用模型内置功能
- 如果遇到内存不足错误，请使用更小的模型或增加虚拟内存
- **instruct 情感控制仅支持 1.7B 模型**，0.6B 模型不支持该功能
- 如需使用 `instruct` 参数控制语气/情感，请下载 1.7B 模型：
  ```
  modelscope download qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir D:\model\Qwen3-TTS-12Hz-1.7B-CustomVoice
  ```
