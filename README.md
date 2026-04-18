# Qwen3-TTS

Qwen3-TTS 本地模型测试项目，基于 Qwen3-TTS 0.6B Base 模型进行语音合成测试。

## 测试环境

- **GPU**: NVIDIA GeForce RTX 2060
- **显存**: 6.0 GB
- **模型**: Qwen3-TTS-12Hz-0.6B-Base
- **模型路径**: `D:\model\Qwen3-TTS-12Hz-0.6B-Base`

## 测试结果

| 测试项 | 状态 |
|--------|------|
| 导入测试 | ✅ 通过 |
| 模型加载 | ✅ 通过 |
| 参考音频 | ✅ 通过 |
| 语音生成 | ✅ 通过 |

## 音频生成详情

- **测试文本**: 你好,欢迎使用 Qwen3-TTS 语音合成系统。这是测试语音。
- **生成耗时**: 15.42 秒
- **采样率**: 24000 Hz
- **音频长度**: 6.64 秒
- **文件大小**: 311.3 KB

## 支持的语言

- auto
- chinese
- english
- french
- german
- italian
- japanese
- korean
- portuguese
- russian
- spanish

## 项目结构

```
qwen3-tts/
├── .gitignore          # Git 忽略文件
├── .python-version    # Python 版本配置
├── README.md          # 项目说明
├── cuda_check.txt     # CUDA 检查结果
├── install.ps1        # 安装脚本
├── main.py            # 主程序
├── pyproject.toml     # 项目配置
├── test_base.py       # Base 模型测试脚本
├── test_custom_voice.py  # CustomVoice 测试脚本
└── test_local.py      # 本地模型测试脚本
```

## 环境配置

### 1. 安装依赖

```powershell
uv venv
uv pip install -r requirements.txt
```

### 2. 下载模型

使用 ModelScope 下载模型到本地：

```powershell
modelscope download qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir D:\model\Qwen3-TTS-12Hz-0.6B-Base
```

### 3. 设置环境变量

```powershell
$env:HF_HOME = "D:\download\huggingface"
$env:TRANSFORMERS_CACHE = "D:\download\huggingface\hub"
$env:MODELSCOPE_CACHE = "D:\download\modelscope"
```

## 运行测试

```powershell
cd D:\workspace\tts\qwen3-tts
.\.venv\Scripts\python.exe test_local.py
```

测试结果将保存到 `test_local_result.txt`，生成的音频文件为 `test_output.wav`。

## 技术要点

1. **模型加载**: 使用 `Qwen3TTSModel.from_pretrained()` 加载本地模型
2. **语言参数**: 使用正确的语言参数 `chinese` 而非 `zh`
3. **音频播放**: 使用 `winsound` 模块直接播放音频
4. **API 使用**: 使用模型的原生 `generate` 方法

## 注意事项

- 模型必须先下载到 `D:\model` 目录
- 测试脚本会自动跳过参考音频下载，使用模型内置功能
- 如果遇到内存不足错误，请使用更小的模型或增加虚拟内存
