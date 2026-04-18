# Qwen3-TTS 安装脚本
# 此脚本将完成 Qwen3-TTS 的全部安装步骤

# 设置环境变量
$env:HF_HOME = "D:\download\huggingface"
$env:TRANSFORMERS_CACHE = "D:\download\huggingface\hub"
$env:TORCH_HOME = "D:\download\torch"
$env:MODELSCOPE_CACHE = "D:\download\modelscope"

Write-Host "=== 环境变量设置完成 ===" -ForegroundColor Green
Write-Host "HF_HOME: $env:HF_HOME"
Write-Host "TORCH_HOME: $env:TORCH_HOME"

# 进入项目目录
$QWEN_DIR = "D:\workspace\tts\qwen3-tts"
Set-Location $QWEN_DIR

Write-Host "`n=== 激活虚拟环境 ===" -ForegroundColor Green
& ".\.venv\Scripts\Activate.ps1"

Write-Host "`n=== 安装 PyTorch (CUDA 12.4) ===" -ForegroundColor Green
uv pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

Write-Host "`n=== 安装 Qwen3-TTS ===" -ForegroundColor Green
uv pip install -U qwen-tts

Write-Host "`n=== 安装额外依赖 ===" -ForegroundColor Green
uv pip install streamlit soundfile librosa

Write-Host "`n=== 验证安装 ===" -ForegroundColor Green
python -c "from qwen_tts import QwenTTS; print('Qwen3-TTS 安装成功!')"

Write-Host "`n=== Qwen3-TTS 安装完成 ===" -ForegroundColor Green
