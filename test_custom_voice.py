"""
Qwen3-TTS Custom Voice 测试 - 测试所有中文 Speaker
"""
import os
import sys
import gc
import torch

# 设置环境变量
os.environ["HF_HOME"] = "D:\\download\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\download\\huggingface\\hub"
os.environ["MODELSCOPE_CACHE"] = "D:\\download\\modelscope"

log = open("custom_voice_test.txt", "w", encoding="utf-8")
def p(msg):
    print(msg)
    log.write(msg + "\n")
    log.flush()

p("=" * 60)
p("Qwen3-TTS Custom Voice 测试")
p("=" * 60)

# 1. 清理内存
p("\n1. 清理内存...")
gc.collect()
torch.cuda.empty_cache()
p("   内存清理完成")

# 2. 检查 CUDA
p(f"\n2. CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p(f"   GPU: {torch.cuda.get_device_name(0)}")
    p(f"   显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 3. 导入
p("\n3. 导入模块...")
try:
    from qwen_tts import Qwen3TTSModel
    p("   ✅ 导入成功!")
except Exception as e:
    p(f"   ❌ 导入失败: {e}")
    sys.exit(1)

# 4. 检查模型
p("\n4. 检查模型...")

# 模型信息
model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
local_model_name = "Qwen3-TTS-12Hz-0.6B-CustomVoice"
local_model_path = r"D:\model\Qwen3-TTS-12Hz-0.6B-CustomVoice"

# 检查本地模型
if os.path.exists(local_model_path):
    p(f"   ✅ 模型已找到: {local_model_path}")
    model_path = local_model_path
else:
    p(f"   ❌ 模型未找到: {local_model_path}")
    p("   ")
    p("   ⚠️  请先下载模型到 D:\model 目录")
    p("   推荐使用 ModelScope 下载:")
    p(f"   modelscope download qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir D:\model\Qwen3-TTS-12Hz-1.7B-CustomVoice")
    p("   ")
    p("   或使用镜像站加速:")
    p(f"   HF_ENDPOINT=https://hf-mirror.com modelscope download qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir D:\model\Qwen3-TTS-12Hz-1.7B-CustomVoice")
    log.close()
    sys.exit(1)

# 5. 加载模型
p("\n5. 加载模型...")
try:
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        dtype=torch.float32,
    )
    p("   ✅ 模型加载成功!")
except Exception as e:
    p(f"   ❌ 模型加载失败: {e}")
    import traceback
    p(traceback.format_exc())
    log.close()
    sys.exit(1)

# 6. 获取所有支持的 speaker
p("\n6. 获取支持的 Speaker...")

# 中文 Speaker 列表（基于 Qwen3-TTS 官方文档）
chinese_speakers = [
    {"name": "Vivian", "description": "明亮，略带锋芒的年轻女声", "language": "Chinese"},
    {"name": "Serena", "description": "温暖，温柔的年轻女声", "language": "Chinese"},
    {"name": "Uncle_Fu", "description": "成熟的男声，低沉醇厚", "language": "Chinese"},
    {"name": "Dylan", "description": "年轻的北京男声，音色清晰自然", "language": "Chinese (Beijing Dialect)"},
    {"name": "Eric", "description": "活泼的成都男声，略带沙哑的明亮", "language": "Chinese (Sichuan Dialect)"},
]

p(f"   中文 Speaker 数量: {len(chinese_speakers)}")
for speaker in chinese_speakers:
    p(f"   - {speaker['name']}: {speaker['description']} ({speaker['language']})")

# 7. 测试每个中文 Speaker
p("\n7. 测试所有中文 Speaker...")
test_text = "你好,欢迎使用 Qwen3-TTS 语音合成系统。这是测试语音。"
p(f"   测试文本: {test_text}")

for speaker in chinese_speakers:
    speaker_name = speaker["name"]
    p(f"\n   测试 Speaker: {speaker_name}")
    p(f"   描述: {speaker['description']}")
    
    try:
        # 生成语音
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        result = model.generate_custom_voice(
            text=test_text,
            speaker=speaker_name,
            instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        )
        
        end_time.record()
        torch.cuda.synchronize()
        gen_time = start_time.elapsed_time(end_time) / 1000
        
        # 调试：检查返回的数据格式
        p(f"   调试：result 类型 = {type(result)}")
        if isinstance(result, tuple):
            p(f"   调试：tuple 长度 = {len(result)}")
            p(f"   调试：第一个元素类型 = {type(result[0])}")
            if hasattr(result[0], 'shape'):
                p(f"   调试：第一个元素 shape = {result[0].shape}")
            if hasattr(result[0], 'dtype'):
                p(f"   调试：第一个元素 dtype = {result[0].dtype}")
        
        # 处理返回结果
        if isinstance(result, tuple) and len(result) == 2:
            wavs, sr = result
            p(f"   ✅ 语音生成成功!")
            p(f"   耗时: {gen_time:.2f} 秒")
            p(f"   采样率: {sr} Hz")
            
            # 确保 wavs 是 numpy 数组
            import numpy as np
            if not isinstance(wavs, np.ndarray):
                if hasattr(wavs, 'cpu'):
                    wavs = wavs.cpu().numpy()
                elif isinstance(wavs, list) and len(wavs) > 0:
                    wavs = np.array(wavs)
            
            audio_length = len(wavs) / sr if sr > 0 else 0
            p(f"   音频长度: {audio_length:.2f} 秒")
            p(f"   音频 shape: {wavs.shape if hasattr(wavs, 'shape') else 'N/A'}")
            
            # 保存文件
            import soundfile as sf
            output_file = f"test_{speaker_name}.wav"
            
            # 如果 wavs 是多维的，只取第一维
            if hasattr(wavs, 'ndim') and wavs.ndim > 1:
                wavs = wavs[0]
            
            sf.write(output_file, wavs, sr)
            p(f"   保存到: {os.path.abspath(output_file)}")

            # 验证文件
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                p(f"   文件大小: {file_size / 1024:.1f} KB")
            else:
                p(f"   ❌ 文件保存失败!")
            
            # 播放音频
            p("   正在播放音频...")
            try:
                if os.name == "nt":  # Windows
                    import winsound
                    winsound.PlaySound(output_file, winsound.SND_FILENAME)
                    p("   ✅ 音频播放完成!")
                else:
                    import subprocess
                    if os.uname().sysname == "Darwin":
                        subprocess.run(["afplay", output_file])
                    else:
                        subprocess.run(["aplay", output_file])
                    p("   ✅ 音频播放完成!")
            except Exception as e:
                p(f"   ⚠️  播放失败: {e}")
                p("   ✅ 音频文件已生成，可手动播放")
        else:
            p(f"   ❌ 返回格式错误")
            
    except Exception as e:
        p(f"   ❌ 生成失败: {e}")
        import traceback
        p(traceback.format_exc())

# 8. 测试完成
p("\n" + "=" * 60)
p("测试完成!")
p("所有中文 Speaker 测试结果已保存到 custom_voice_test.txt")
p("=" * 60)
log.close()
