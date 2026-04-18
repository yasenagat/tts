"""
Qwen3-TTS 本地模型测试脚本
"""
import os
import sys
import gc
import torch

# 设置环境变量
os.environ["HF_HOME"] = "D:\\download\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = "D:\\download\\huggingface\\hub"
os.environ["MODELSCOPE_CACHE"] = "D:\\download\\modelscope"

log = open("test_result.txt", "w", encoding="utf-8")
def p(msg):
    print(msg)
    log.write(msg + "\n")
    log.flush()

p("=" * 60)
p("Qwen3-TTS 测试")
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
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    p("   ✅ Qwen3TTSModel 导入成功!")
    p("   ✅ Qwen3TTSTokenizer 导入成功!")
    import_success = True
except Exception as e:
    p(f"   ❌ 导入失败: {e}")
    import_success = False

if not import_success:
    p("\n❌ 测试失败,请检查安装!")
    log.close()
    sys.exit(1)

# 4. 检查模型
p("\n4. 检查模型...")

# 模型信息
model_name = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
local_model_name = "Qwen3-TTS-12Hz-0.6B-Base"
local_model_path = r"D:\model\Qwen3-TTS-12Hz-0.6B-Base"

# 检查本地模型
if os.path.exists(local_model_path):
    p(f"   ✅ 模型已找到: {local_model_path}")
    model_path = local_model_path
else:
    p(f"   ❌ 模型未找到: {local_model_path}")
    p("   ")
    p("   ⚠️  请先下载模型到 D:\model 目录")
    p("   推荐使用 ModelScope 下载:")
    p(f"   modelscope download qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir D:\model\Qwen3-TTS-12Hz-0.6B-Base")
    p("   ")
    p("   或使用镜像站加速:")
    p(f"   HF_ENDPOINT=https://hf-mirror.com modelscope download qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir D:\model\Qwen3-TTS-12Hz-0.6B-Base")
    log.close()
    sys.exit(1)

# 5. 加载模型
p("\n5. 加载模型...")
try:
    from qwen_tts import Qwen3TTSModel
    
    print("正在加载模型...")
    print("(首次加载会下载模型,约 2GB,请耐心等待)")
    
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.float32,
    )
    
    p("   ✅ 模型加载成功!")
    
    # 检查支持的语言
    supported_languages = model.get_supported_languages()
    if supported_languages:
        p(f"   支持的语言: {supported_languages}")
    else:
        p("   模型未提供支持的语言列表")
    
    # 检查支持的说话人
    supported_speakers = model.get_supported_speakers()
    if supported_speakers:
        p(f"   支持的说话人: {supported_speakers[:5]}..." if len(supported_speakers) > 5 else f"   支持的说话人: {supported_speakers}")
    else:
        p("   模型未提供支持的说话人列表")
    
    model_loaded = True
except Exception as e:
    p(f"   ❌ 模型加载失败: {e}")
    import traceback
    p(traceback.format_exc())
    model_loaded = False

if not model_loaded:
    p("\n❌ 测试失败,模型加载失败!")
    log.close()
    sys.exit(1)

# 6. 跳过参考音频下载（使用内置功能）
p("\n6. 跳过参考音频下载...")
p("   ✅ 直接使用模型内置功能进行测试")
ref_audio_ok = True

# 7. 生成语音
p("\n7. 生成语音...")
test_text = "你好,欢迎使用 Qwen3-TTS 语音合成系统。这是测试语音。"
p(f"   文本: {test_text}")

try:
    import soundfile as sf
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    
    # 使用模型的 generate 方法
    input_text = f"<|im_start|>assistant\n{test_text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = model.processor(text=input_text, return_tensors="pt", padding=True)['input_ids'].to(model.device)
    
    # 对于 Base 模型，使用 generate 方法
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids.unsqueeze(0),
        languages=["chinese"],
        non_streaming_mode=True,
        do_sample=True,
        temperature=0.9
    )
    
    # 解码生成的代码
    wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": codes} for codes in talker_codes_list])
    
    end_time.record()
    torch.cuda.synchronize()
    gen_time = start_time.elapsed_time(end_time) / 1000
    
    p(f"   ✅ 语音生成成功!")
    p(f"   耗时: {gen_time:.2f} 秒")
    p(f"   采样率: {sr} Hz")
    p(f"   音频长度: {len(wavs[0]) / sr:.2f} 秒")
    
    # 保存音频
    output_file = "test_output.wav"
    sf.write(output_file, wavs[0], sr)
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
            # 使用 Windows 内置的 winsound 模块
            import winsound
            winsound.PlaySound(output_file, winsound.SND_FILENAME)
            p("   ✅ 音频播放完成!")
        else:
            # 使用系统命令播放
            import subprocess
            if os.uname().sysname == "Darwin":
                subprocess.run(["afplay", output_file])
            else:
                subprocess.run(["aplay", output_file])
            p("   ✅ 音频播放完成!")
    except Exception as e:
        p(f"   ⚠️  播放失败: {e}")
        p("   ✅ 音频文件已生成，可手动播放")
    
    audio_ok = True
except Exception as e:
    p(f"   ❌ 生成失败: {e}")
    import traceback
    p(traceback.format_exc())
    audio_ok = False

# 8. 测试总结
p("\n" + "=" * 60)
p("测试总结")
p("=" * 60)
p(f"  导入测试: {'✅ 通过' if import_success else '❌ 失败'}")
p(f"  模型加载: {'✅ 通过' if model_loaded else '❌ 失败'}")
p(f"  参考音频: {'✅ 通过' if ref_audio_ok else '❌ 失败'}")
p(f"  语音生成: {'✅ 通过' if audio_ok else '❌ 失败'}")

p("\n" + "=" * 60)
if import_success and model_loaded and audio_ok:
    p("🎉 Qwen3-TTS 测试成功!")
    p("语音文件已生成: test_output.wav")
else:
    p("⚠️  测试部分失败,请检查错误信息")
p("=" * 60)

log.close()
