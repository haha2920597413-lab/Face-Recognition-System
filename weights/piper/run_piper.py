import subprocess

text = "王楠老师您好!"

cmd = [
    r"D:\Liang\Pcode\Face Recognition System\weights\piper\piper.exe",
    "--model", r"D:\Liang\Pcode\Face Recognition System\weights\piper\zh_CN-huayan-medium.onnx",
    "--output_file", r"D:\Liang\Pcode\Face Recognition System\weights\piper\test.wav"
]

subprocess.run(
    cmd,
    input=text.encode("utf-8"),
    check=True
)

print("生成完成：piper\\test.wav")