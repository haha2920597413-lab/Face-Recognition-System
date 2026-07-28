环境配置：
conda create -n face_gpu python=3.10 -y
conda activate face_gpu
python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install onnxruntime-gpu

pip install -r requirements-gpu.txt

conda activate face_gpu

1. 人脸裁剪
python face_crop_align.py
2. 数据增强
python augment_identities.py
3. 特征库构建
python build_gallery.py. 每次都保存
python build_gallery_stable.py，每20轮保存一次(数据库人数多的时候更稳定)

【在infer文件夹下评估单张图像的结果：python recognize_raw_in_memory.py 
评估所有图像：python evaluate_raw_test_set.py】

4. 视频运行
python video_face_recognition.py

5. 带有语音的运行
python run.py

6. 新增人物时增加特征库
python pipeline_update_gallery.py



1.项目组成：
  raw:所有人员的文件夹，后期加人自己创建（工号创建文件夹，证件照一张对应重命名为file（工号）_name（姓名）_000.jpg放在该文件夹下）！！！

  identities\label.csv：人物对应的标签表，后期加人要在这里加！！！

  identities：所有人员经过（裁剪——数据增强）的人脸图片，数据增强1张照片变换成15张照片

  log:训练的日志文件

  weights：放配置文件，不要动

  recognition:视频/摄像头识别结果（方便后期查看）

  preprocess：底层代码

  infer下放未处理的原始图片，方便模型对单张图片进行检测

  【 face_crop_align.py：人脸面部裁剪
    augment_identities.py：数据增强，多角度多方位
    build_gallery.py/python build_gallery_stable.py:构建数据集的特征库】

  python pipeline_update_gallery.py：新增人物总流水线

  【recognize_raw_in_memory.py ：对单张图片进行识别
    evaluate_raw_test_set.py：走一遍原始图片的完整处理+评估
    这两个都是对模型的评估代码，使用原来的图片进行模型好坏的检测，一个是多张图片的评估，另一个是包含所有图片的评估】

  video_face_recognition.py：视频识别脚本,没有语音播报
  run.py：有语音播报

2：特征库维护：
  把需要增加的人的图片和标签信息填到相应的raw/工号文件夹下（对应照片重新命名）:
  【标签放在identities\label.csv中，最后新增即可，格式按照上面的格式
    图片用工号创建文件夹，证件照一张对应重命名为file_name_000.jpg（工号_名字_000）放在该文件夹下】

  运行python pipeline_update_gallery.py即可将新增人员特征加入原来的特征库中

  PS：删除不好删除，因为特征标签在特征库中三个是一一对应的，没有必要可以不用删除，一定要删除某个人的情况下，最简单的方法就是删除当前特征库，在raw，identities中删除相关人的文件夹，重新构建特征库（耗时）










