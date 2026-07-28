Face Recognition System
一套轻量化、可落地的全离线人脸识别系统，完整实现「证件照入库 → 人脸预处理增强 → 特征库构建 → 图片/视频/摄像头实时识别 → 离线语音播报」全链路。无需联网、无需云端API，纯本地推理，适配 Windows 端门禁、考勤、课堂人脸签到、本地身份核验等场景。

---
✨ 项目特点
- 全离线本地化推理：检测、对齐、特征提取、语音播报全部本地运行，无数据上传，隐私性极高
- 多级高精度识别流水线：检测+关键点对齐+特征匹配三级架构，大幅降低侧脸、光照、模糊带来的识别误差
- 多样本鲁棒增强：单人单张原图生成15类不同场景增强样本，适配明暗、旋转、遮挡、压缩等复杂真实场景
- 智能增量更新：支持新增人员单独入库，无需重建全量特征库，迭代部署更高效
- 多终端场景适配：同时支持批量图片核验、本地视频解析、摄像头实时流识别
- 离线中文语音交互：内置 Piper TTS 离线语音合成，识别成功自动播报自定义问候语
- 全面兼容适配：原生支持 Windows 中文路径、中文文件名，解决传统开源项目乱码、读图失败问题
- 低算力优化：视频跳帧推理、结果缓存复用，兼顾识别精度与运行流畅度，普通GPU即可流畅运行

---
🧩 核心架构与实现原理
本项目采用经典人脸识别四阶段流水线，摒弃单一模型粗识别方案，通过分步精细化处理，大幅提升落地准确率，整体轻量高效、适合终端部署。
1. 人脸检测模块（YOLOv8-Face）
使用轻量化 YOLOv8s-face 专用人脸检测模型，相较于通用检测模型，对小脸、侧脸、模糊人脸、遮挡人脸适配性更强，快速定位画面中所有人脸坐标，输出精准人脸外接框，为后续对齐裁剪提供基础。
2. 关键点对齐模块（RetinaFace）
通过 RetinaFace 提取人脸五点关键特征点（双眼、鼻尖、左右嘴角），通过仿射变换将所有人脸统一矫正对齐至 112×112 标准尺寸。有效解决人脸角度偏移、俯仰、歪脸导致的特征提取偏差，是高精度识别的核心关键。
3. 特征提取模块（ArcFace/MobileFaceNet）
采用工业级 ArcFace 轻量化 ONNX 模型，将对齐后的人脸图像转换为 512维归一化特征向量。摒弃传统像素比对方式，通过高维特征表征人脸唯一性，抗干扰能力极强。
4. 多模板匹配策略
不同于单人单特征的简易方案，本项目为每人构建15组差异化特征模板。识别时通过余弦相似度计算，结合最大值/TopK均值聚合策略，综合判定身份，有效避免单一样本误识、拒识问题。
5. 离线语音播报模块（Piper TTS）
集成轻量级 Piper 离线语音合成引擎，无需联网，支持自定义播报文案、语音冷却时间，避免重复播报，实现无感智能语音提示。

---
🛠️ 技术栈
Python3.10 / PyTorch / ONNX Runtime GPU / YOLOv8-face / RetinaFace / ArcFace / OpenCV / Piper TTS / PyYAML

---
📁 项目结构
Face-Recognition-System/
├── config.yaml                 # 全局统一配置文件
├── requirements-gpu.txt        # GPU版本依赖清单
├── face_crop_align.py          # 人脸检测+关键点对齐裁剪
├── augment_identities.py       # 多场景数据增强生成
├── build_gallery.py            # 全量构建人脸特征库
├── pipeline_update_gallery.py  # 增量人员更新流水线
├── recognize_raw_in_memory.py  # 批量图片离线识别
├── video_face_recognition.py   # 视频/摄像头识别（无语音）
├── run.py                      # 完整版识别（含离线语音播报）
├── raw/                        # 原始人员证件照
├── identities/                 # 对齐+增强后的标准人脸库
├── gallery/                    # 最终人脸特征向量库
├── infer/                      # 待识别测试图片目录
├── weights/                    # 模型权重存放目录
├── log/                        # 全流程运行日志
└── recognition/                # 识别结果导出目录

---
💻 环境与安装
环境要求
- 系统：Windows10/11（完整适配，Linux可兼容运行）
- Python：3.10（固定版本，避免依赖报错）
- 运行环境：推荐 NVIDIA 独立显卡 + CUDA12.1（GPU加速），支持CPU降级推理
快速安装部署
# 克隆项目
git clone https://github.com/haha2920597413-lab/Face-Recognition-System.git
cd Face-Recognition-System

# 创建专属虚拟环境
conda create -n face_gpu python=3.10 -y
conda activate face_gpu

# 安装 CUDA12.1 版本 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装GPU推理引擎与项目依赖
pip install onnxruntime-gpu
pip install -r requirements-gpu.txt

---
🚀 快速使用教程
1. 数据准备
在 raw/ 目录下，以「工号/学号」命名新建子文件夹，每个文件夹存放单人证件照。同时在 identities/label.csv 中完善工号与姓名映射关系，完成人员信息录入。
2. 构建专属人脸特征库
依次执行预处理、增强、特征入库，生成可用于识别的人脸特征库：
python face_crop_align.py
python augment_identities.py
python build_gallery.py
3. 多场景识别推理
# 批量图片离线识别
python recognize_raw_in_memory.py

# 摄像头实时人脸识别 + 语音播报（主推荐）
python run.py

# 本地视频文件人脸识别（无语音）
python video_face_recognition.py
4. 新增人员增量更新
新增人员照片与信息后，一键流水线更新，无需重构全部特征库：
python pipeline_update_gallery.py

---
⚙️ 核心可调参数
所有超参统一在 config.yaml 中配置，按需微调适配场景：
- 识别相似度阈值：默认0.4，调高更严格、调低提升通过率
- 视频跳帧间隔：平衡识别帧率与设备算力，低配设备可适当调大
- 语音播报配置：支持自定义问候文案、播报冷却时间，防止重复播报
- 特征聚合模式：支持最大值/TopK均值两种匹配策略，适配不同人群场景
- 推理设备自动切换：优先CUDA，无GPU自动降级CPU推理

---
📌 模型权重说明
因模型文件体积较大，不纳入Git仓库上传，需自行下载后放入 weights/ 目录：
- yolov8s-face.pt：轻量化人脸检测模型，精准快速
- Resnet50_Final.pth：RetinaFace五点关键点对齐模型
- w600k_mbf.onnx：轻量化ArcFace人脸特征提取模型
- Piper TTS 资源文件：离线中文语音合成模型

---
⚠️ 部署注意事项
- 人脸原始数据、特征库属于隐私数据，请勿随意上传、分享或公开
- 首次部署必须修改 config.yaml 中项目根路径，否则读写文件报错
- 识别效果不佳时，优先调低相似度阈值、检查人脸对齐是否正常
- 大模型权重文件禁止直接推送GitHub，避免仓库超限

---
📄 开源许可证
本项目仅供学习、科研与个人二次开发使用，商用需谨慎。各基础模型遵循其原作者开源协议。
⭐ 致谢
基于 YOLOv8 / RetinaFace / InsightFace / Piper TTS 开源项目二次开发，感谢开源社区贡献。