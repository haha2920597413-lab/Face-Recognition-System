# 与 biubug6/Pytorch_Retinaface 一致；pretrain 在加载 .pth 时应为 False，避免再找 mobilenetV1X0.25_pretrain.tar

CFG_MNET = {
    "name": "mobilenet0.25",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 32,
    "ngpu": 1,
    "epoch": 250,
    "decay1": 190,
    "decay2": 220,
    "image_size": 640,
    "pretrain": False,
    # torchvision>=0.13 要求 value 为 str
    "return_layers": {"stage1": "feat1", "stage2": "feat2", "stage3": "feat3"},
    "in_channel": 32,
    "out_channel": 64,
}

CFG_RE50 = {
    "name": "Resnet50",
    "min_sizes": [[16, 32], [64, 128], [256, 512]],
    "steps": [8, 16, 32],
    "variance": [0.1, 0.2],
    "clip": False,
    "loc_weight": 2.0,
    "gpu_train": True,
    "batch_size": 24,
    "ngpu": 4,
    "epoch": 100,
    "decay1": 70,
    "decay2": 90,
    "image_size": 840,
    "pretrain": False,
    "return_layers": {"layer2": "feat1", "layer3": "feat2", "layer4": "feat3"},
    "in_channel": 256,
    "out_channel": 256,
}
