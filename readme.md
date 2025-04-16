1. 首先运行 data_organize.ipynb 下载数据并整理
2. 训练代码 train.ipynb
3. 测试代码 demo.ipynb

# input

jpg 格式面部表情图像

# output

'Autistic', 'Non_Autistic' 二分类结果

# 代码结构

```
FaceExpr/
├── ckpt/                         # 模型检查点目录
│   ├── best_classifier.pth
│   └── final_classifier.pth
├── imgs/                         # 示例图像数据目录
│   ├── A_0001.jpg                # Autistic 示例
│   └── N_0001.jpg                # Non_Autistic 示例
├── models/                       # 模型代码目录
│   ├── classifier.py             # 分类器
│   └── feature_extractor.py      # 特征提取器(ViT)
├── data_organize.ipynb       # 数据下载和整理
├── demo.ipynb                # 测试代码
├── readme.md
├── train.ipynb               # 训练代码
└── utils.py
```

# 测试数据

示例图像在 imgs 目录下。

结果以 print 形式输出预测结果 pred，pred为0表示自闭症，为1表示不是自闭症。

# demo

```
# vit 特征提取
feature_extractor = ViTFeatureExtractor(
        model_name="vit_large_patch16_224"
    ).to(device)

# 训练好的分类器
classifier = ViTLargeClassifier(input_dim=1000).to(device)
classifier.load_state_dict(torch.load('./ckpt/best_classifier.pth'))

feature_extractor.eval()
classifier.eval()

with torch.no_grad():
    features = feature_extractor(img_tensor)
    outputs = classifier(features)

    probs = torch.softmax(outputs, dim=1)
    _, pred = torch.max(outputs, 1)
```
