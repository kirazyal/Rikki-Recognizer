# Taki Recognizer

椎名立希（来自《BanG Dream! It's MyGO!!!!!》）图片识别系统。

## 功能

- 通过网页上传图片，识别是否为椎名立希
- 使用深度学习模型进行图像分类
- 显示识别结果的置信度

## 在线使用

访问部署的网页即可使用。

## 本地运行

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动网页服务

```bash
cd web_app
python app.py
```

然后打开浏览器访问 `http://localhost:5000`

### 训练模型

```bash
cd scripts
python train.py
```

训练数据放在 `data/raw/` 目录下：
- `official/` - 官方立希图片
- `taki/` - 所有立希图片（包含同人）

## 技术栈

- Python
- PyTorch
- Flask
- OpenCV

## 许可证

MIT
