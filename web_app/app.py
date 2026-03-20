import os
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify
import sys
import random
import glob

# 添加scripts路径以便导入模型
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.model import TakiClassifier

app = Flask(__name__)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制16MB

# 创建上传文件夹（如果不存在）
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 加载模型（全局加载一次，避免每次请求都加载）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TakiClassifier(num_classes=2)
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'models', 'taki_final.pth')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()
print(f"Model loaded successfully, using device: {device}")

# 预处理函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_with_image(image):
    """对PIL Image对象进行预测"""
    # 确保是 RGB 模式
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 预处理
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 预测
    # 预测
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probabilities).item()

        # 如果是立希（标签1），置信度用"是立希"的概率
        # 如果不是立希（标签0），置信度用"非立希"的概率
        if pred_class == 1:  # 是立希
            confidence = probabilities[1].item()
        else:  # 不是立希
            confidence = probabilities[0].item()


    print(f"[OK] 预测完成: 类别={pred_class}, 置信度={confidence:.2%}")

    # 结果解释：只有预测为立希（标签1）且置信度足够高才判定为立希
    is_taki = (pred_class == 1 and confidence >= 0.5)

    return {
        'is_taki': is_taki,
        'confidence': float(confidence),
        'prob_not_taki': float(probabilities[0].item()),
        'prob_is_taki': float(probabilities[1].item()),
        'uncertain': confidence < 0.5  # 标记是否不确定
    }


def predict_image(image_path):
    """对单张图片进行预测 - 最强王者版"""
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import io

        print(f"📷 尝试读取图片: {image_path}")

        # 方法1: OpenCV
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(img_cv)
            print("[OK] OpenCV 读取成功")
            return predict_with_image(image)

        # 方法2: PIL 直接打开
        try:
            image = Image.open(image_path)
            print("[OK] PIL 直接打开成功")
            return predict_with_image(image)
        except:
            pass

        # 方法3: 二进制读取
        print("⚠️ 尝试二进制读取...")
        with open(image_path, 'rb') as f:
            img_data = f.read()
            print(f"📊 文件大小: {len(img_data)} 字节")

        # 方法4: 尝试用不同格式打开
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        for ext in ['jpeg', 'png', 'bmp', 'gif', 'tiff']:
            try:
                image = Image.open(io.BytesIO(img_data))
                print(f"[OK] BytesIO + {ext} 成功")
                return predict_with_image(image)
            except:
                continue

        # 方法5: 如果都不行，尝试用 cv2 解码内存数据
        try:
            img_array = np.frombuffer(img_data, np.uint8)
            img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_cv is not None:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_cv)
                print("[OK] cv2.imdecode 成功")
                return predict_with_image(image)
        except:
            pass

        raise Exception("所有图片读取方法都失败了")

    except Exception as e:
        print(f"[ERROR] 预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


@app.route('/', methods=['GET'])
def index():
    """首页：上传表单"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """处理上传的图片并返回预测结果"""

    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式，请上传jpg/png图片'}), 400

    try:
        # 保存上传的文件
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 进行预测
        result = predict_image(filepath)

        # 删除临时文件
        os.remove(filepath)

        # 返回结果
        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        return jsonify({
            'success': True,
            'filename': file.filename,
            'is_taki': result['is_taki'],
            'confidence': f"{result['confidence']:.2%}",
            'probabilities': {
                '非立希': result['prob_not_taki'],  # 直接返回数值
                '是立希': result['prob_is_taki']     # 直接返回数值
            },
            'message': '是伟大的紫瞳黑长直鼓手椎名立希！' if result['is_taki'] else '不是椎名立希哦',
            'praise': '立希漂亮漂亮漂亮' if result['is_taki'] else '是另一个可爱的女孩子哦'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/random-taki', methods=['GET'])
def random_taki():
    """返回一张随机的立希图片（从展示图库中选）"""
    import glob

    # 展示图库路径
    gallery_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'static', 'gallery')

    all_images = []

    if os.path.exists(gallery_folder):
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
            images = glob.glob(os.path.join(gallery_folder, ext))
            all_images.extend(images)

    if not all_images:
        return jsonify({'error': '展示图库中没有图片', 'total_images': 0}), 404

    random_image = random.choice(all_images)

    return jsonify({
        'success': True,
        'image_url': f'/static/gallery/{os.path.basename(random_image)}',
        'filename': os.path.basename(random_image),
        'total_images': len(all_images)
    })


if __name__ == '__main__':
    # 启动Flask开发服务器
    # Render 使用 PORT 环境变量
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)