import torch
from torchvision import transforms
from PIL import Image
from model import TakiClassifier
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import cv2  # 👈 添加这一行

console = Console()  # 只用 Rich 显示文字


def predict_single_image(image_path, model_path='../models/taki_final.pth'):
    """预测单张图片（纯文字版）"""

    # 检查文件是否存在
    if not os.path.exists(image_path):
        console.print(f"❌ [bold red]图片不存在: {image_path}[/bold red]")
        return

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TakiClassifier(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ✨✨✨ 用 OpenCV 读取图片（替换原来的读取代码）✨✨✨
    try:
        # 用 OpenCV 读取
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise Exception("OpenCV 也无法读取")

        # 转换颜色空间 BGR -> RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # 转换为 PIL Image（因为 transform 需要 PIL 格式）
        original_image = Image.fromarray(img_cv)
        console.print(f"✅ 用 OpenCV 成功读取图片")

    except Exception as e:
        console.print(f"[dim]OpenCV 读取失败: {e}[/dim]")
        console.print(f"[dim]尝试用 PIL 强制读取...[/dim]")

        # 回退到原来的 PIL 强制读取
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        try:
            # 先尝试正常打开
            original_image = Image.open(image_path)
        except Exception as e2:
            print(f"正常打开失败，尝试强制打开: {e2}")
            # 如果失败，用二进制模式读取后强制解码
            with open(image_path, 'rb') as f:
                img_data = f.read()
            from io import BytesIO
            original_image = Image.open(BytesIO(img_data))

        original_image = original_image.convert('RGB')

    # 预测（后面的代码完全不变）
    image_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[pred_class].item()

    # 清屏并显示标题
    console.clear()
    console.rule("[bold yellow]椎名立希识别结果[/bold yellow]")
    console.print()

    # 只显示图片信息，不显示图片本身
    console.print(f"📷 [dim]图片: {os.path.basename(image_path)} ({original_image.width}x{original_image.height})[/dim]")
    console.print()

    # 创建结果表格
    table = Table(show_header=False, box=None, width=80)
    table.add_column("属性", style="cyan", width=15)
    table.add_column("值", style="green", width=65)

    table.add_row("文件名", os.path.basename(image_path))
    table.add_row("图片尺寸", f"{original_image.width} x {original_image.height}")

    # 判断结果
    if pred_class == 0 or pred_class == 1:
        result_text = Text(f"✅ 是伟大的紫瞳黑长直鼓手椎名立希！ (置信度: {confidence:.2%})", style="bold green")
        table.add_row("识别结果", result_text)
        table.add_row("评价", "立希漂亮漂亮漂亮 ✨")
    else:
        result_text = Text(f"❌ 不是椎名立希 (置信度: {confidence:.2%})", style="bold red")
        table.add_row("识别结果", result_text)
        table.add_row("评价", "是另一个可爱的女孩子哦 🌸")

    # 详细概率
    table.add_row("详细概率",
                  f"是立希: {probabilities[0].item():.2%} | 不是立希: {probabilities[1].item():.2%}")

    # 显示结果
    console.print(Panel(table, title="识别详情", border_style="bright_yellow"))
    console.rule("[dim]Powered by TakiRecognizer[/dim]")


if __name__ == "__main__":
    test_image = input("请输入图片路径: ").strip()
    if test_image:
        predict_single_image(test_image)