"""
生成测试图像 - 模拟机甲大师装甲板
用于测试图像预处理和识别功能
支持生成多种场景的测试图像
"""

import cv2
import numpy as np
import os
import random

def draw_armor_plate(img, x, y, width, height, color_bgr, number, 
                     brightness=1.0, angle=0, blur_level=0):
    """
    在图像上绘制一个装甲板
    
    参数:
        img: 图像数组
        x, y: 装甲板左上角坐标
        width, height: 装甲板宽度和高度
        color_bgr: 装甲板颜色 (B, G, R)
        number: 装甲板上的数字 (0-9)
        brightness: 亮度系数 (0.5-1.5)
        angle: 旋转角度（度）
        blur_level: 模糊级别 (0-5)
    """
    # 创建临时图像用于绘制
    temp_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 应用亮度
    base_color = np.array(color_bgr, dtype=np.float32) * brightness
    base_color = np.clip(base_color, 0, 255).astype(np.uint8)
    
    # 绘制装甲板主体（矩形）
    cv2.rectangle(temp_img, (0, 0), (width, height), base_color.tolist(), -1)
    
    # 绘制左灯条（垂直，更亮的颜色）
    bar_width = 12
    bar_margin = 15
    left_bar_x = bar_margin
    left_bar_y1 = 10
    left_bar_y2 = height - 10
    
    # 灯条颜色更亮
    bar_color = np.array(base_color, dtype=np.float32) * 1.3
    bar_color = np.clip(bar_color, 0, 255).astype(np.uint8)
    cv2.line(temp_img, (left_bar_x, left_bar_y1), 
             (left_bar_x, left_bar_y2), bar_color.tolist(), bar_width)
    
    # 绘制右灯条
    right_bar_x = width - bar_margin
    cv2.line(temp_img, (right_bar_x, left_bar_y1), 
             (right_bar_x, left_bar_y2), bar_color.tolist(), bar_width)
    
    # 绘制数字区域（中间，白色背景）
    number_margin = 5
    number_x = left_bar_x + bar_width + number_margin
    number_y = 15
    number_w = right_bar_x - number_x - number_margin
    number_h = height - 30
    
    if number_w > 10 and number_h > 10:
        cv2.rectangle(temp_img, (number_x, number_y), 
                      (number_x + number_w, number_y + number_h), 
                      (255, 255, 255), -1)
        
        # 绘制数字
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(number)
        font_scale = min(number_w / 40, number_h / 50, 1.5)
        thickness = max(1, int(font_scale * 2))
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = number_x + (number_w - text_size[0]) // 2
        text_y = number_y + (number_h + text_size[1]) // 2
        cv2.putText(temp_img, text, (text_x, text_y), 
                   font, font_scale, (0, 0, 0), thickness)
    
    # 应用旋转（如果需要）
    if angle != 0:
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        temp_img = cv2.warpAffine(temp_img, M, (width, height), 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(50, 50, 50))
    
    # 应用模糊（如果需要）
    if blur_level > 0:
        kernel_size = blur_level * 2 + 1
        temp_img = cv2.GaussianBlur(temp_img, (kernel_size, kernel_size), 0)
    
    # 将临时图像复制到主图像
    y_end = min(y + height, img.shape[0])
    x_end = min(x + width, img.shape[1])
    temp_h = y_end - y
    temp_w = x_end - x
    
    if temp_h > 0 and temp_w > 0:
        img[y:y_end, x:x_end] = temp_img[:temp_h, :temp_w]
    
    return img


def create_test_armor_image(output_path: str = "test_images/armor.jpg", 
                           width: int = 640, height: int = 480,
                           scenario: str = "normal"):
    """
    创建一个模拟的装甲板测试图像
    
    参数:
        output_path: 输出路径
        width, height: 图像尺寸
        scenario: 场景类型
            - "normal": 正常场景（红色和蓝色装甲板）
            - "bright": 明亮场景
            - "dark": 暗场景
            - "multiple": 多个装甲板
            - "angled": 倾斜的装甲板
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # 根据场景设置背景
    if scenario == "dark":
        bg_color = 20
        brightness = 0.7
    elif scenario == "bright":
        bg_color = 200
        brightness = 1.3
    else:
        bg_color = 50
        brightness = 1.0
    
    # 创建背景
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img.fill(bg_color)
    
    # 添加一些随机噪声（模拟真实场景）
    noise_level = 15 if scenario == "dark" else 30
    noise = np.random.randint(0, noise_level, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    # 根据场景生成不同的装甲板
    if scenario == "multiple":
        # 生成多个装甲板
        positions = [
            (100, 100, 0, 0, 1),    # 红色，数字0
            (400, 100, 1, 0, 2),    # 蓝色，数字1
            (100, 300, 0, 0, 3),    # 红色，数字2
            (400, 300, 1, 0, 4),    # 蓝色，数字3
        ]
        for x, y, color_idx, angle, num in positions:
            color = (0, 0, 255) if color_idx == 0 else (255, 0, 0)  # 红色或蓝色
            draw_armor_plate(img, x, y, 180, 100, color, num, brightness, angle)
    
    elif scenario == "angled":
        # 生成倾斜的装甲板
        draw_armor_plate(img, 150, 150, 200, 120, (0, 0, 255), 1, brightness, 15)
        draw_armor_plate(img, 400, 200, 200, 120, (255, 0, 0), 2, brightness, -10)
    
    else:
        # 正常场景：一个红色和一个蓝色装甲板
        draw_armor_plate(img, 150, 150, 200, 120, (0, 0, 255), 1, brightness)
        draw_armor_plate(img, 400, 150, 200, 120, (255, 0, 0), 2, brightness)
    
    # 添加轻微的整体模糊模拟真实拍摄效果
    if scenario != "angled":
        img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"测试图像已生成: {output_path}")
    print(f"场景: {scenario}, 图像尺寸: {width}x{height}")
    
    return img


def generate_dataset(output_dir: str = "test_images", num_images: int = 10):
    """
    生成一组测试图像数据集
    
    参数:
        output_dir: 输出目录
        num_images: 生成图像数量
    """
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = ["normal", "bright", "dark", "multiple", "angled"]
    
    print(f"开始生成 {num_images} 张测试图像...")
    
    for i in range(num_images):
        # 随机选择场景
        scenario = random.choice(scenarios)
        
        # 随机尺寸变化
        width = random.randint(480, 800)
        height = random.randint(360, 600)
        
        # 生成文件名
        filename = f"armor_{i+1:03d}_{scenario}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        # 生成图像
        create_test_armor_image(output_path, width, height, scenario)
    
    print(f"\n完成！共生成 {num_images} 张测试图像，保存在: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        # 生成数据集
        num_images = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        generate_dataset("test_images", num_images)
    else:
        # 生成单张测试图像
        test_image_path = "test_images/armor.jpg"
        scenario = sys.argv[1] if len(sys.argv) > 1 else "normal"
        create_test_armor_image(test_image_path, scenario=scenario)
        
        # 显示生成的图像
        img = cv2.imread(test_image_path)
        if img is not None:
            cv2.imshow("生成的测试图像", img)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("无法读取生成的图像")

