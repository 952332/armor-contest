"""
生成数字模板（0-9）
用于第五题的数字识别
"""

import cv2
import numpy as np
import os

def generate_digit_templates(output_dir: str = "digit_templates", 
                            template_size: tuple = (64, 64)):
    """
    生成0-9的数字模板
    
    参数:
        output_dir: 输出目录
        template_size: 模板尺寸 (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    width, height = template_size
    
    print(f"生成数字模板 (尺寸: {width}x{height})...")
    
    for digit in range(10):
        # 创建白色背景
        template = np.ones((height, width), dtype=np.uint8) * 255
        
        # 设置字体参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 5
        color = 0  # 黑色
        
        # 获取文字大小
        text = str(digit)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # 计算文字位置（居中）
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # 绘制数字
        cv2.putText(template, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # 保存模板
        filename = f"digit_{digit}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, template)
        print(f"  已生成: {filename}")
    
    print(f"\n完成！共生成10个数字模板，保存在: {output_dir}")


if __name__ == "__main__":
    generate_digit_templates()

