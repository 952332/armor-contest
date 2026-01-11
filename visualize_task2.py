"""
可视化第二题的结果
显示颜色分割的效果
"""

import cv2
import numpy as np
import os
from armor import task2_color_segmentation

def visualize_task2_results(image_path: str = "test_images/armor.jpg"):
    """可视化第二题的结果"""
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    # 读取图像
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 执行颜色分割
    result = task2_color_segmentation(img_bgr, output_dir="output", show_windows=False)
    
    # 创建组合图像用于显示
    h, w = img_bgr.shape[:2]
    
    # 将掩码转换为3通道以便显示
    mask_red_3ch = cv2.cvtColor(result['mask_red'], cv2.COLOR_GRAY2BGR)
    mask_blue_3ch = cv2.cvtColor(result['mask_blue'], cv2.COLOR_GRAY2BGR)
    
    # 创建2x3的网格显示
    # 第一行：原始图像、HSV图像、红色掩码
    # 第二行：蓝色掩码、红色区域、蓝色区域
    row1 = np.hstack([img_bgr, result['hsv'], mask_red_3ch])
    row2 = np.hstack([mask_blue_3ch, result['red_region'], result['blue_region']])
    
    # 调整大小以便显示
    scale = min(800 / row1.shape[1], 600 / (row1.shape[0] + row2.shape[0]))
    new_w = int(row1.shape[1] * scale)
    new_h1 = int(row1.shape[0] * scale)
    new_h2 = int(row2.shape[0] * scale)
    
    row1_resized = cv2.resize(row1, (new_w, new_h1))
    row2_resized = cv2.resize(row2, (new_w, new_h2))
    
    combined = np.vstack([row1_resized, row2_resized])
    
    # 添加标签和统计信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    labels = [
        "Original", "HSV", "Red Mask",
        "Blue Mask", "Red Region", "Blue Region"
    ]
    
    # 计算统计信息
    total_pixels = result['mask_red'].size
    red_pixels = np.sum(result['mask_red'] > 0)
    blue_pixels = np.sum(result['mask_blue'] > 0)
    red_percent = red_pixels / total_pixels * 100
    blue_percent = blue_pixels / total_pixels * 100
    
    label_w = new_w // 3
    for i, label in enumerate(labels):
        x = (i % 3) * label_w + 10
        y = (i // 3) * new_h1 + 25
        
        # 添加背景矩形使文字更清晰
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(combined, (x-5, y-text_size[1]-5), 
                     (x+text_size[0]+5, y+5), bg_color, -1)
        cv2.putText(combined, label, (x, y), font, font_scale, color, thickness)
    
    # 在图像底部添加统计信息
    stats_y = combined.shape[0] - 30
    stats_text = f"Red: {red_pixels} pixels ({red_percent:.2f}%) | Blue: {blue_pixels} pixels ({blue_percent:.2f}%)"
    text_size = cv2.getTextSize(stats_text, font, 0.5, 1)[0]
    stats_x = (combined.shape[1] - text_size[0]) // 2
    cv2.rectangle(combined, (stats_x-10, stats_y-text_size[1]-5), 
                 (stats_x+text_size[0]+10, stats_y+5), bg_color, -1)
    cv2.putText(combined, stats_text, (stats_x, stats_y), font, 0.5, color, 1)
    
    # 显示结果
    cv2.imshow("Task 2: Color Segmentation Results", combined)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存组合图像
    cv2.imwrite("output/task2_visualization.jpg", combined)
    print("可视化结果已保存到: output/task2_visualization.jpg")


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/armor.jpg"
    visualize_task2_results(image_path)

