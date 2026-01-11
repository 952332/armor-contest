"""
可视化第三题的结果
显示灯条提取的效果
"""

import cv2
import numpy as np
import os
from armor import task1_image_preprocessing, task3_light_bar_extraction

def visualize_task3_results(image_path: str = "test_images/armor.jpg"):
    """可视化第三题的结果"""
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    # 读取图像并预处理
    result1 = task1_image_preprocessing(image_path, output_dir="output")
    img_gray = result1["gray"]
    img_bgr = result1["original_bgr"]
    
    # 执行灯条提取
    result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir="output", show_windows=False)
    
    # 创建组合图像用于显示
    h, w = img_bgr.shape[:2]
    
    # 将边缘检测结果转换为3通道以便显示
    edges_3ch = cv2.cvtColor(result3['edges'], cv2.COLOR_GRAY2BGR)
    
    # 创建一个显示所有检测直线的图像
    img_with_all_lines = img_bgr.copy()
    for line_info in result3['valid_lines']:
        x1, y1, x2, y2 = line_info['line']
        cv2.line(img_with_all_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 标记中心点
        cv2.circle(img_with_all_lines, line_info['center'], 3, (255, 0, 0), -1)
    
    # 创建2x2的网格显示
    # 第一行：原始图像、边缘检测
    # 第二行：检测到的所有直线、灯条区域（带标注）
    row1 = np.hstack([img_bgr, edges_3ch])
    row2 = np.hstack([img_with_all_lines, result3['result_image']])
    
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
        "Original", "Canny Edges",
        "Detected Lines", "Light Bars"
    ]
    
    label_w = new_w // 2
    for i, label in enumerate(labels):
        x = (i % 2) * label_w + 10
        y = (i // 2) * new_h1 + 25
        
        # 添加背景矩形使文字更清晰
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(combined, (x-5, y-text_size[1]-5), 
                     (x+text_size[0]+5, y+5), bg_color, -1)
        cv2.putText(combined, label, (x, y), font, font_scale, color, thickness)
    
    # 在图像底部添加统计信息
    stats = result3
    stats_y = combined.shape[0] - 30
    stats_text = (f"Valid Lines: {len(stats['valid_lines'])} | "
                 f"Left Bars: {len(stats['left_bars'])} | "
                 f"Right Bars: {len(stats['right_bars'])}")
    text_size = cv2.getTextSize(stats_text, font, 0.5, 1)[0]
    stats_x = (combined.shape[1] - text_size[0]) // 2
    cv2.rectangle(combined, (stats_x-10, stats_y-text_size[1]-5), 
                 (stats_x+text_size[0]+10, stats_y+5), bg_color, -1)
    cv2.putText(combined, stats_text, (stats_x, stats_y), font, 0.5, color, 1)
    
    # 显示结果
    cv2.imshow("Task 3: Light Bar Extraction Results", combined)
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存组合图像
    cv2.imwrite("output/task3_visualization.jpg", combined)
    print("可视化结果已保存到: output/task3_visualization.jpg")
    
    # 打印详细信息
    print(f"\n检测统计:")
    print(f"  有效灯条数: {len(stats['valid_lines'])}")
    print(f"  左灯条数: {len(stats['left_bars'])}")
    print(f"  右灯条数: {len(stats['right_bars'])}")
    
    if stats['valid_lines']:
        print(f"\n前5条灯条信息:")
        for i, line_info in enumerate(stats['valid_lines'][:5]):
            print(f"  灯条 {i+1}: 长度={line_info['length']:.2f}px, "
                  f"角度={line_info['angle']:.2f}°, "
                  f"中心=({line_info['center'][0]}, {line_info['center'][1]})")


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/armor.jpg"
    visualize_task3_results(image_path)

