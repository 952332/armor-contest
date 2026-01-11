"""
可视化第五题的结果
展示数字区域提取和识别的效果
"""

import cv2
import numpy as np
import os
from armor import task1_image_preprocessing, task3_light_bar_extraction, task5_number_recognition

def visualize_task5_results(image_path: str = "test_images/armor.jpg"):
    """可视化第五题的结果"""
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    print("=" * 80)
    print("第五题：装甲板数字识别 - 可视化")
    print("=" * 80)
    
    # 1. 图像预处理
    result1 = task1_image_preprocessing(image_path, output_dir="output")
    img_gray = result1["gray"]
    img_bgr = result1["original_bgr"]
    
    # 2. 灯条提取
    result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir="output", show_windows=False)
    left_bars = result3["left_bars"]
    right_bars = result3["right_bars"]
    
    print(f"\n检测到左灯条: {len(left_bars)}, 右灯条: {len(right_bars)}")
    
    if len(left_bars) == 0 or len(right_bars) == 0:
        print("警告: 未能检测到足够的灯条")
        return
    
    # 3. 数字识别
    result5 = task5_number_recognition(
        img_bgr, left_bars, right_bars,
        output_dir="output",
        template_dir="digit_templates",
        show_windows=False
    )
    
    # 4. 创建可视化
    img_result = result5['result_image']
    
    # 如果有识别结果，显示数字区域
    if result5['recognized_numbers']:
        print(f"\n识别结果:")
        for i, num_info in enumerate(result5['recognized_numbers']):
            print(f"  数字区域 {i+1}:")
            print(f"    数字: {num_info['digit']}")
            print(f"    置信度: {num_info['confidence']:.2f}")
            print(f"    位置: {num_info['bbox']}")
            
            # 提取数字区域用于显示
            x, y, w, h = num_info['bbox']
            roi = img_bgr[y:y+h, x:x+w]
            
            if roi.size > 0:
                # 预处理数字区域
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                roi_resized = cv2.resize(roi_binary, (64, 64))
                
                # 创建对比显示
                roi_display = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
                
                # 添加标签
                cv2.putText(roi_display, f"Digit: {num_info['digit']}", (5, 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(roi_display, f"Conf: {num_info['confidence']:.2f}", (5, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        print("\n未识别到数字")
    
    # 保存结果
    output_path = "output/task5_visualization.jpg"
    cv2.imwrite(output_path, img_result)
    print(f"\n可视化结果已保存: {output_path}")
    
    # 显示结果
    print("\n显示结果（按任意键关闭）...")
    cv2.imshow("Task 5: Number Recognition", img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/armor.jpg"
    visualize_task5_results(image_path)

