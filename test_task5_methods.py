"""
测试第五题的不同识别方法
对比各种方法的效果
"""

import cv2
import numpy as np
import os
from armor import task1_image_preprocessing, task3_light_bar_extraction
from task5_improved_methods import task5_number_recognition_improved

def compare_methods():
    """对比不同识别方法的效果"""
    
    print("=" * 80)
    print("第五题：不同识别方法对比测试")
    print("=" * 80)
    
    image_path = "test_images/armor.jpg"
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    # 预处理
    result1 = task1_image_preprocessing(image_path, output_dir="output")
    img_gray = result1["gray"]
    img_bgr = result1["original_bgr"]
    
    # 灯条提取
    result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir="output", show_windows=False)
    left_bars = result3["left_bars"]
    right_bars = result3["right_bars"]
    
    print(f"\n检测到左灯条: {len(left_bars)}, 右灯条: {len(right_bars)}")
    
    if len(left_bars) == 0 or len(right_bars) == 0:
        print("警告: 未能检测到足够的灯条")
        return
    
    # 测试不同方法
    methods = ["template", "features", "contour", "combined"]
    results = {}
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"测试方法: {method}")
        print(f"{'='*80}")
        
        result = task5_number_recognition_improved(
            img_bgr, left_bars, right_bars,
            output_dir="output",
            template_dir="digit_templates",
            method=method,
            show_windows=False
        )
        
        results[method] = result
    
    # 对比结果
    print("\n" + "=" * 80)
    print("方法对比结果")
    print("=" * 80)
    
    print(f"\n{'方法':<15} {'识别数字':<12} {'置信度':<12} {'状态'}")
    print("-" * 80)
    
    for method in methods:
        result = results[method]
        if result['recognized_numbers']:
            num_info = result['recognized_numbers'][0]
            digit = num_info['digit']
            conf = num_info['confidence']
            status = "成功" if digit != "?" else "失败"
            print(f"{method:<15} {digit:<12} {conf:<12.2f} {status}")
        else:
            print(f"{method:<15} {'无结果':<12} {'0.00':<12} 失败")


def test_multiple_images():
    """使用组合方法测试多张图像"""
    
    print("\n" + "=" * 80)
    print("使用组合方法测试多张图像")
    print("=" * 80)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        print(f"\n处理: {os.path.basename(img_path)}")
        
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 灯条提取
            result3 = task3_light_bar_extraction(img_gray, img_bgr, 
                                                 output_dir=None, 
                                                 show_windows=False)
            left_bars = result3["left_bars"]
            right_bars = result3["right_bars"]
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                print(f"  跳过: 未检测到足够的灯条")
                continue
            
            # 使用组合方法识别
            result5 = task5_number_recognition_improved(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir="digit_templates",
                method="combined",
                show_windows=False
            )
            
            if result5['recognized_numbers']:
                for num_info in result5['recognized_numbers']:
                    print(f"  识别结果: 数字={num_info['digit']}, "
                          f"置信度={num_info['confidence']:.2f}, "
                          f"方法={num_info['method']}")
            else:
                print(f"  未识别到数字")
                
        except Exception as e:
            print(f"  处理失败: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_methods()
    else:
        compare_methods()
        test_multiple_images()

