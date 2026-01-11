"""
测试第六题：装甲板轮廓匹配与识别
"""

import cv2
import numpy as np
import os
from armor import (
    task1_image_preprocessing,
    task2_color_segmentation,
    task3_light_bar_extraction,
    task5_number_recognition,
    task6_armor_detection
)

def test_task6():
    """测试题目6的功能"""
    
    print("=" * 80)
    print("第六题：装甲板轮廓匹配与识别测试")
    print("=" * 80)
    
    image_path = "test_images/armor.jpg"
    output_dir = "output"
    template_dir = "digit_templates"
    
    # 检查文件
    if not os.path.exists(image_path):
        print(f"错误：测试图像不存在: {image_path}")
        return
    
    try:
        # 1. 图像预处理
        print("\n步骤1: 图像预处理")
        result1 = task1_image_preprocessing(image_path, output_dir)
        img_bgr = result1["original_bgr"]
        img_gray = result1["gray"]
        
        # 2. 颜色分割
        print("\n步骤2: 颜色分割")
        result2 = task2_color_segmentation(img_bgr, output_dir, show_windows=False)
        mask_red = result2["mask_red"]
        mask_blue = result2["mask_blue"]
        
        # 3. 灯条提取
        print("\n步骤3: 灯条提取")
        result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir, show_windows=False)
        left_bars = result3["left_bars"]
        right_bars = result3["right_bars"]
        
        print(f"检测到左灯条: {len(left_bars)}, 右灯条: {len(right_bars)}")
        
        if len(left_bars) == 0 or len(right_bars) == 0:
            print("警告: 未能检测到足够的灯条，装甲板识别可能失败")
        
        # 4. 数字识别
        print("\n步骤4: 数字识别")
        result5 = task5_number_recognition(
            img_bgr, 
            left_bars, 
            right_bars,
            output_dir=output_dir,
            template_dir=template_dir,
            show_windows=False
        )
        recognized_numbers = result5["recognized_numbers"]
        
        print(f"识别到 {len(recognized_numbers)} 个数字区域")
        for num_info in recognized_numbers:
            print(f"  数字: {num_info['digit']}, 置信度: {num_info['confidence']:.2f}")
        
        # 5. 装甲板识别
        print("\n步骤5: 装甲板识别")
        result6 = task6_armor_detection(
            img_bgr,
            left_bars,
            right_bars,
            mask_red,
            mask_blue,
            recognized_numbers,
            output_dir=output_dir
        )
        
        print("\n测试完成！")
        print(f"检测结果:")
        print(f"  检测到的有效装甲板数: {len(result6['valid_armors'])}")
        
        for i, armor in enumerate(result6['valid_armors']):
            print(f"  装甲板 {i + 1}:")
            print(f"    颜色: {armor['color']}")
            print(f"    数字: {armor['number']}")
            print(f"    数字置信度: {armor['number_confidence']:.2f}")
            print(f"    位置: {armor['bbox']}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_images():
    """测试多张图像的装甲板识别"""
    
    print("\n" + "=" * 80)
    print("多图像测试")
    print("=" * 80)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    template_dir = "digit_templates"
    output_dir = "output"
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        print(f"\n处理: {os.path.basename(img_path)}")
        
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 颜色分割
            result2 = task2_color_segmentation(img_bgr, output_dir=None, show_windows=False)
            mask_red = result2["mask_red"]
            mask_blue = result2["mask_blue"]
            
            # 灯条提取
            result3 = task3_light_bar_extraction(img_gray, img_bgr, 
                                                 output_dir=None, 
                                                 show_windows=False)
            left_bars = result3["left_bars"]
            right_bars = result3["right_bars"]
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                print(f"  跳过: 未检测到足够的灯条 (L:{len(left_bars)} R:{len(right_bars)})")
                continue
            
            # 数字识别
            result5 = task5_number_recognition(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir=template_dir,
                show_windows=False
            )
            recognized_numbers = result5["recognized_numbers"]
            
            # 装甲板识别
            result6 = task6_armor_detection(
                img_bgr, left_bars, right_bars,
                mask_red, mask_blue,
                recognized_numbers,
                output_dir=output_dir
            )
            
            if result6['valid_armors']:
                for armor in result6['valid_armors']:
                    print(f"  检测结果: {armor['color']} {armor['number']}, "
                          f"置信度={armor['number_confidence']:.2f}")
            else:
                print(f"  未检测到有效装甲板")
                
        except Exception as e:
            print(f"  处理失败: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "multiple":
        test_multiple_images()
    else:
        test_task6()
        test_multiple_images()

