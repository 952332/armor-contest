"""
测试第七题：位姿解算
"""

import cv2
import numpy as np
import os
import json
from armor import (
    task1_image_preprocessing,
    task2_color_segmentation,
    task3_light_bar_extraction,
    task4_camera_calibration,
    task5_number_recognition,
    task6_armor_detection,
    task7_pose_estimation
)

def test_task7():
    """测试题目7的功能"""
    
    print("=" * 80)
    print("第七题：位姿解算测试")
    print("=" * 80)
    
    image_path = "test_images/armor.jpg"
    calibration_dir = "calibration_images"
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
            print("警告: 未能检测到足够的灯条")
            return
        
        # 4. 相机标定
        print("\n步骤4: 相机标定")
        if os.path.isdir(calibration_dir):
            result4 = task4_camera_calibration(
                calibration_dir, 
                img_bgr=img_bgr, 
                output_dir=output_dir,
                show_windows=False
            )
            camera_matrix = result4["camera_matrix"]
            dist_coeffs = result4["dist_coeffs"]
        else:
            print("使用默认内参矩阵")
            camera_matrix = np.array([[800, 0, 320], 
                                     [0, 800, 240], 
                                     [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))
        
        print(f"相机内参矩阵:\n{camera_matrix}")
        
        # 5. 数字识别
        print("\n步骤5: 数字识别")
        result5 = task5_number_recognition(
            img_bgr, left_bars, right_bars,
            output_dir=output_dir,
            template_dir=template_dir,
            show_windows=False
        )
        recognized_numbers = result5["recognized_numbers"]
        
        # 6. 装甲板识别
        print("\n步骤6: 装甲板识别")
        result6 = task6_armor_detection(
            img_bgr, left_bars, right_bars,
            mask_red, mask_blue,
            recognized_numbers,
            output_dir=output_dir
        )
        valid_armors = result6["valid_armors"]
        
        print(f"检测到 {len(valid_armors)} 个有效装甲板")
        
        if len(valid_armors) == 0:
            print("警告: 未能检测到有效装甲板，无法进行位姿解算")
            return
        
        # 7. 位姿解算
        print("\n步骤7: 位姿解算")
        for i, armor in enumerate(valid_armors):
            print(f"\n对装甲板 {i+1} 进行位姿解算:")
            print(f"  颜色: {armor['color']}, 数字: {armor['number']}")
            
            result7 = task7_pose_estimation(
                armor,
                camera_matrix,
                dist_coeffs,
                output_dir=output_dir
            )
            
            if result7:
                print(f"\n位姿解算成功!")
                print(f"  距离: {result7['distance_m']:.2f} m")
                print(f"  欧拉角: X={result7['euler_angles'][0]:.2f}°, "
                      f"Y={result7['euler_angles'][1]:.2f}°, "
                      f"Z={result7['euler_angles'][2]:.2f}°")
            else:
                print("  位姿解算失败")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_with_improved_detection():
    """使用改进版检测进行位姿解算"""
    
    from task6_improved import task6_armor_detection_improved
    
    print("\n" + "=" * 80)
    print("使用改进版检测进行位姿解算")
    print("=" * 80)
    
    image_path = "test_images/armor.jpg"
    calibration_dir = "calibration_images"
    output_dir = "output"
    template_dir = "digit_templates"
    
    try:
        # 预处理
        result1 = task1_image_preprocessing(image_path, output_dir)
        img_bgr = result1["original_bgr"]
        img_gray = result1["gray"]
        
        result2 = task2_color_segmentation(img_bgr, output_dir, show_windows=False)
        mask_red = result2["mask_red"]
        mask_blue = result2["mask_blue"]
        
        result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir, show_windows=False)
        left_bars = result3["left_bars"]
        right_bars = result3["right_bars"]
        
        # 相机标定
        if os.path.isdir(calibration_dir):
            result4 = task4_camera_calibration(
                calibration_dir, 
                img_bgr=img_bgr, 
                output_dir=output_dir,
                show_windows=False
            )
            camera_matrix = result4["camera_matrix"]
            dist_coeffs = result4["dist_coeffs"]
        else:
            camera_matrix = np.array([[800, 0, 320], 
                                     [0, 800, 240], 
                                     [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.zeros((4, 1))
        
        # 数字识别
        result5 = task5_number_recognition(
            img_bgr, left_bars, right_bars,
            output_dir=output_dir,
            template_dir=template_dir,
            show_windows=False
        )
        recognized_numbers = result5["recognized_numbers"]
        
        # 改进版装甲板检测
        result6 = task6_armor_detection_improved(
            img_bgr, left_bars, right_bars,
            mask_red, mask_blue,
            recognized_numbers,
            output_dir=output_dir,
            min_score=50.0,
            require_number=False,
            show_windows=False
        )
        valid_armors = result6["valid_armors"]
        
        print(f"\n改进版检测到 {len(valid_armors)} 个有效装甲板")
        
        # 位姿解算
        for i, armor in enumerate(valid_armors):
            print(f"\n装甲板 {i+1}: {armor['color']} {armor['number']}, 评分: {armor['score']:.1f}")
            
            result7 = task7_pose_estimation(
                armor,
                camera_matrix,
                dist_coeffs,
                output_dir=output_dir
            )
            
            if result7:
                print(f"  距离: {result7['distance_m']:.2f} m")
                print(f"  欧拉角: X={result7['euler_angles'][0]:.2f}°, "
                      f"Y={result7['euler_angles'][1]:.2f}°, "
                      f"Z={result7['euler_angles'][2]:.2f}°")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "improved":
        test_with_improved_detection()
    else:
        test_task7()
        test_with_improved_detection()

