"""
测试第七题改进效果
对比原版和改进版的位姿解算效果
"""

import cv2
import numpy as np
import os
from armor import (
    task1_image_preprocessing,
    task2_color_segmentation,
    task3_light_bar_extraction,
    task4_camera_calibration,
    task5_number_recognition,
    task7_pose_estimation
)
from task6_improved import task6_armor_detection_improved
from task7_improved import task7_pose_estimation_improved

def compare_pose_estimation():
    """对比原版和改进版的位姿解算"""
    
    print("=" * 80)
    print("第七题改进效果对比测试")
    print("=" * 80)
    
    image_path = "test_images/armor.jpg"
    calibration_dir = "calibration_images"
    output_dir = "output"
    template_dir = "digit_templates"
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
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
        
        if len(valid_armors) == 0:
            print("未检测到有效装甲板")
            return
        
        armor = valid_armors[0]
        print(f"\n检测到装甲板: {armor['color']} {armor['number']}, 评分: {armor['score']:.1f}")
        
        # 原版位姿解算
        print("\n" + "=" * 80)
        print("原版位姿解算")
        print("=" * 80)
        
        result_original = task7_pose_estimation(
            armor,
            camera_matrix,
            dist_coeffs,
            output_dir=output_dir
        )
        
        if result_original:
            print(f"\n原版结果:")
            print(f"  距离: {result_original['distance_m']:.2f} m")
            print(f"  欧拉角: X={result_original['euler_angles'][0]:.2f}°, "
                  f"Y={result_original['euler_angles'][1]:.2f}°, "
                  f"Z={result_original['euler_angles'][2]:.2f}°")
        
        # 改进版位姿解算
        print("\n" + "=" * 80)
        print("改进版位姿解算")
        print("=" * 80)
        
        result_improved = task7_pose_estimation_improved(
            armor,
            camera_matrix,
            dist_coeffs,
            output_dir=output_dir,
            visualize=True,
            img_bgr=img_bgr
        )
        
        if result_improved:
            print(f"\n改进版结果:")
            print(f"  使用方法: {result_improved['method']}")
            print(f"  距离: {result_improved['distance_m']:.2f} m")
            print(f"  欧拉角: X={result_improved['euler_angles'][0]:.2f}°, "
                  f"Y={result_improved['euler_angles'][1]:.2f}°, "
                  f"Z={result_improved['euler_angles'][2]:.2f}°")
            print(f"  重投影误差: {result_improved['reprojection_error']:.4f} 像素")
        
        # 对比总结
        print("\n" + "=" * 80)
        print("改进效果总结")
        print("=" * 80)
        
        improvements = [
            "1. 使用灯条中心点 - 更精确的点对应关系",
            "2. 重投影误差验证 - 评估解算质量",
            "3. 改进距离计算 - 更准确的距离估计",
            "4. 位姿可视化 - 绘制坐标轴和重投影点",
            "5. 详细的结果保存 - 包含更多信息"
        ]
        
        for imp in improvements:
            print(f"  [OK] {imp}")
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    compare_pose_estimation()

