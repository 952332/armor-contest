"""
第七题可视化：位姿解算
展示位姿解算结果和可视化
"""

import cv2
import numpy as np
import os
from armor import (
    task1_image_preprocessing,
    task2_color_segmentation,
    task3_light_bar_extraction,
    task4_camera_calibration,
    task5_number_recognition
)
from task6_improved import task6_armor_detection_improved
from task7_improved import task7_pose_estimation_improved

def visualize_task7(image_path: str = "test_images/armor.jpg",
                    output_dir: str = "output",
                    template_dir: str = "digit_templates",
                    calibration_dir: str = "calibration_images"):
    """
    可视化第七题的位姿解算结果
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
        template_dir: 数字模板目录
        calibration_dir: 标定图像目录
    """
    
    print("=" * 80)
    print("第七题可视化：位姿解算")
    print("=" * 80)
    
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # 5. 数字识别
        print("\n步骤5: 数字识别")
        result5 = task5_number_recognition(
            img_bgr, left_bars, right_bars,
            output_dir=output_dir,
            template_dir=template_dir,
            show_windows=False
        )
        recognized_numbers = result5["recognized_numbers"]
        
        # 6. 装甲板检测（改进版）
        print("\n步骤6: 装甲板检测")
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
            print("未检测到有效装甲板，无法进行位姿解算")
            return
        
        # 7. 位姿解算（改进版）
        print("\n步骤7: 位姿解算")
        armor = valid_armors[0]
        
        result7 = task7_pose_estimation_improved(
            armor,
            camera_matrix,
            dist_coeffs,
            output_dir=output_dir,
            visualize=True,
            img_bgr=img_bgr
        )
        
        if not result7:
            print("位姿解算失败")
            return
        
        # 创建详细可视化
        img_vis = img_bgr.copy()
        
        # 绘制装甲板边界框
        x, y, w, h = armor['bbox']
        color_bgr = (0, 0, 255) if armor['color'] == "red" else (255, 0, 0)
        cv2.rectangle(img_vis, (x, y), (x + w, y + h), color_bgr, 3)
        
        # 绘制灯条（如果存在）
        if 'left_bar' in armor and 'right_bar' in armor:
            left_bar = armor['left_bar']
            right_bar = armor['right_bar']
            
            # 左灯条
            x1, y1, x2, y2 = left_bar['line']
            cv2.line(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(img_vis, left_bar['center'], 5, (255, 0, 0), -1)
            
            # 右灯条
            x1, y1, x2, y2 = right_bar['line']
            cv2.line(img_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(img_vis, right_bar['center'], 5, (0, 0, 255), -1)
        
        # 绘制坐标轴
        armor_width_mm = 100.0
        armor_height_mm = 50.0
        
        # 定义坐标轴3D点（在装甲板中心）
        axis_length = 50  # 毫米
        axis_points_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],  # X轴（红色）
            [0, axis_length, 0],  # Y轴（绿色）
            [0, 0, -axis_length]  # Z轴（蓝色）
        ], dtype=np.float32)
        
        rvec = result7['rvec']
        tvec = result7['tvec']
        
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        
        origin = tuple(axis_points_2d[0])
        cv2.line(img_vis, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3)  # X轴（BGR红色）
        cv2.line(img_vis, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3)  # Y轴（BGR绿色）
        cv2.line(img_vis, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3)  # Z轴（BGR蓝色）
        
        # 添加标签
        cv2.putText(img_vis, "X", tuple(axis_points_2d[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img_vis, "Y", tuple(axis_points_2d[2]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img_vis, "Z", tuple(axis_points_2d[3]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 添加位姿信息
        info_text = [
            f"Armor: {armor['color']} {armor['number']}",
            f"Distance: {result7['distance_m']:.2f} m",
            f"Euler Angles:",
            f"  X: {result7['euler_angles'][0]:.2f} deg",
            f"  Y: {result7['euler_angles'][1]:.2f} deg",
            f"  Z: {result7['euler_angles'][2]:.2f} deg",
            f"Reproj Error: {result7['reprojection_error']:.4f} px",
            f"Method: {result7['method']}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(img_vis, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        # 保存可视化结果
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(output_dir, f"task7_visualization_{img_name}.jpg")
        cv2.imwrite(vis_path, img_vis)
        print(f"\n可视化结果已保存: {vis_path}")
        
        # 显示结果
        print("\n位姿解算结果:")
        print(f"  距离: {result7['distance_m']:.2f} m")
        print(f"  欧拉角: X={result7['euler_angles'][0]:.2f}°, "
              f"Y={result7['euler_angles'][1]:.2f}°, "
              f"Z={result7['euler_angles'][2]:.2f}°")
        print(f"  重投影误差: {result7['reprojection_error']:.4f} 像素")
        print(f"  使用方法: {result7['method']}")
        
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_images/armor.jpg"
    visualize_task7(image_path)

