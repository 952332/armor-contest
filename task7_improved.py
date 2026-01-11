"""
第七题改进版本：位姿解算
主要改进：
1. 使用灯条中心点进行更精确的点对应
2. 改进3D模型定义（使用灯条位置）
3. 添加重投影误差验证
4. 改进距离计算
5. 添加位姿可视化
"""

import cv2
import numpy as np
import os
import json
from typing import Dict, Tuple, Optional

def task7_pose_estimation_improved(
    armor_info: Dict, 
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray, 
    output_dir: str = "output",
    armor_width_mm: float = 100.0,
    armor_height_mm: float = 50.0,
    visualize: bool = False,
    img_bgr: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    第七题改进版：位姿解算
    
    主要改进：
    1. 使用灯条中心点进行更精确的点对应
    2. 改进3D模型定义
    3. 添加重投影误差验证
    4. 改进距离计算
    5. 添加位姿可视化
    
    参数:
        armor_info: 装甲板信息（包含left_bar, right_bar, bbox等）
        camera_matrix: 相机内参矩阵
        dist_coeffs: 畸变系数
        output_dir: 输出目录
        armor_width_mm: 装甲板宽度（毫米）
        armor_height_mm: 装甲板高度（毫米）
        visualize: 是否可视化
        img_bgr: 原始图像（用于可视化）
    """
    print("=" * 60)
    print("题目7改进版：位姿解算")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 方法1：使用装甲板四个角点（原方法）
    bbox = armor_info['bbox']
    x, y, w, h = bbox
    
    # 定义装甲板四个角点的3D坐标（世界坐标系，z=0）
    # 以装甲板中心为原点
    object_points_3d_corners = np.array([
        [-armor_width_mm / 2, -armor_height_mm / 2, 0],  # 左上
        [armor_width_mm / 2, -armor_height_mm / 2, 0],  # 右上
        [armor_width_mm / 2, armor_height_mm / 2, 0],  # 右下
        [-armor_width_mm / 2, armor_height_mm / 2, 0]  # 左下
    ], dtype=np.float32)
    
    # 装甲板四个角点的2D图像坐标
    image_points_2d_corners = np.array([
        [x, y],  # 左上
        [x + w, y],  # 右上
        [x + w, y + h],  # 右下
        [x, y + h]  # 左下
    ], dtype=np.float32)
    
    # 方法2：使用灯条中心点（更精确）
    use_light_bars = False
    object_points_3d_bars = None
    image_points_2d_bars = None
    
    if 'left_bar' in armor_info and 'right_bar' in armor_info:
        left_bar = armor_info['left_bar']
        right_bar = armor_info['right_bar']
        
        # 定义灯条中心点的3D坐标
        # 假设灯条在装甲板左右两侧，距离中心各为armor_width_mm/2
        left_bar_3d = np.array([-armor_width_mm / 2, 0, 0], dtype=np.float32)
        right_bar_3d = np.array([armor_width_mm / 2, 0, 0], dtype=np.float32)
        
        # 获取灯条中心点的2D坐标
        left_bar_2d = np.array(left_bar['center'], dtype=np.float32)
        right_bar_2d = np.array(right_bar['center'], dtype=np.float32)
        
        # 使用四个点：两个灯条中心 + 装甲板上下两个角点
        object_points_3d_bars = np.array([
            left_bar_3d,  # 左灯条中心
            right_bar_3d,  # 右灯条中心
            [armor_width_mm / 2, -armor_height_mm / 2, 0],  # 右上角
            [-armor_width_mm / 2, armor_height_mm / 2, 0]  # 左下角
        ], dtype=np.float32)
        
        image_points_2d_bars = np.array([
            left_bar_2d,
            right_bar_2d,
            [x + w, y],  # 右上角
            [x, y + h]  # 左下角
        ], dtype=np.float32)
        
        use_light_bars = True
    
    # 选择使用哪种方法
    if use_light_bars:
        object_points_3d = object_points_3d_bars
        image_points_2d = image_points_2d_bars
        method_name = "灯条中心点方法"
    else:
        object_points_3d = object_points_3d_corners
        image_points_2d = image_points_2d_corners
        method_name = "角点方法"
    
    print(f"使用方法: {method_name}")
    print(f"3D点数量: {len(object_points_3d)}")
    
    # PnP位姿解算
    success, rvec, tvec = cv2.solvePnP(
        object_points_3d,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        print("警告：位姿解算失败")
        return {}

    # 旋转向量转旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 计算重投影误差（验证解算质量）
    projected_points, _ = cv2.projectPoints(
        object_points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected_points = projected_points.reshape(-1, 2)
    
    reprojection_error = np.mean(np.linalg.norm(image_points_2d - projected_points, axis=1))
    
    print(f"重投影误差: {reprojection_error:.4f} 像素")
    
    if reprojection_error > 5.0:
        print("警告：重投影误差较大，解算结果可能不准确")

    # 计算距离
    # 距离 = 平移向量的z分量（相机坐标系）
    distance_mm = abs(tvec[2, 0])  # z方向距离
    distance_cm = distance_mm / 10
    distance_m = distance_mm / 1000
    
    # 如果距离异常大，可能是单位问题，尝试使用另一种方法
    if distance_m > 100:  # 如果距离超过100米，可能有问题
        # 使用平移向量的模长
        distance_mm = np.linalg.norm(tvec)
        distance_cm = distance_mm / 10
        distance_m = distance_mm / 1000
        print("警告：使用平移向量模长计算距离（可能单位不匹配）")

    # 计算欧拉角
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x_angle = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x_angle = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_angle = np.arctan2(-rotation_matrix[2, 0], sy)
        z_angle = 0

    euler_angles = np.array([x_angle, y_angle, z_angle]) * 180 / np.pi

    print("位姿解算结果:")
    print(f"  旋转向量 (rvec): {rvec.flatten()}")
    print(f"  平移向量 (tvec): {tvec.flatten()}")
    print(f"  距离: {distance_mm:.2f} mm ({distance_cm:.2f} cm, {distance_m:.2f} m)")
    print(f"  欧拉角 (度): X={euler_angles[0]:.2f}, Y={euler_angles[1]:.2f}, Z={euler_angles[2]:.2f}")
    print(f"  重投影误差: {reprojection_error:.4f} 像素")

    # 可视化（如果提供图像）
    if visualize and img_bgr is not None:
        img_vis = img_bgr.copy()
        
        # 绘制2D点
        for pt in image_points_2d:
            cv2.circle(img_vis, tuple(pt.astype(int)), 5, (0, 255, 0), -1)
        
        # 绘制重投影点
        for pt in projected_points:
            cv2.circle(img_vis, tuple(pt.astype(int)), 3, (255, 0, 0), -1)
        
        # 绘制坐标轴
        axis_length = 50  # 像素
        axis_points_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],  # X轴（红色）
            [0, axis_length, 0],  # Y轴（绿色）
            [0, 0, -axis_length]  # Z轴（蓝色）
        ], dtype=np.float32)
        
        axis_points_2d, _ = cv2.projectPoints(
            axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs
        )
        axis_points_2d = axis_points_2d.reshape(-1, 2).astype(int)
        
        origin = tuple(axis_points_2d[0])
        cv2.line(img_vis, origin, tuple(axis_points_2d[1]), (0, 0, 255), 3)  # X轴
        cv2.line(img_vis, origin, tuple(axis_points_2d[2]), (0, 255, 0), 3)  # Y轴
        cv2.line(img_vis, origin, tuple(axis_points_2d[3]), (255, 0, 0), 3)  # Z轴
        
        # 添加文本信息
        info_text = [
            f"Distance: {distance_m:.2f}m",
            f"Euler: X={euler_angles[0]:.1f} Y={euler_angles[1]:.1f} Z={euler_angles[2]:.1f}",
            f"Reproj Error: {reprojection_error:.2f}px"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(img_vis, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        vis_path = os.path.join(output_dir, "task7_pose_visualization.jpg")
        cv2.imwrite(vis_path, img_vis)
        print(f"可视化结果已保存: {vis_path}")

    # 保存结果
    pose_data = {
        "rotation_vector": rvec.tolist(),
        "translation_vector": tvec.tolist(),
        "rotation_matrix": rotation_matrix.tolist(),
        "distance_mm": float(distance_mm),
        "distance_cm": float(distance_cm),
        "distance_m": float(distance_m),
        "euler_angles_deg": euler_angles.tolist(),
        "reprojection_error": float(reprojection_error),
        "method": method_name,
        "armor_size_mm": {
            "width": armor_width_mm,
            "height": armor_height_mm
        }
    }

    with open(os.path.join(output_dir, "task7_pose_improved.json"), 'w') as f:
        json.dump(pose_data, f, indent=2)

    return {
        "rvec": rvec,
        "tvec": tvec,
        "rotation_matrix": rotation_matrix,
        "distance_mm": distance_mm,
        "distance_cm": distance_cm,
        "distance_m": distance_m,
        "euler_angles": euler_angles,
        "reprojection_error": reprojection_error,
        "method": method_name
    }

