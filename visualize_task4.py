"""
可视化第四题的结果
展示相机标定和畸变矫正的效果
"""

import cv2
import numpy as np
import os
import json
from armor import task4_camera_calibration

def visualize_task4_results(calibration_dir: str = "calibration_images",
                           test_image_path: str = "test_images/armor.jpg"):
    """可视化第四题的结果"""
    
    print("=" * 80)
    print("第四题：相机标定与畸变矫正 - 可视化")
    print("=" * 80)
    
    # 检查标定图像
    if not os.path.isdir(calibration_dir):
        print(f"标定图像目录不存在: {calibration_dir}")
        return
    
    # 读取测试图像
    img_bgr = None
    if os.path.exists(test_image_path):
        img_bgr = cv2.imread(test_image_path)
        print(f"测试图像: {test_image_path}")
    else:
        print(f"测试图像不存在: {test_image_path}")
        print("将只进行标定，不进行畸变矫正")
    
    # 进行标定
    result = task4_camera_calibration(
        calibration_dir,
        chessboard_size=(9, 6),
        img_bgr=img_bgr,
        output_dir="output",
        show_windows=False
    )
    
    # 显示标定结果
    print("\n" + "=" * 80)
    print("标定结果")
    print("=" * 80)
    
    camera_matrix = result['camera_matrix']
    dist_coeffs = result['dist_coeffs']
    
    print(f"\n相机内参矩阵:")
    print(f"  fx (焦距x): {camera_matrix[0, 0]:.2f}")
    print(f"  fy (焦距y): {camera_matrix[1, 1]:.2f}")
    print(f"  cx (主点x): {camera_matrix[0, 2]:.2f}")
    print(f"  cy (主点y): {camera_matrix[1, 2]:.2f}")
    
    print(f"\n畸变系数:")
    print(f"  形状: {dist_coeffs.shape}")
    if dist_coeffs.shape[0] >= 1:
        print(f"  k1 (径向畸变1): {dist_coeffs[0, 0]:.6f}")
    if dist_coeffs.shape[0] >= 2:
        print(f"  k2 (径向畸变2): {dist_coeffs[1, 0]:.6f}")
    if dist_coeffs.shape[0] >= 3:
        print(f"  p1 (切向畸变1): {dist_coeffs[2, 0]:.6f}")
    if dist_coeffs.shape[0] >= 4:
        print(f"  p2 (切向畸变2): {dist_coeffs[3, 0]:.6f}")
    if dist_coeffs.shape[0] >= 5:
        print(f"  k3 (径向畸变3): {dist_coeffs[4, 0]:.6f}")
    
    # 如果有畸变矫正图像，创建可视化
    if result.get('undistorted_image') is not None and img_bgr is not None:
        img_undistorted = result['undistorted_image']
        
        # 创建对比图
        h, w = img_bgr.shape[:2]
        comparison = np.hstack([img_bgr, img_undistorted])
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)
        
        cv2.putText(comparison, "Original (Distorted)", (10, 30),
                   font, font_scale, color, thickness)
        cv2.putText(comparison, "Undistorted", (w + 10, 30),
                   font, font_scale, color, thickness)
        
        # 保存对比图
        output_path = "output/task4_visualization.jpg"
        cv2.imwrite(output_path, comparison)
        print(f"\n对比图已保存: {output_path}")
        
        # 显示（如果需要）
        print("\n显示对比图（按任意键关闭）...")
        cv2.imshow("Distortion Correction Comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 检查标定结果文件
    calibration_file = "output/task4_calibration.json"
    if os.path.exists(calibration_file):
        print(f"\n标定结果已保存: {calibration_file}")
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
            print(f"  相机内参矩阵形状: {len(calib_data['camera_matrix'])}x{len(calib_data['camera_matrix'][0])}")
            print(f"  畸变系数数量: {len(calib_data['dist_coeffs'])}")


def analyze_calibration_quality(calibration_dir: str = "calibration_images"):
    """分析标定质量"""
    
    print("\n" + "=" * 80)
    print("标定质量分析")
    print("=" * 80)
    
    if not os.path.isdir(calibration_dir):
        print(f"标定图像目录不存在: {calibration_dir}")
        return
    
    # 统计标定图像
    image_files = [f for f in os.listdir(calibration_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\n找到 {len(image_files)} 张标定图像")
    
    # 检测每张图像的角点
    chessboard_size = (9, 6)
    successful_detections = 0
    
    for img_file in image_files:
        img_path = os.path.join(calibration_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            successful_detections += 1
            print(f"  [+] {img_file}: 成功检测到角点")
        else:
            print(f"  [-] {img_file}: 未能检测到角点")
    
    print(f"\n成功检测: {successful_detections}/{len(image_files)} 张图像")
    print(f"成功率: {successful_detections/len(image_files)*100:.1f}%")
    
    if successful_detections < 3:
        print("\n警告: 成功检测的图像数量少于3张，标定结果可能不够准确")
        print("建议: 至少需要3-5张不同角度的标定图像")


if __name__ == "__main__":
    import sys
    
    calibration_dir = sys.argv[1] if len(sys.argv) > 1 else "calibration_images"
    test_image = sys.argv[2] if len(sys.argv) > 2 else "test_images/armor.jpg"
    
    # 分析标定质量
    analyze_calibration_quality(calibration_dir)
    
    # 可视化结果
    visualize_task4_results(calibration_dir, test_image)

