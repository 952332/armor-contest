"""
第四题实现质量评估
全面分析相机标定与畸变矫正的效果和质量
"""

import cv2
import numpy as np
import os
import json
from armor import task4_camera_calibration

def evaluate_task4_quality():
    """评估第四题的实现质量"""
    
    print("=" * 80)
    print("第四题：相机标定与畸变矫正 - 实现质量评估报告")
    print("=" * 80)
    
    calibration_dir = "calibration_images"
    output_dir = "output"
    
    # 1. 检查标定图像
    print("\n【1. 标定图像质量评估】")
    print("-" * 80)
    
    if not os.path.isdir(calibration_dir):
        print(f"标定图像目录不存在: {calibration_dir}")
        return
    
    image_files = [f for f in os.listdir(calibration_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"找到 {len(image_files)} 张标定图像")
    
    # 检测每张图像的角点
    chessboard_size = (9, 6)
    successful_detections = 0
    detection_details = []
    
    for img_file in image_files:
        img_path = os.path.join(calibration_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            successful_detections += 1
            # 计算角点质量（角点数量）
            num_corners = corners.shape[0]
            expected_corners = chessboard_size[0] * chessboard_size[1]
            detection_details.append({
                'filename': img_file,
                'success': True,
                'corners': num_corners,
                'expected': expected_corners,
                'completeness': num_corners / expected_corners * 100
            })
            print(f"  [+] {img_file}: 成功 ({num_corners}/{expected_corners} 角点)")
        else:
            detection_details.append({
                'filename': img_file,
                'success': False,
                'corners': 0,
                'expected': expected_corners,
                'completeness': 0
            })
            print(f"  [-] {img_file}: 失败")
    
    success_rate = successful_detections / len(image_files) * 100 if image_files else 0
    print(f"\n检测成功率: {success_rate:.1f}% ({successful_detections}/{len(image_files)})")
    
    if successful_detections < 3:
        print("警告: 成功检测的图像数量少于3张，标定结果可能不够准确")
    
    # 2. 标定结果评估
    print("\n【2. 标定结果评估】")
    print("-" * 80)
    
    # 读取测试图像
    test_image_path = "test_images/armor.jpg"
    img_bgr = None
    if os.path.exists(test_image_path):
        img_bgr = cv2.imread(test_image_path)
    
    # 进行标定
    result = task4_camera_calibration(
        calibration_dir,
        chessboard_size=chessboard_size,
        img_bgr=img_bgr,
        output_dir=output_dir,
        show_windows=False
    )
    
    camera_matrix = result['camera_matrix']
    dist_coeffs = result['dist_coeffs']
    
    # 分析内参矩阵
    print("\n相机内参矩阵分析:")
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"  fx (焦距x): {fx:.2f}")
    print(f"  fy (焦距y): {fy:.2f}")
    print(f"  cx (主点x): {cx:.2f}")
    print(f"  cy (主点y): {cy:.2f}")
    
    # 评估内参合理性
    if img_bgr is not None:
        h, w = img_bgr.shape[:2]
        print(f"\n图像尺寸: {w}x{h}")
        
        # 焦距应该与图像尺寸相关
        # 通常焦距在图像宽度的0.5-2倍之间比较合理
        focal_ratio_x = fx / w
        focal_ratio_y = fy / h
        
        print(f"  焦距/宽度比: {focal_ratio_x:.2f}")
        print(f"  焦距/高度比: {focal_ratio_y:.2f}")
        
        if focal_ratio_x > 100 or focal_ratio_y > 100:
            print("  警告: 焦距值异常大，可能是由于模拟棋盘格没有真实物理尺寸")
        elif 0.5 < focal_ratio_x < 2.0 and 0.5 < focal_ratio_y < 2.0:
            print("  评估: 焦距值在合理范围内")
        else:
            print("  评估: 焦距值可能需要验证")
        
        # 主点应该在图像中心附近
        center_x = w / 2
        center_y = h / 2
        offset_x = abs(cx - center_x) / w
        offset_y = abs(cy - center_y) / h
        
        print(f"\n主点位置分析:")
        print(f"  图像中心: ({center_x:.1f}, {center_y:.1f})")
        print(f"  主点位置: ({cx:.1f}, {cy:.1f})")
        print(f"  偏移比例: x={offset_x*100:.1f}%, y={offset_y*100:.1f}%")
        
        if offset_x < 0.1 and offset_y < 0.1:
            print("  评估: 主点位置合理（接近图像中心）")
        elif offset_x < 0.2 and offset_y < 0.2:
            print("  评估: 主点位置基本合理")
        else:
            print("  警告: 主点位置偏移较大")
    
    # 3. 畸变系数分析
    print("\n【3. 畸变系数分析】")
    print("-" * 80)
    
    print(f"畸变系数形状: {dist_coeffs.shape}")
    
    if dist_coeffs.shape[0] >= 1:
        k1 = dist_coeffs[0, 0]
        print(f"  k1 (径向畸变1): {k1:.6f}")
        
        if abs(k1) < 0.1:
            print("    评估: 径向畸变较小，相机质量较好")
        elif abs(k1) < 0.5:
            print("    评估: 径向畸变中等")
        else:
            print("    评估: 径向畸变较大，需要矫正")
    
    if dist_coeffs.shape[0] >= 2:
        k2 = dist_coeffs[1, 0]
        print(f"  k2 (径向畸变2): {k2:.6f}")
    
    if dist_coeffs.shape[0] >= 3:
        p1 = dist_coeffs[2, 0]
        print(f"  p1 (切向畸变1): {p1:.6f}")
    
    if dist_coeffs.shape[0] >= 4:
        p2 = dist_coeffs[3, 0]
        print(f"  p2 (切向畸变2): {p2:.6f}")
    
    # 4. 畸变矫正效果评估
    print("\n【4. 畸变矫正效果评估】")
    print("-" * 80)
    
    if result.get('undistorted_image') is not None and img_bgr is not None:
        img_undistorted = result['undistorted_image']
        
        # 计算图像差异
        diff = cv2.absdiff(img_bgr, img_undistorted)
        diff_mean = np.mean(diff)
        diff_max = np.max(diff)
        
        print(f"矫正前后图像差异:")
        print(f"  平均差异: {diff_mean:.2f}")
        print(f"  最大差异: {diff_max:.2f}")
        
        if diff_mean < 5:
            print("  评估: 差异很小，可能是原始图像畸变较小，或模拟图像")
        elif diff_mean < 20:
            print("  评估: 差异中等，矫正效果可见")
        else:
            print("  评估: 差异较大，矫正效果明显")
        
        # 检查是否有明显的矫正效果
        # 对于模拟图像，可能没有明显畸变，所以差异会很小
        print("\n注意: 由于使用的是模拟棋盘格图像，可能没有明显的畸变")
        print("      实际相机拍摄的图像会有更明显的畸变和矫正效果")
    else:
        print("未进行畸变矫正（缺少测试图像）")
    
    # 5. 算法实现评估
    print("\n【5. 算法实现评估】")
    print("-" * 80)
    
    implementation_points = {
        "优点": [
            "[+] 使用OpenCV标准标定流程，实现正确",
            "[+] 支持亚像素级角点精确化（cornerSubPix）",
            "[+] 使用getOptimalNewCameraMatrix优化新相机矩阵",
            "[+] 标定结果保存为JSON格式，便于后续使用",
            "[+] 支持多张标定图像，提高标定精度"
        ],
        "可以改进": [
            "[-] 未计算重投影误差（reprojection error）",
            "[-] 未验证标定结果的准确性",
            "[-] 棋盘格尺寸假设为1单位，实际应用需要真实尺寸",
            "[-] 未提供标定质量评估指标",
            "[-] 未处理标定失败的情况（如角点检测失败）"
        ]
    }
    
    for category, points in implementation_points.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  {point}")
    
    # 6. 标定质量指标
    print("\n【6. 标定质量指标】")
    print("-" * 80)
    
    quality_metrics = {
        "标定图像数量": len(image_files),
        "成功检测数量": successful_detections,
        "检测成功率": f"{success_rate:.1f}%",
        "标定图像质量": "良好" if success_rate >= 80 else "需要改进",
        "内参合理性": "需要验证" if fx > 100000 else "基本合理",
        "畸变系数": "已计算" if dist_coeffs.shape[0] >= 1 else "未计算"
    }
    
    for metric, value in quality_metrics.items():
        print(f"  {metric}: {value}")
    
    # 7. 综合评分
    print("\n【7. 综合评分】")
    print("-" * 80)
    
    scores = {
        "算法实现": {"得分": 9, "满分": 10, "说明": "使用标准OpenCV流程，实现正确"},
        "标定质量": {"得分": 7, "满分": 10, "说明": "基本完成标定，但缺少质量评估"},
        "代码质量": {"得分": 8, "满分": 10, "说明": "代码清晰，但可以添加更多验证"},
        "实用性": {"得分": 6, "满分": 10, "说明": "使用模拟图像，实际应用需要真实数据"},
        "功能完整性": {"得分": 8, "满分": 10, "说明": "基本功能完整，缺少质量评估指标"}
    }
    
    total_score = 0
    for aspect, score_info in scores.items():
        score = score_info['得分']
        total_score += score
        print(f"{aspect}: {score}/{score_info['满分']} - {score_info['说明']}")
    
    avg_score = total_score / len(scores)
    print(f"\n平均得分: {avg_score:.1f}/10")
    print(f"总体评价: ", end="")
    
    if avg_score >= 8:
        print("优秀 - 实现质量高，可以投入使用")
    elif avg_score >= 7:
        print("良好 - 基本满足要求，建议添加质量评估")
    elif avg_score >= 6:
        print("中等 - 需要改进，建议添加验证和质量评估")
    else:
        print("需要改进 - 存在明显问题")
    
    # 8. 改进建议
    print("\n【8. 改进建议】")
    print("-" * 80)
    
    improvements = [
        {
            "优先级": "高",
            "改进点": "计算重投影误差",
            "说明": "重投影误差是评估标定质量的重要指标",
            "实现": "使用cv2.projectPoints计算重投影点，计算与检测角点的误差"
        },
        {
            "优先级": "高",
            "改进点": "添加标定质量验证",
            "说明": "验证标定结果的合理性，如焦距范围、主点位置等",
            "实现": "检查内参矩阵和畸变系数的合理性"
        },
        {
            "优先级": "中",
            "改进点": "支持真实棋盘格尺寸",
            "说明": "实际应用中需要测量棋盘格的真实尺寸",
            "实现": "添加square_size参数，在objp中乘以真实尺寸"
        },
        {
            "优先级": "中",
            "改进点": "改进错误处理",
            "说明": "处理角点检测失败、标定失败等情况",
            "实现": "添加异常处理和错误提示"
        },
        {
            "优先级": "低",
            "改进点": "可视化标定过程",
            "说明": "显示角点检测结果、重投影误差等",
            "实现": "创建可视化函数，显示标定过程和质量"
        }
    ]
    
    for imp in improvements:
        print(f"\n[{imp['优先级']}优先级] {imp['改进点']}:")
        print(f"  说明: {imp['说明']}")
        print(f"  实现: {imp['实现']}")
    
    # 保存评估报告
    report = {
        "评估时间": str(np.datetime64('now')),
        "标定图像": {
            "总数": len(image_files),
            "成功检测": successful_detections,
            "成功率": success_rate,
            "详细信息": detection_details
        },
        "标定结果": {
            "相机内参": {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy)
            },
            "畸变系数": dist_coeffs.tolist()
        },
        "质量评估": quality_metrics,
        "评分": {k: v['得分'] for k, v in scores.items()},
        "平均得分": avg_score
    }
    
    with open("output/task4_evaluation_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n评估报告已保存到: output/task4_evaluation_report.json")


if __name__ == "__main__":
    from datetime import datetime
    evaluate_task4_quality()

