"""
测试第四题改进效果
展示重投影误差计算和标定质量验证功能
"""

import cv2
import numpy as np
import os
import json
from armor import task4_camera_calibration

def test_improvements():
    """测试改进后的功能"""
    
    print("=" * 80)
    print("第四题改进效果测试")
    print("=" * 80)
    
    calibration_dir = "calibration_images"
    output_dir = "output"
    test_image_path = "test_images/armor.jpg"
    
    # 检查标定图像
    if not os.path.isdir(calibration_dir):
        print(f"错误：标定图像目录不存在: {calibration_dir}")
        return
    
    image_files = [f for f in os.listdir(calibration_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n找到 {len(image_files)} 张标定图像")
    
    # 读取测试图像
    img_bgr = None
    if os.path.exists(test_image_path):
        img_bgr = cv2.imread(test_image_path)
        print(f"测试图像: {test_image_path}")
    
    print("\n" + "=" * 80)
    print("开始标定（使用改进后的版本）")
    print("=" * 80)
    
    try:
        # 使用改进后的函数
        result = task4_camera_calibration(
            calibration_dir,
            chessboard_size=(9, 6),
            square_size=25.0,  # 假设每个方格25mm
            img_bgr=img_bgr,
            output_dir=output_dir,
            show_windows=False
        )
        
        print("\n" + "=" * 80)
        print("改进效果总结")
        print("=" * 80)
        
        # 1. 重投影误差
        if result.get('reprojection_error') is not None:
            mean_error = result['reprojection_error']
            print(f"\n【1. 重投影误差】")
            print(f"  平均误差: {mean_error:.4f} 像素")
            
            if mean_error < 0.5:
                quality = "优秀"
            elif mean_error < 1.0:
                quality = "良好"
            elif mean_error < 2.0:
                quality = "一般"
            else:
                quality = "较差"
            print(f"  质量评估: {quality}")
            print(f"  [OK] 新增功能：自动计算并评估标定质量")
        else:
            print("\n【1. 重投影误差】")
            print("  [FAIL] 未计算（可能标定失败）")
        
        # 2. 标定结果验证
        if result.get('validation') is not None:
            validation = result['validation']
            print(f"\n【2. 标定结果验证】")
            
            # 焦距验证
            focal = validation['focal_length']
            print(f"  焦距验证:")
            print(f"    fx = {focal['fx']:.2f}, fy = {focal['fy']:.2f}")
            print(f"    期望范围: {focal['expected_range'][0]:.2f} - {focal['expected_range'][1]:.2f}")
            print(f"    验证结果: {'[OK] 通过' if focal['valid'] else '[FAIL] 失败'}")
            
            # 主点验证
            principal = validation['principal_point']
            print(f"  主点验证:")
            print(f"    主点位置: ({principal['cx']:.2f}, {principal['cy']:.2f})")
            print(f"    图像中心: ({principal['image_center'][0]:.2f}, {principal['image_center'][1]:.2f})")
            print(f"    偏移量: ({principal['offset'][0]:.2f}, {principal['offset'][1]:.2f})")
            print(f"    验证结果: {'[OK] 通过' if principal['valid'] else '[FAIL] 失败'}")
            
            # 畸变系数验证
            distortion = validation['distortion']
            print(f"  畸变系数验证:")
            print(f"    系数: {[f'{c:.6f}' for c in distortion['coefficients'][:4]]}")
            print(f"    验证结果: {'[OK] 通过' if distortion['valid'] else '[FAIL] 失败'}")
            
            print(f"  [OK] 新增功能：自动验证标定结果的合理性")
        else:
            print("\n【2. 标定结果验证】")
            print("  [FAIL] 未验证（可能标定失败）")
        
        # 3. 检查保存的JSON文件
        json_path = os.path.join(output_dir, "task4_calibration.json")
        if os.path.exists(json_path):
            print(f"\n【3. 保存的标定结果】")
            print(f"  文件路径: {json_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                calib_data = json.load(f)
            
            if 'reprojection_error' in calib_data:
                print(f"  [OK] 包含重投影误差信息")
                print(f"    平均误差: {calib_data['reprojection_error']['mean_error']:.4f} 像素")
                print(f"    质量: {calib_data['reprojection_error']['quality']}")
            
            if 'validation' in calib_data:
                print(f"  [OK] 包含验证信息")
                print(f"    焦距验证: {'通过' if calib_data['validation']['focal_length']['valid'] else '失败'}")
                print(f"    主点验证: {'通过' if calib_data['validation']['principal_point']['valid'] else '失败'}")
                print(f"    畸变验证: {'通过' if calib_data['validation']['distortion']['valid'] else '失败'}")
            
            if 'square_size_mm' in calib_data:
                print(f"  [OK] 包含棋盘格尺寸信息: {calib_data['square_size_mm']} mm")
            
            if 'num_images' in calib_data:
                print(f"  [OK] 包含标定图像数量: {calib_data['num_images']}")
        
        # 4. 改进对比
        print(f"\n【4. 改进对比】")
        print("=" * 80)
        print("改进前:")
        print("  [X] 无重投影误差计算")
        print("  [X] 无标定质量评估")
        print("  [X] 无结果验证功能")
        print("  [X] 不支持真实棋盘格尺寸")
        print("  [X] 缺少错误处理")
        
        print("\n改进后:")
        print("  [OK] 自动计算重投影误差")
        print("  [OK] 质量评估（优秀/良好/一般/较差）")
        print("  [OK] 验证焦距、主点、畸变系数合理性")
        print("  [OK] 支持真实棋盘格尺寸参数")
        print("  [OK] 完善的错误处理机制")
        print("  [OK] 详细的结果保存（包含所有评估信息）")
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_improvements()

