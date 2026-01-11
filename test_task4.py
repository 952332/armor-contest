"""
测试第四题：相机标定与畸变矫正
"""

import cv2
import numpy as np
import os
from armor import task4_camera_calibration

def test_task4():
    """测试题目4的功能"""
    
    print("=" * 80)
    print("第四题：相机标定与畸变矫正测试")
    print("=" * 80)
    
    calibration_dir = "calibration_images"
    output_dir = "output"
    test_image_path = "test_images/armor.jpg"
    
    # 检查标定图像目录
    if not os.path.isdir(calibration_dir):
        print(f"\n警告：标定图像目录不存在: {calibration_dir}")
        print("将使用默认内参矩阵")
    else:
        image_files = [f for f in os.listdir(calibration_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n找到 {len(image_files)} 张标定图像")
    
    # 读取测试图像（如果有）
    img_bgr = None
    if os.path.exists(test_image_path):
        img_bgr = cv2.imread(test_image_path)
        print(f"测试图像: {test_image_path}")
    else:
        print(f"测试图像不存在: {test_image_path}，将只进行标定")
    
    try:
        # 调用第四题函数（非交互模式需要修改代码）
        print("\n开始相机标定...")
        result = task4_camera_calibration(
            calibration_dir,
            chessboard_size=(9, 6),
            img_bgr=img_bgr,
            output_dir=output_dir,
            show_windows=False
        )
        
        print("\n测试完成！")
        print(f"相机内参矩阵:\n{result['camera_matrix']}")
        print(f"畸变系数:\n{result['dist_coeffs']}")
        
        if result.get('undistorted_image') is not None:
            print("畸变矫正图像已生成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_with_generated_images():
    """使用生成的标定图像进行测试"""
    
    print("\n" + "=" * 80)
    print("使用生成的标定图像进行测试")
    print("=" * 80)
    
    # 生成标定图像
    from generate_calibration_images import generate_calibration_dataset
    print("\n步骤1: 生成标定图像")
    generate_calibration_dataset("calibration_images", num_images=5)
    
    # 进行标定
    print("\n步骤2: 进行相机标定")
    test_image_path = "test_images/armor.jpg"
    img_bgr = None
    if os.path.exists(test_image_path):
        img_bgr = cv2.imread(test_image_path)
    
    result = task4_camera_calibration(
        "calibration_images",
        chessboard_size=(9, 6),
        img_bgr=img_bgr,
        output_dir="output",
        show_windows=False
    )
    
    print("\n标定结果:")
    print(f"相机内参矩阵:\n{result['camera_matrix']}")
    print(f"畸变系数:\n{result['dist_coeffs']}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        test_with_generated_images()
    else:
        test_task4()

