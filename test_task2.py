"""
测试第二题：颜色识别与阈值分割
"""

from armor import task1_image_preprocessing, task2_color_segmentation
import cv2
import os
import numpy as np

def test_task2():
    """测试题目2的功能"""
    # 测试图像路径
    image_path = "test_images/armor.jpg"
    output_dir = "output"
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误：测试图像不存在: {image_path}")
        print("请先运行 generate_test_image.py 生成测试图像")
        return
    
    print("开始测试题目2：颜色识别与阈值分割")
    print("-" * 60)
    
    try:
        # 方法1：直接读取图像并测试
        print("\n方法1：直接读取图像测试")
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"错误：无法读取图像: {image_path}")
            return
        
        # 调用第二题函数（非交互模式）
        result2 = task2_color_segmentation(img_bgr, output_dir, show_windows=False)
        
        print("\n测试成功！")
        print(f"返回结果包含以下键: {list(result2.keys())}")
        print(f"HSV图像尺寸: {result2['hsv'].shape}")
        print(f"红色掩码尺寸: {result2['mask_red'].shape}")
        print(f"蓝色掩码尺寸: {result2['mask_blue'].shape}")
        
        # 统计掩码中的像素数量
        red_pixels = np.sum(result2['mask_red'] > 0)
        blue_pixels = np.sum(result2['mask_blue'] > 0)
        total_pixels = result2['mask_red'].size
        
        print(f"\n颜色统计:")
        print(f"红色像素数: {red_pixels} ({red_pixels/total_pixels*100:.2f}%)")
        print(f"蓝色像素数: {blue_pixels} ({blue_pixels/total_pixels*100:.2f}%)")
        
        # 方法2：使用第一题的结果
        print("\n" + "="*60)
        print("方法2：使用第一题的结果进行测试")
        print("="*60)
        result1 = task1_image_preprocessing(image_path, output_dir)
        result2_from_task1 = task2_color_segmentation(result1["original_bgr"], output_dir, show_windows=False)
        
        print("\n两种方法的结果对比:")
        print(f"红色掩码像素数: {np.sum(result2_from_task1['mask_red'] > 0)}")
        print(f"蓝色掩码像素数: {np.sum(result2_from_task1['mask_blue'] > 0)}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_multiple_images():
    """测试多张不同场景的图像"""
    print("\n" + "="*60)
    print("测试多张不同场景的图像")
    print("="*60)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n处理图像: {os.path.basename(img_path)}")
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                try:
                    result = task2_color_segmentation(img_bgr, output_dir="output", show_windows=False)
                    red_pixels = np.sum(result['mask_red'] > 0)
                    blue_pixels = np.sum(result['mask_blue'] > 0)
                    print(f"  红色像素: {red_pixels}, 蓝色像素: {blue_pixels}")
                except Exception as e:
                    print(f"  处理失败: {e}")
        else:
            print(f"图像不存在: {img_path}")


if __name__ == "__main__":
    # 测试单张图像
    test_task2()
    
    # 测试多张图像
    test_multiple_images()

