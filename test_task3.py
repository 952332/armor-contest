"""
测试第三题：装甲板灯条提取
"""

from armor import task1_image_preprocessing, task3_light_bar_extraction
import cv2
import os
import numpy as np

def test_task3():
    """测试题目3的功能"""
    # 测试图像路径
    image_path = "test_images/armor.jpg"
    output_dir = "output"
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误：测试图像不存在: {image_path}")
        print("请先运行 generate_test_image.py 生成测试图像")
        return
    
    print("开始测试题目3：装甲板灯条提取")
    print("-" * 60)
    
    try:
        # 方法1：使用第一题的结果
        print("\n方法1：使用第一题的结果进行测试")
        result1 = task1_image_preprocessing(image_path, output_dir)
        img_gray = result1["gray"]
        img_bgr = result1["original_bgr"]
        
        # 调用第三题函数（非交互模式）
        result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir, show_windows=False)
        
        print("\n测试成功！")
        print(f"返回结果包含以下键: {list(result3.keys())}")
        print(f"检测到的有效灯条数: {len(result3['valid_lines'])}")
        print(f"左灯条数: {len(result3['left_bars'])}")
        print(f"右灯条数: {len(result3['right_bars'])}")
        
        # 显示灯条详细信息
        if result3['valid_lines']:
            print("\n有效灯条详细信息:")
            for i, line_info in enumerate(result3['valid_lines'][:5]):  # 只显示前5条
                print(f"  灯条 {i+1}: 长度={line_info['length']:.2f}, "
                      f"角度={line_info['angle']:.2f}°, "
                      f"中心=({line_info['center'][0]}, {line_info['center'][1]})")
        
        # 方法2：直接读取图像
        print("\n" + "="*60)
        print("方法2：直接读取图像测试")
        print("="*60)
        img_bgr2 = cv2.imread(image_path)
        img_gray2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2GRAY)
        result3_2 = task3_light_bar_extraction(img_gray2, img_bgr2, output_dir, show_windows=False)
        
        print(f"\n检测到的有效灯条数: {len(result3_2['valid_lines'])}")
        print(f"左灯条数: {len(result3_2['left_bars'])}")
        print(f"右灯条数: {len(result3_2['right_bars'])}")
        
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
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    result = task3_light_bar_extraction(img_gray, img_bgr, 
                                                        output_dir="output", 
                                                        show_windows=False)
                    print(f"  有效灯条: {len(result['valid_lines'])}, "
                          f"左灯条: {len(result['left_bars'])}, "
                          f"右灯条: {len(result['right_bars'])}")
                except Exception as e:
                    print(f"  处理失败: {e}")
        else:
            print(f"图像不存在: {img_path}")


if __name__ == "__main__":
    # 测试单张图像
    test_task3()
    
    # 测试多张图像
    test_multiple_images()

