"""
测试第一题：基础图像读取与预处理
"""

from armor import task1_image_preprocessing
import os

def test_task1():
    """测试题目1的功能"""
    # 测试图像路径
    image_path = "test_images/armor.jpg"
    output_dir = "output"
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"错误：测试图像不存在: {image_path}")
        print("请先运行 generate_test_image.py 生成测试图像")
        return
    
    print("开始测试题目1：基础图像读取与预处理")
    print("-" * 60)
    
    try:
        # 调用第一题函数
        result = task1_image_preprocessing(image_path, output_dir)
        
        print("\n测试成功！")
        print(f"返回结果包含以下键: {list(result.keys())}")
        print(f"原始图像尺寸: {result['original_bgr'].shape}")
        print(f"RGB图像尺寸: {result['rgb'].shape}")
        print(f"灰度图像尺寸: {result['gray'].shape}")
        print(f"模糊图像尺寸: {result['blurred'].shape}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_task1()

