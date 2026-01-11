"""
批量可视化第二题的结果
可以一次性查看多张图像的颜色分割效果
"""

import cv2
import numpy as np
import os
from armor import task2_color_segmentation

def batch_visualize_task2(image_dir: str = "test_images", output_dir: str = "output"):
    """批量可视化多张图像的第二题结果"""
    
    if not os.path.isdir(image_dir):
        print(f"目录不存在: {image_dir}")
        return
    
    # 获取所有jpg图像
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"目录中没有找到图像文件: {image_dir}")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始处理...")
    
    results = []
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        print(f"\n处理: {img_file}")
        
        # 读取图像
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  无法读取图像: {img_path}")
            continue
        
        # 执行颜色分割（不保存中间结果，只返回数据）
        try:
            result = task2_color_segmentation(img_bgr, output_dir=None, show_windows=False)
            
            # 计算统计信息
            total_pixels = result['mask_red'].size
            red_pixels = np.sum(result['mask_red'] > 0)
            blue_pixels = np.sum(result['mask_blue'] > 0)
            red_percent = red_pixels / total_pixels * 100
            blue_percent = blue_pixels / total_pixels * 100
            
            results.append({
                'filename': img_file,
                'image': img_bgr,
                'result': result,
                'stats': {
                    'red_pixels': red_pixels,
                    'blue_pixels': blue_pixels,
                    'red_percent': red_percent,
                    'blue_percent': blue_percent
                }
            })
            
            print(f"  红色像素: {red_pixels} ({red_percent:.2f}%)")
            print(f"  蓝色像素: {blue_pixels} ({blue_percent:.2f}%)")
            
        except Exception as e:
            print(f"  处理失败: {e}")
            continue
    
    if not results:
        print("没有成功处理的图像")
        return
    
    # 创建对比可视化
    print(f"\n创建对比可视化...")
    create_comparison_visualization(results, output_dir)
    
    print(f"\n完成！共处理 {len(results)} 张图像")


def create_comparison_visualization(results, output_dir: str):
    """创建对比可视化图像"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每张图像创建可视化
    for i, res in enumerate(results):
        img_bgr = res['image']
        result = res['result']
        
        h, w = img_bgr.shape[:2]
        
        # 将掩码转换为3通道
        mask_red_3ch = cv2.cvtColor(result['mask_red'], cv2.COLOR_GRAY2BGR)
        mask_blue_3ch = cv2.cvtColor(result['mask_blue'], cv2.COLOR_GRAY2BGR)
        
        # 创建2x3网格
        row1 = np.hstack([img_bgr, result['hsv'], mask_red_3ch])
        row2 = np.hstack([mask_blue_3ch, result['red_region'], result['blue_region']])
        
        # 调整大小
        scale = min(800 / row1.shape[1], 600 / (row1.shape[0] + row2.shape[0]))
        new_w = int(row1.shape[1] * scale)
        new_h1 = int(row1.shape[0] * scale)
        new_h2 = int(row2.shape[0] * scale)
        
        row1_resized = cv2.resize(row1, (new_w, new_h1))
        row2_resized = cv2.resize(row2, (new_w, new_h2))
        
        combined = np.vstack([row1_resized, row2_resized])
        
        # 添加标签和统计信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        labels = [
            "Original", "HSV", "Red Mask",
            "Blue Mask", "Red Region", "Blue Region"
        ]
        
        label_w = new_w // 3
        for j, label in enumerate(labels):
            x = (j % 3) * label_w + 10
            y = (j // 3) * new_h1 + 25
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(combined, (x-5, y-text_size[1]-5), 
                         (x+text_size[0]+5, y+5), bg_color, -1)
            cv2.putText(combined, label, (x, y), font, font_scale, color, thickness)
        
        # 添加文件名和统计信息
        stats = res['stats']
        stats_y = combined.shape[0] - 30
        stats_text = (f"{res['filename']} | "
                     f"Red: {stats['red_pixels']} ({stats['red_percent']:.2f}%) | "
                     f"Blue: {stats['blue_pixels']} ({stats['blue_percent']:.2f}%)")
        text_size = cv2.getTextSize(stats_text, font, 0.5, 1)[0]
        stats_x = (combined.shape[1] - text_size[0]) // 2
        cv2.rectangle(combined, (stats_x-10, stats_y-text_size[1]-5), 
                     (stats_x+text_size[0]+10, stats_y+5), bg_color, -1)
        cv2.putText(combined, stats_text, (stats_x, stats_y), font, 0.5, color, 1)
        
        # 保存
        output_path = os.path.join(output_dir, f"task2_vis_{res['filename']}")
        cv2.imwrite(output_path, combined)
        print(f"  已保存: {output_path}")


if __name__ == "__main__":
    import sys
    
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    batch_visualize_task2(image_dir, output_dir)

