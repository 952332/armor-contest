"""
批量可视化第三题的结果
可以一次性查看多张图像的灯条提取效果
"""

import cv2
import numpy as np
import os
from armor import task3_light_bar_extraction

def batch_visualize_task3(image_dir: str = "test_images", output_dir: str = "output"):
    """批量可视化多张图像的第三题结果"""
    
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
        
        # 转换为灰度图
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 执行灯条提取
        try:
            result = task3_light_bar_extraction(img_gray, img_bgr, 
                                               output_dir=None, 
                                               show_windows=False)
            
            results.append({
                'filename': img_file,
                'image': img_bgr,
                'result': result
            })
            
            print(f"  有效灯条: {len(result['valid_lines'])}, "
                  f"左灯条: {len(result['left_bars'])}, "
                  f"右灯条: {len(result['right_bars'])}")
            
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
    for res in results:
        img_bgr = res['image']
        result = res['result']
        
        h, w = img_bgr.shape[:2]
        
        # 将边缘检测结果转换为3通道
        edges_3ch = cv2.cvtColor(result['edges'], cv2.COLOR_GRAY2BGR)
        
        # 创建显示所有检测直线的图像
        img_with_all_lines = img_bgr.copy()
        for line_info in result['valid_lines']:
            x1, y1, x2, y2 = line_info['line']
            cv2.line(img_with_all_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 标记中心点
            cv2.circle(img_with_all_lines, line_info['center'], 3, (255, 0, 0), -1)
        
        # 创建2x2网格
        row1 = np.hstack([img_bgr, edges_3ch])
        row2 = np.hstack([img_with_all_lines, result['result_image']])
        
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
            "Original", "Canny Edges",
            "Detected Lines", "Light Bars"
        ]
        
        label_w = new_w // 2
        for j, label in enumerate(labels):
            x = (j % 2) * label_w + 10
            y = (j // 2) * new_h1 + 25
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.rectangle(combined, (x-5, y-text_size[1]-5), 
                         (x+text_size[0]+5, y+5), bg_color, -1)
            cv2.putText(combined, label, (x, y), font, font_scale, color, thickness)
        
        # 添加文件名和统计信息
        stats_y = combined.shape[0] - 30
        stats_text = (f"{res['filename']} | "
                     f"Valid Lines: {len(result['valid_lines'])} | "
                     f"Left: {len(result['left_bars'])} | "
                     f"Right: {len(result['right_bars'])}")
        text_size = cv2.getTextSize(stats_text, font, 0.5, 1)[0]
        stats_x = (combined.shape[1] - text_size[0]) // 2
        cv2.rectangle(combined, (stats_x-10, stats_y-text_size[1]-5), 
                     (stats_x+text_size[0]+10, stats_y+5), bg_color, -1)
        cv2.putText(combined, stats_text, (stats_x, stats_y), font, 0.5, color, 1)
        
        # 保存
        output_path = os.path.join(output_dir, f"task3_vis_{res['filename']}")
        cv2.imwrite(output_path, combined)
        print(f"  已保存: {output_path}")


if __name__ == "__main__":
    import sys
    
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    batch_visualize_task3(image_dir, output_dir)

