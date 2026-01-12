"""
测试第三题改进版本
对比原版和改进版的效果
"""

import cv2
import numpy as np
import os
from armor import task1_image_preprocessing, task2_color_segmentation, task3_light_bar_extraction
from armor_improved import task3_light_bar_extraction_improved

def compare_versions():
    """对比原版和改进版的效果"""
    
    print("=" * 80)
    print("第三题改进版对比测试")
    print("=" * 80)
    
    image_path = "test_images/armor.jpg"
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    # 读取图像
    result1 = task1_image_preprocessing(image_path, output_dir="output")
    img_gray = result1["gray"]
    img_bgr = result1["original_bgr"]
    
    # 获取颜色分割结果
    result2 = task2_color_segmentation(img_bgr, output_dir="output", show_windows=False)
    mask_red = result2["mask_red"]
    mask_blue = result2["mask_blue"]
    
    print("\n" + "="*80)
    print("原版测试")
    print("="*80)
    result_original = task3_light_bar_extraction(img_gray, img_bgr, 
                                               output_dir="output", 
                                               show_windows=False)
    
    print("\n" + "="*80)
    print("改进版测试")
    print("="*80)
    result_improved = task3_light_bar_extraction_improved(
        img_gray, img_bgr, 
        mask_red=mask_red, 
        mask_blue=mask_blue,
        output_dir="output", 
        show_windows=False
    )
    
    # 对比结果
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    
    comparison = {
        "检测到的灯条数": {
            "原版": len(result_original['valid_lines']),
            "改进版": len(result_improved['valid_lines']),
            "说明": "改进版应该更准确（结合颜色信息）"
        },
        "左灯条数": {
            "原版": len(result_original['left_bars']),
            "改进版": len(result_improved['left_bars']),
            "说明": "改进版分组应该更合理"
        },
        "右灯条数": {
            "原版": len(result_original['right_bars']),
            "改进版": len(result_improved['right_bars']),
            "说明": "改进版分组应该更合理"
        }
    }
    
    print(f"\n{'指标':<20} {'原版':<15} {'改进版':<15} {'说明'}")
    print("-" * 80)
    for metric, data in comparison.items():
        print(f"{metric:<20} {data['原版']:<15} {data['改进版']:<15} {data['说明']}")
    
    # 创建对比可视化
    create_comparison_visualization(img_bgr, result_original, result_improved, 
                                   result2, "output/task3_comparison.jpg")
    
    print("\n对比可视化已保存到: output/task3_comparison.jpg")


def create_comparison_visualization(img_bgr, result_original, result_improved, 
                                   color_result, output_path):
    """创建对比可视化图像"""
    
    h, w = img_bgr.shape[:2]
    
    # 创建2x3网格
    # 第一行：原图、颜色掩码、原版边缘检测
    # 第二行：改进版边缘检测、原版结果、改进版结果
    
    # 颜色掩码可视化
    mask_combined = cv2.bitwise_or(color_result['mask_red'], color_result['mask_blue'])
    mask_3ch = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
    
    # 边缘检测对比
    edges_original_3ch = cv2.cvtColor(result_original['edges'], cv2.COLOR_GRAY2BGR)
    edges_improved_3ch = cv2.cvtColor(result_improved['edges'], cv2.COLOR_GRAY2BGR)
    

    row1 = np.hstack([img_bgr, mask_3ch, edges_original_3ch])
    row2 = np.hstack([edges_improved_3ch, result_original['result_image'], 
                     result_improved['result_image']])
    
    # 调整大小
    scale = min(1200 / row1.shape[1], 800 / (row1.shape[0] + row2.shape[0]))
    new_w = int(row1.shape[1] * scale)
    new_h1 = int(row1.shape[0] * scale)
    new_h2 = int(row2.shape[0] * scale)
    
    row1_resized = cv2.resize(row1, (new_w, new_h1))
    row2_resized = cv2.resize(row2, (new_w, new_h2))
    
    combined = np.vstack([row1_resized, row2_resized])

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    bg_color = (0, 0, 0)
    
    labels = [
        "Original", "Color Mask", "Original Edges",
        "Improved Edges", "Original Result", "Improved Result"
    ]
    
    label_w = new_w // 3
    for i, label in enumerate(labels):
        x = (i % 3) * label_w + 10
        y = (i // 3) * new_h1 + 30
        
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        cv2.rectangle(combined, (x-5, y-text_size[1]-5), 
                     (x+text_size[0]+5, y+5), bg_color, -1)
        cv2.putText(combined, label, (x, y), font, font_scale, color, thickness)
    
    # 添加统计信息
    stats_y = combined.shape[0] - 30
    stats_text = (f"Original: {len(result_original['valid_lines'])} lines, "
                 f"L:{len(result_original['left_bars'])} R:{len(result_original['right_bars'])} | "
                 f"Improved: {len(result_improved['valid_lines'])} lines, "
                 f"L:{len(result_improved['left_bars'])} R:{len(result_improved['right_bars'])}")
    text_size = cv2.getTextSize(stats_text, font, 0.5, 1)[0]
    stats_x = (combined.shape[1] - text_size[0]) // 2
    cv2.rectangle(combined, (stats_x-10, stats_y-text_size[1]-5), 
                 (stats_x+text_size[0]+10, stats_y+5), bg_color, -1)
    cv2.putText(combined, stats_text, (stats_x, stats_y), font, 0.5, color, 1)
    
    cv2.imwrite(output_path, combined)


def test_multiple_images():
    """测试多张图像的改进效果"""
    
    print("\n" + "="*80)
    print("多图像测试")
    print("="*80)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    results_summary = []
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        print(f"\n处理: {os.path.basename(img_path)}")
        
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 颜色分割
        result2 = task2_color_segmentation(img_bgr, output_dir=None, show_windows=False)
        mask_red = result2["mask_red"]
        mask_blue = result2["mask_blue"]
        
        # 原版
        result_orig = task3_light_bar_extraction(img_gray, img_bgr, 
                                                 output_dir=None, show_windows=False)
        
        # 改进版
        result_imp = task3_light_bar_extraction_improved(
            img_gray, img_bgr, mask_red=mask_red, mask_blue=mask_blue,
            output_dir=None, show_windows=False
        )
        
        results_summary.append({
            'filename': os.path.basename(img_path),
            'original': {
                'total': len(result_orig['valid_lines']),
                'left': len(result_orig['left_bars']),
                'right': len(result_orig['right_bars'])
            },
            'improved': {
                'total': len(result_imp['valid_lines']),
                'left': len(result_imp['left_bars']),
                'right': len(result_imp['right_bars'])
            }
        })
        
        print(f"  原版: {len(result_orig['valid_lines'])}条 (L:{len(result_orig['left_bars'])} R:{len(result_orig['right_bars'])})")
        print(f"  改进版: {len(result_imp['valid_lines'])}条 (L:{len(result_imp['left_bars'])} R:{len(result_imp['right_bars'])})")

    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"{'图像':<30} {'原版':<20} {'改进版':<20} {'改进'}")
    print("-" * 80)
    
    for res in results_summary:
        orig_str = f"{res['original']['total']}条(L:{res['original']['left']} R:{res['original']['right']})"
        imp_str = f"{res['improved']['total']}条(L:{res['improved']['left']} R:{res['improved']['right']})"
        
        # 计算改进（减少误检或提高分组准确性）
        improvement = ""
        if res['improved']['total'] < res['original']['total']:
            improvement = "减少误检"
        elif abs(res['improved']['left'] - res['improved']['right']) < abs(res['original']['left'] - res['original']['right']):
            improvement = "分组更均衡"
        else:
            improvement = "保持"
        
        print(f"{res['filename']:<30} {orig_str:<20} {imp_str:<20} {improvement}")


if __name__ == "__main__":
    compare_versions()
    test_multiple_images()

