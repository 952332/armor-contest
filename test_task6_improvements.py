"""
测试第六题改进效果
对比原版和改进版的识别效果
"""

import cv2
import numpy as np
import os
from armor import (
    task1_image_preprocessing,
    task2_color_segmentation,
    task3_light_bar_extraction,
    task5_number_recognition,
    task6_armor_detection
)
from task6_improved import task6_armor_detection_improved

def compare_original_vs_improved():
    """对比原版和改进版的效果"""
    
    print("=" * 80)
    print("第六题改进效果对比测试")
    print("=" * 80)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    template_dir = "digit_templates"
    output_dir = "output"
    results = []
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        print(f"\n{'='*80}")
        print(f"处理图像: {os.path.basename(img_path)}")
        print(f"{'='*80}")
        
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # 颜色分割
            result2 = task2_color_segmentation(img_bgr, output_dir=None, show_windows=False)
            mask_red = result2["mask_red"]
            mask_blue = result2["mask_blue"]
            
            # 灯条提取
            result3 = task3_light_bar_extraction(img_gray, img_bgr, 
                                                 output_dir=None, 
                                                 show_windows=False)
            left_bars = result3["left_bars"]
            right_bars = result3["right_bars"]
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                print(f"  跳过: 未检测到足够的灯条")
                continue
            
            # 数字识别
            result5 = task5_number_recognition(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir=template_dir,
                show_windows=False
            )
            recognized_numbers = result5["recognized_numbers"]
            
            # 原版方法
            print("\n【原版方法】")
            result_original = task6_armor_detection(
                img_bgr, left_bars, right_bars,
                mask_red, mask_blue,
                recognized_numbers,
                output_dir=output_dir
            )
            
            original_count = len(result_original['valid_armors'])
            print(f"  检测结果: {original_count} 个装甲板")
            
            # 改进版方法（要求数字）
            print("\n【改进版方法 - 要求数字验证】")
            result_improved1 = task6_armor_detection_improved(
                img_bgr, left_bars, right_bars,
                mask_red, mask_blue,
                recognized_numbers,
                output_dir=output_dir,
                min_score=50.0,
                require_number=True,
                show_windows=False
            )
            
            improved1_count = len(result_improved1['valid_armors'])
            print(f"  检测结果: {improved1_count} 个装甲板")
            
            # 改进版方法（不要求数字）
            print("\n【改进版方法 - 不要求数字验证】")
            result_improved2 = task6_armor_detection_improved(
                img_bgr, left_bars, right_bars,
                mask_red, mask_blue,
                recognized_numbers,
                output_dir=output_dir,
                min_score=50.0,
                require_number=False,
                show_windows=False
            )
            
            improved2_count = len(result_improved2['valid_armors'])
            print(f"  检测结果: {improved2_count} 个装甲板")
            
            if improved2_count > 0:
                for armor in result_improved2['valid_armors']:
                    print(f"    {armor['color']} {armor['number']}, "
                          f"评分: {armor['score']:.1f}")
            
            # 记录结果
            results.append({
                'image': os.path.basename(img_path),
                'original': original_count,
                'improved_require_number': improved1_count,
                'improved_no_number': improved2_count
            })
            
        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结对比
    print("\n" + "=" * 80)
    print("改进效果总结")
    print("=" * 80)
    
    print(f"\n{'图像':<25} {'原版':<10} {'改进版(要求数字)':<20} {'改进版(不要求数字)':<25}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['image']:<25} {r['original']:<10} {r['improved_require_number']:<20} {r['improved_no_number']:<25}")
    
    # 统计改进效果
    print("\n" + "=" * 80)
    print("改进效果分析")
    print("=" * 80)
    
    orig_total = sum(r['original'] for r in results)
    imp1_total = sum(r['improved_require_number'] for r in results)
    imp2_total = sum(r['improved_no_number'] for r in results)
    
    print(f"\n检测总数:")
    print(f"  原版: {orig_total}")
    print(f"  改进版(要求数字): {imp1_total}")
    print(f"  改进版(不要求数字): {imp2_total}")
    
    print(f"\n改进效果:")
    if orig_total == 0:
        print(f"  原版检测失败，改进版成功检测 {imp2_total} 个装甲板")
    else:
        improvement = ((imp2_total - orig_total) / orig_total * 100) if orig_total > 0 else 0
        print(f"  检测数量提升: {imp2_total - orig_total} ({improvement:+.1f}%)")
    
    # 改进点总结
    print("\n" + "=" * 80)
    print("改进点总结")
    print("=" * 80)
    
    improvements = [
        "1. 评分机制 - 综合多个因素进行评分，更灵活",
        "2. 降低数字置信度阈值 - 从0.5降低到0.3，更实用",
        "3. 支持无数字验证模式 - 仅基于几何特征识别",
        "4. 改进灯条配对算法 - 避免重复检测",
        "5. 详细的评分信息 - 便于调试和优化"
    ]
    
    for imp in improvements:
        print(f"  [OK] {imp}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    compare_original_vs_improved()

