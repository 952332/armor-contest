"""
测试第五题改进效果
对比原版和改进版的识别效果
"""

import cv2
import numpy as np
import os
from armor import task1_image_preprocessing, task3_light_bar_extraction, task5_number_recognition
from task5_improved_methods import task5_number_recognition_improved

def compare_original_vs_improved():
    """对比原版和改进版的效果"""
    
    print("=" * 80)
    print("第五题改进效果对比测试")
    print("=" * 80)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    template_dir = "digit_templates"
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
            
            # 灯条提取
            result3 = task3_light_bar_extraction(img_gray, img_bgr, 
                                                 output_dir=None, 
                                                 show_windows=False)
            left_bars = result3["left_bars"]
            right_bars = result3["right_bars"]
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                print(f"  跳过: 未检测到足够的灯条")
                continue
            
            # 原版方法
            print("\n【原版方法】")
            result_original = task5_number_recognition(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir=template_dir,
                show_windows=False
            )
            
            original_digit = "?"
            original_conf = 0.0
            if result_original['recognized_numbers']:
                num_info = result_original['recognized_numbers'][0]
                original_digit = num_info['digit']
                original_conf = num_info['confidence']
            
            print(f"  识别结果: 数字={original_digit}, 置信度={original_conf:.2f}")
            
            # 改进版方法（组合方法）
            print("\n【改进版方法 - 组合方法】")
            result_improved = task5_number_recognition_improved(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir=template_dir,
                method="combined",
                show_windows=False
            )
            
            improved_digit = "?"
            improved_conf = 0.0
            improved_method = ""
            if result_improved['recognized_numbers']:
                num_info = result_improved['recognized_numbers'][0]
                improved_digit = num_info['digit']
                improved_conf = num_info['confidence']
                improved_method = num_info['method']
            
            print(f"  识别结果: 数字={improved_digit}, 置信度={improved_conf:.2f}, 方法={improved_method}")
            
            # 改进版方法（模板匹配）
            print("\n【改进版方法 - 多尺度模板匹配】")
            result_template = task5_number_recognition_improved(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir=template_dir,
                method="template",
                show_windows=False
            )
            
            template_digit = "?"
            template_conf = 0.0
            if result_template['recognized_numbers']:
                num_info = result_template['recognized_numbers'][0]
                template_digit = num_info['digit']
                template_conf = num_info['confidence']
            
            print(f"  识别结果: 数字={template_digit}, 置信度={template_conf:.2f}")
            
            # 记录结果
            results.append({
                'image': os.path.basename(img_path),
                'original': {'digit': original_digit, 'conf': original_conf},
                'improved_combined': {'digit': improved_digit, 'conf': improved_conf, 'method': improved_method},
                'improved_template': {'digit': template_digit, 'conf': template_conf}
            })
            
        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结对比
    print("\n" + "=" * 80)
    print("改进效果总结")
    print("=" * 80)
    
    print(f"\n{'图像':<25} {'原版':<15} {'改进版(组合)':<20} {'改进版(模板)':<20}")
    print("-" * 80)
    
    for r in results:
        orig = f"{r['original']['digit']} ({r['original']['conf']:.2f})"
        imp_comb = f"{r['improved_combined']['digit']} ({r['improved_combined']['conf']:.2f})"
        imp_tpl = f"{r['improved_template']['digit']} ({r['improved_template']['conf']:.2f})"
        print(f"{r['image']:<25} {orig:<15} {imp_comb:<20} {imp_tpl:<20}")
    
    # 统计改进效果
    print("\n" + "=" * 80)
    print("改进效果分析")
    print("=" * 80)
    
    # 置信度提升统计
    conf_improvements = []
    for r in results:
        orig_conf = r['original']['conf']
        imp_conf = r['improved_combined']['conf']
        if orig_conf > 0 or imp_conf > 0:
            improvement = imp_conf - orig_conf
            conf_improvements.append(improvement)
    
    if conf_improvements:
        avg_improvement = np.mean(conf_improvements)
        print(f"\n置信度提升:")
        print(f"  平均提升: {avg_improvement:.3f}")
        print(f"  最大提升: {max(conf_improvements):.3f}")
        print(f"  最小提升: {min(conf_improvements):.3f}")
    
    # 识别成功率统计
    orig_success = sum(1 for r in results if r['original']['digit'] != "?")
    imp_success = sum(1 for r in results if r['improved_combined']['digit'] != "?")
    
    print(f"\n识别成功率:")
    print(f"  原版: {orig_success}/{len(results)} ({orig_success/len(results)*100:.1f}%)")
    print(f"  改进版: {imp_success}/{len(results)} ({imp_success/len(results)*100:.1f}%)")
    
    # 改进点总结
    print("\n" + "=" * 80)
    print("改进点总结")
    print("=" * 80)
    
    improvements = [
        "1. 多尺度模板匹配 - 支持不同缩放比例的模板匹配",
        "2. 特征匹配 - 基于几何特征和Hu矩的识别",
        "3. 轮廓匹配 - 基于轮廓特征的匹配",
        "4. 组合方法 - 多种方法投票，提高准确率",
        "5. 改进的预处理 - 更好的二值化和形态学操作"
    ]
    
    for imp in improvements:
        print(f"  [OK] {imp}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    compare_original_vs_improved()

