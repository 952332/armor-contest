"""
评估第六题检测质量
详细分析检测效果
"""

import cv2
import numpy as np
import os
from armor import (
    task1_image_preprocessing,
    task2_color_segmentation,
    task3_light_bar_extraction,
    task5_number_recognition
)
from task6_improved import task6_armor_detection_improved

def evaluate_detection_quality():
    """评估检测质量"""
    
    print("=" * 80)
    print("第六题检测质量评估")
    print("=" * 80)
    
    test_images = [
        ("test_images/armor.jpg", "armor.jpg"),
        ("test_images/armor_001_normal.jpg", "armor_001_normal.jpg"),
        ("test_images/armor_002_dark.jpg", "armor_002_dark.jpg"),
        ("test_images/armor_005_angled.jpg", "armor_005_angled.jpg")
    ]
    
    template_dir = "digit_templates"
    output_dir = "output"
    
    all_results = []
    
    for img_path, img_name in test_images:
        if not os.path.exists(img_path):
            continue
        
        print(f"\n{'='*80}")
        print(f"评估图像: {img_name}")
        print(f"{'='*80}")
        
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            
            h, w = img_bgr.shape[:2]
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
            
            print(f"\n【1. 灯条检测】")
            print(f"  左灯条数: {len(left_bars)}")
            print(f"  右灯条数: {len(right_bars)}")
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                print(f"  [FAIL] 未检测到足够的灯条")
                all_results.append({
                    'image': img_name,
                    'status': 'failed',
                    'reason': 'no_bars'
                })
                continue
            
            # 数字识别
            result5 = task5_number_recognition(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir=template_dir,
                show_windows=False
            )
            recognized_numbers = result5["recognized_numbers"]
            
            print(f"\n【2. 数字识别】")
            if recognized_numbers:
                for num_info in recognized_numbers:
                    print(f"  数字: {num_info['digit']}, 置信度: {num_info['confidence']:.2f}")
            else:
                print(f"  [WARN] 未识别到数字")
            
            # 装甲板检测（改进版）
            result6 = task6_armor_detection_improved(
                img_bgr, left_bars, right_bars,
                mask_red, mask_blue,
                recognized_numbers,
                output_dir=output_dir,
                min_score=50.0,
                require_number=False,
                show_windows=False
            )
            
            valid_armors = result6['valid_armors']
            
            print(f"\n【3. 装甲板检测】")
            print(f"  检测数量: {len(valid_armors)}")
            
            if len(valid_armors) == 0:
                print(f"  [FAIL] 未检测到装甲板")
                all_results.append({
                    'image': img_name,
                    'status': 'failed',
                    'reason': 'no_armor'
                })
                continue
            
            # 详细分析每个检测结果
            for i, armor in enumerate(valid_armors):
                print(f"\n  装甲板 {i+1}:")
                print(f"    颜色: {armor['color']}")
                print(f"    数字: {armor['number']}")
                print(f"    数字置信度: {armor['number_confidence']:.2f}")
                print(f"    综合评分: {armor['score']:.1f}/100")
                
                # 评分明细
                details = armor['details']
                print(f"    评分明细:")
                print(f"      角度相似性: {details['angle_score']:.1f}/30")
                print(f"      长度比例: {details['length_score']:.1f}/20")
                print(f"      间距合理性: {details['distance_score']:.1f}/20")
                print(f"      垂直位置: {details['vertical_score']:.1f}/10")
                print(f"      数字验证: {details['number_score']:.1f}/15")
                print(f"      颜色一致性: {details['color_score']:.1f}/5")
                
                # 边界框信息
                x, y, w, h = armor['bbox']
                print(f"    位置: ({x}, {y}), 尺寸: {w}x{h}")
                print(f"    占图像比例: {w/h:.2f} (宽/高)")
                
                # 质量评估
                quality = "优秀" if armor['score'] >= 80 else "良好" if armor['score'] >= 60 else "一般"
                print(f"    质量评估: {quality}")
            
            all_results.append({
                'image': img_name,
                'status': 'success',
                'armor_count': len(valid_armors),
                'armors': valid_armors
            })
            
        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'image': img_name,
                'status': 'error',
                'error': str(e)
            })
    
    # 总结统计
    print("\n" + "=" * 80)
    print("检测质量总结")
    print("=" * 80)
    
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    total_count = len(all_results)
    
    print(f"\n【总体统计】")
    print(f"  测试图像数: {total_count}")
    print(f"  成功检测: {success_count}")
    print(f"  成功率: {success_count/total_count*100:.1f}%")
    
    # 评分统计
    all_scores = []
    for r in all_results:
        if r['status'] == 'success' and 'armors' in r:
            for armor in r['armors']:
                all_scores.append(armor['score'])
    
    if all_scores:
        print(f"\n【评分统计】")
        print(f"  平均评分: {np.mean(all_scores):.1f}/100")
        print(f"  最高评分: {max(all_scores):.1f}/100")
        print(f"  最低评分: {min(all_scores):.1f}/100")
        print(f"  评分标准差: {np.std(all_scores):.1f}")
        
        # 质量分布
        excellent = sum(1 for s in all_scores if s >= 80)
        good = sum(1 for s in all_scores if 60 <= s < 80)
        fair = sum(1 for s in all_scores if s < 60)
        
        print(f"\n【质量分布】")
        print(f"  优秀 (≥80分): {excellent} ({excellent/len(all_scores)*100:.1f}%)")
        print(f"  良好 (60-79分): {good} ({good/len(all_scores)*100:.1f}%)")
        print(f"  一般 (<60分): {fair} ({fair/len(all_scores)*100:.1f}%)")
    
    # 数字识别统计
    number_success = 0
    number_total = 0
    for r in all_results:
        if r['status'] == 'success' and 'armors' in r:
            for armor in r['armors']:
                number_total += 1
                if armor['number'] != "?":
                    number_success += 1
    
    if number_total > 0:
        print(f"\n【数字识别统计】")
        print(f"  有数字的装甲板: {number_success}/{number_total} ({number_success/number_total*100:.1f}%)")
    
    # 颜色统计
    red_count = 0
    blue_count = 0
    for r in all_results:
        if r['status'] == 'success' and 'armors' in r:
            for armor in r['armors']:
                if armor['color'] == 'red':
                    red_count += 1
                else:
                    blue_count += 1
    
    print(f"\n【颜色统计】")
    print(f"  红色装甲板: {red_count}")
    print(f"  蓝色装甲板: {blue_count}")
    
    # 检测质量评估
    print(f"\n【检测质量评估】")
    if success_count == total_count and all_scores and np.mean(all_scores) >= 70:
        print(f"  [EXCELLENT] 检测效果优秀！")
        print(f"    - 所有图像都能成功检测")
        print(f"    - 平均评分 {np.mean(all_scores):.1f} 分，质量良好")
    elif success_count == total_count:
        print(f"  [GOOD] 检测效果良好")
        print(f"    - 所有图像都能成功检测")
        if all_scores:
            print(f"    - 平均评分 {np.mean(all_scores):.1f} 分，可以进一步优化")
    elif success_count >= total_count * 0.75:
        print(f"  [FAIR] 检测效果一般")
        print(f"    - 成功率 {success_count/total_count*100:.1f}%，需要改进")
    else:
        print(f"  [POOR] 检测效果较差")
        print(f"    - 成功率 {success_count/total_count*100:.1f}%，需要大幅改进")
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_detection_quality()

