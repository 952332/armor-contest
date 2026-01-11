"""
详细对比原版和改进版第三题的效果
生成可视化对比和详细分析报告
"""

import cv2
import numpy as np
import os
from armor import task1_image_preprocessing, task2_color_segmentation, task3_light_bar_extraction
from armor_improved import task3_light_bar_extraction_improved

def create_detailed_comparison(image_path: str):
    """创建详细的对比可视化"""
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    print(f"\n处理图像: {os.path.basename(image_path)}")
    print("=" * 80)
    
    # 1. 图像预处理
    result1 = task1_image_preprocessing(image_path, output_dir="output")
    img_gray = result1["gray"]
    img_bgr = result1["original_bgr"]
    
    # 2. 颜色分割
    result2 = task2_color_segmentation(img_bgr, output_dir="output", show_windows=False)
    mask_red = result2["mask_red"]
    mask_blue = result2["mask_blue"]
    mask_combined = cv2.bitwise_or(mask_red, mask_blue)
    
    # 3. 原版检测
    print("\n【原版检测】")
    result_original = task3_light_bar_extraction(img_gray, img_bgr, 
                                                 output_dir="output", 
                                                 show_windows=False)
    
    # 4. 改进版检测
    print("\n【改进版检测】")
    result_improved = task3_light_bar_extraction_improved(
        img_gray, img_bgr, 
        mask_red=mask_red, 
        mask_blue=mask_blue,
        output_dir="output", 
        show_windows=False
    )
    
    # 5. 创建详细对比可视化
    h, w = img_bgr.shape[:2]
    
    # 准备各个图像
    mask_3ch = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
    
    # 原版边缘检测
    edges_orig_3ch = cv2.cvtColor(result_original['edges'], cv2.COLOR_GRAY2BGR)
    
    # 改进版边缘检测（如果有原始边缘）
    if 'edges_original' in result_improved:
        edges_imp_orig_3ch = cv2.cvtColor(result_improved['edges_original'], cv2.COLOR_GRAY2BGR)
        edges_imp_final_3ch = cv2.cvtColor(result_improved['edges'], cv2.COLOR_GRAY2BGR)
    else:
        edges_imp_orig_3ch = edges_orig_3ch.copy()
        edges_imp_final_3ch = cv2.cvtColor(result_improved['edges'], cv2.COLOR_GRAY2BGR)
    
    # 创建带标注的结果图像
    img_orig_annotated = result_original['result_image'].copy()
    img_imp_annotated = result_improved['result_image'].copy()
    
    # 添加统计信息到图像
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # 原版统计
    orig_text = f"Original: {len(result_original['valid_lines'])} lines (L:{len(result_original['left_bars'])} R:{len(result_original['right_bars'])})"
    cv2.putText(img_orig_annotated, orig_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
    
    # 改进版统计
    imp_text = f"Improved: {len(result_improved['valid_lines'])} lines (L:{len(result_improved['left_bars'])} R:{len(result_improved['right_bars'])})"
    cv2.putText(img_imp_annotated, imp_text, (10, 30), font, font_scale, (0, 255, 0), thickness)
    
    # 创建3x3网格对比
    # 第一行：原图、颜色掩码、原版边缘
    # 第二行：改进版原始边缘、改进版过滤后边缘、原版结果
    # 第三行：改进版结果、对比图1、对比图2
    
    row1 = np.hstack([img_bgr, mask_3ch, edges_orig_3ch])
    row2 = np.hstack([edges_imp_orig_3ch, edges_imp_final_3ch, img_orig_annotated])
    row3 = np.hstack([img_imp_annotated, img_orig_annotated, img_imp_annotated])
    
    # 调整大小
    scale = min(1400 / row1.shape[1], 1000 / (row1.shape[0] + row2.shape[0] + row3.shape[0]))
    new_w = int(row1.shape[1] * scale)
    new_h1 = int(row1.shape[0] * scale)
    new_h2 = int(row2.shape[0] * scale)
    new_h3 = int(row3.shape[0] * scale)
    
    row1_resized = cv2.resize(row1, (new_w, new_h1))
    row2_resized = cv2.resize(row2, (new_w, new_h2))
    row3_resized = cv2.resize(row3, (new_w, new_h3))
    
    combined = np.vstack([row1_resized, row2_resized, row3_resized])
    
    # 添加标签
    labels = [
        "Original Image", "Color Mask", "Original Edges",
        "Improved Original Edges", "Improved Filtered Edges", "Original Result",
        "Improved Result", "Original Result", "Improved Result"
    ]
    
    label_w = new_w // 3
    for i, label in enumerate(labels):
        x = (i % 3) * label_w + 10
        y = (i // 3) * new_h1 + 25
        
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        cv2.rectangle(combined, (x-5, y-text_size[1]-5), 
                     (x+text_size[0]+5, y+5), (0, 0, 0), -1)
        cv2.putText(combined, label, (x, y), font, 0.5, (255, 255, 255), 1)
    
    # 保存
    filename_base = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"output/task3_detailed_comparison_{filename_base}.jpg"
    cv2.imwrite(output_path, combined)
    print(f"\n详细对比图已保存: {output_path}")
    
    return {
        'filename': os.path.basename(image_path),
        'original': result_original,
        'improved': result_improved,
        'comparison_image': output_path
    }


def analyze_differences(result_original, result_improved):
    """分析原版和改进版的差异"""
    
    print("\n" + "=" * 80)
    print("详细对比分析")
    print("=" * 80)
    
    # 1. 检测数量对比
    print("\n【1. 检测数量对比】")
    print(f"原版检测到的灯条数: {len(result_original['valid_lines'])}")
    print(f"改进版检测到的灯条数: {len(result_improved['valid_lines'])}")
    diff = len(result_improved['valid_lines']) - len(result_original['valid_lines'])
    if diff > 0:
        print(f"改进版多检测到 {diff} 条（可能更敏感或检测到更多细节）")
    elif diff < 0:
        print(f"改进版少检测到 {abs(diff)} 条（可能过滤了误检）")
    else:
        print("检测数量相同")
    
    # 2. 分组对比
    print("\n【2. 左右分组对比】")
    orig_left = len(result_original['left_bars'])
    orig_right = len(result_original['right_bars'])
    imp_left = len(result_improved['left_bars'])
    imp_right = len(result_improved['right_bars'])
    
    print(f"原版: 左={orig_left}, 右={orig_right}, 差值={abs(orig_left - orig_right)}")
    print(f"改进版: 左={imp_left}, 右={imp_right}, 差值={abs(imp_left - imp_right)}")
    
    orig_balance = abs(orig_left - orig_right)
    imp_balance = abs(imp_left - imp_right)
    
    if imp_balance < orig_balance:
        print(f"[+] 改进版分组更均衡（差值减少 {orig_balance - imp_balance}）")
    elif imp_balance > orig_balance:
        print(f"[-] 改进版分组均衡性变差（差值增加 {imp_balance - orig_balance}）")
    else:
        print("分组均衡性相同")
    
    # 3. 灯条长度分析
    print("\n【3. 灯条长度分析】")
    if result_original['valid_lines']:
        orig_lengths = [l['length'] for l in result_original['valid_lines']]
        print(f"原版平均长度: {np.mean(orig_lengths):.1f}px, "
              f"范围: {np.min(orig_lengths):.1f}-{np.max(orig_lengths):.1f}px")
    
    if result_improved['valid_lines']:
        imp_lengths = [l['length'] for l in result_improved['valid_lines']]
        print(f"改进版平均长度: {np.mean(imp_lengths):.1f}px, "
              f"范围: {np.min(imp_lengths):.1f}-{np.max(imp_lengths):.1f}px")
    
    # 4. 角度分析
    print("\n【4. 角度分析】")
    if result_original['valid_lines']:
        orig_angles = [abs(abs(l['angle']) - 90) for l in result_original['valid_lines']]
        print(f"原版平均角度偏差: {np.mean(orig_angles):.2f}°")
    
    if result_improved['valid_lines']:
        imp_angles = [abs(abs(l['angle']) - 90) for l in result_improved['valid_lines']]
        print(f"改进版平均角度偏差: {np.mean(imp_angles):.2f}°")
    
    # 5. 改进效果总结
    print("\n【5. 改进效果总结】")
    improvements = []
    regressions = []
    
    if len(result_improved['valid_lines']) > len(result_original['valid_lines']):
        improvements.append("检测到更多灯条（可能更敏感）")
    elif len(result_improved['valid_lines']) < len(result_original['valid_lines']):
        regressions.append("检测数量减少（可能过滤了误检）")
    
    if imp_balance < orig_balance:
        improvements.append("左右分组更均衡")
    elif imp_balance > orig_balance:
        regressions.append("左右分组均衡性变差")
    
    if improvements:
        print("[+] 改进点:")
        for imp in improvements:
            print(f"  - {imp}")
    
    if regressions:
        print("[-] 需要注意:")
        for reg in regressions:
            print(f"  - {reg}")
    
    if not improvements and not regressions:
        print("检测结果基本相同")


def compare_all_images():
    """对比所有测试图像"""
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    all_results = []
    
    for img_path in test_images:
        if os.path.exists(img_path):
            result = create_detailed_comparison(img_path)
            all_results.append(result)
            
            # 分析差异
            analyze_differences(result['original'], result['improved'])
    
    # 生成总结报告
    print("\n" + "=" * 80)
    print("总体对比总结")
    print("=" * 80)
    
    print(f"\n{'图像':<30} {'原版':<20} {'改进版':<20} {'改进效果'}")
    print("-" * 80)
    
    for res in all_results:
        orig = res['original']
        imp = res['improved']
        
        orig_str = f"{len(orig['valid_lines'])}条(L:{len(orig['left_bars'])} R:{len(orig['right_bars'])})"
        imp_str = f"{len(imp['valid_lines'])}条(L:{len(imp['left_bars'])} R:{len(imp['right_bars'])})"
        
        # 判断改进效果
        orig_balance = abs(len(orig['left_bars']) - len(orig['right_bars']))
        imp_balance = abs(len(imp['left_bars']) - len(imp['right_bars']))
        
        if imp_balance < orig_balance:
            effect = "分组更均衡"
        elif len(imp['valid_lines']) > len(orig['valid_lines']):
            effect = "检测更多"
        elif len(imp['valid_lines']) < len(orig['valid_lines']):
            effect = "过滤误检"
        else:
            effect = "基本相同"
        
        print(f"{res['filename']:<30} {orig_str:<20} {imp_str:<20} {effect}")
    
    print("\n详细对比图已保存在 output/ 目录")


if __name__ == "__main__":
    compare_all_images()

