"""
第三题实现质量评估
全面分析灯条提取算法的效果和质量
"""

import cv2
import numpy as np
import os
from armor import task3_light_bar_extraction
import json

def evaluate_task3_quality():
    """评估第三题的实现质量"""
    
    print("=" * 80)
    print("第三题：装甲板灯条提取 - 实现质量评估报告")
    print("=" * 80)
    
    # 1. 算法设计评估
    print("\n【1. 算法设计评估】")
    print("-" * 80)
    
    algorithm_points = {
        "优点": [
            "[+] 使用Canny边缘检测，适合提取灯条边缘",
            "[+] 使用HoughLinesP概率霍夫变换，适合检测线段",
            "[+] 通过长度和角度筛选，符合灯条几何特征",
            "[+] 角度标准化处理正确（[-90, 90]范围）",
            "[+] 简单的左右分组策略，实现简单"
        ],
        "缺点": [
            "[-] 未结合颜色信息（第二题的结果），可能检测到非灯条边缘",
            "[-] 左右分组过于简单，仅按图像中心分割，不够精确",
            "[-] 未考虑灯条配对（左右灯条应该成对出现）",
            "[-] 未使用形态学操作优化边缘检测结果",
            "[-] 参数固定，未根据图像自适应调整"
        ]
    }
    
    for category, points in algorithm_points.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  {point}")
    
    # 2. 参数设置评估
    print("\n【2. 参数设置评估】")
    print("-" * 80)
    
    parameters = {
        "Canny边缘检测": {
            "low_threshold": 50,
            "high_threshold": 150,
            "评估": "固定阈值，对不同光照条件适应性差",
            "建议": "使用自适应阈值或Otsu方法"
        },
        "霍夫变换": {
            "rho": 1,
            "theta": "π/180",
            "threshold": 50,
            "minLineLength": 30,
            "maxLineGap": 10,
            "评估": "参数设置合理，但threshold可能需要根据图像大小调整",
            "建议": "threshold可以设为图像宽度的5-10%"
        },
        "灯条筛选": {
            "min_length": 30,
            "angle_threshold": 20,
            "评估": "角度阈值20度较宽松，可能包含非垂直线段",
            "建议": "可以缩小到10-15度，或根据实际灯条角度调整"
        }
    }
    
    for param_name, param_info in parameters.items():
        print(f"\n{param_name}:")
        for key, value in param_info.items():
            if key != "评估" and key != "建议":
                print(f"  {key}: {value}")
        print(f"  评估: {param_info['评估']}")
        print(f"  建议: {param_info['建议']}")
    
    # 3. 代码实现评估
    print("\n【3. 代码实现评估】")
    print("-" * 80)
    
    code_quality = {
        "正确性": [
            "[+] 角度计算和标准化逻辑正确",
            "[+] 边界框计算使用cv2.boundingRect，正确",
            "[+] 异常处理：检查lines是否为None"
        ],
        "问题": [
            "[-] 左右分组逻辑过于简单，仅按图像中心分割",
            "[-] 未合并相近的灯条线段（同一灯条可能被检测为多条线段）",
            "[-] 未考虑灯条的宽度信息（灯条有一定宽度，不是单条线）",
            "[-] 未与颜色分割结果结合，可能检测到背景边缘"
        ]
    }
    
    for category, points in code_quality.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  {point}")
    
    # 4. 实际测试效果评估
    print("\n【4. 实际测试效果评估】")
    print("-" * 80)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    results = []
    
    for img_path in test_images:
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                result = task3_light_bar_extraction(img_gray, img_bgr, 
                                                   output_dir=None, 
                                                   show_windows=False)
                
                # 分析检测结果
                valid_lines = result['valid_lines']
                left_bars = result['left_bars']
                right_bars = result['right_bars']
                
                # 计算统计信息
                if valid_lines:
                    lengths = [l['length'] for l in valid_lines]
                    angles = [abs(abs(l['angle']) - 90) for l in valid_lines]
                    
                    results.append({
                        'filename': os.path.basename(img_path),
                        'total_lines': len(valid_lines),
                        'left_bars': len(left_bars),
                        'right_bars': len(right_bars),
                        'avg_length': np.mean(lengths),
                        'min_length': np.min(lengths),
                        'max_length': np.max(lengths),
                        'avg_angle_deviation': np.mean(angles),
                        'max_angle_deviation': np.max(angles)
                    })
    
    # 打印测试结果
    print("\n测试图像结果统计:")
    print(f"{'图像':<30} {'总灯条':<8} {'左':<6} {'右':<6} {'平均长度':<10} {'角度偏差':<10}")
    print("-" * 80)
    
    for res in results:
        print(f"{res['filename']:<30} {res['total_lines']:<8} "
              f"{res['left_bars']:<6} {res['right_bars']:<6} "
              f"{res['avg_length']:<10.1f} {res['avg_angle_deviation']:<10.2f}")
    
    # 5. 鲁棒性评估
    print("\n【5. 鲁棒性评估】")
    print("-" * 80)
    
    robustness_analysis = {
        "光照变化": {
            "表现": "中等",
            "说明": "Canny固定阈值在不同光照下表现不稳定",
            "改进": "使用自适应阈值或结合颜色信息"
        },
        "角度变化": {
            "表现": "良好",
            "说明": "角度筛选范围20度，能适应一定角度变化",
            "改进": "可以动态调整角度阈值"
        },
        "噪声": {
            "表现": "中等",
            "说明": "未使用形态学操作，可能检测到噪声边缘",
            "改进": "对边缘检测结果进行形态学开运算"
        },
        "多目标": {
            "表现": "较差",
            "说明": "简单分组策略，无法准确区分多个装甲板",
            "改进": "使用聚类算法或基于距离的配对策略"
        }
    }
    
    for aspect, info in robustness_analysis.items():
        print(f"\n{aspect}:")
        print(f"  表现: {info['表现']}")
        print(f"  说明: {info['说明']}")
        print(f"  改进: {info['改进']}")
    
    # 6. 改进建议
    print("\n【6. 改进建议】")
    print("-" * 80)
    
    improvements = [
        {
            "优先级": "高",
            "改进点": "结合颜色信息",
            "说明": "使用第二题的颜色分割结果，只检测红色/蓝色区域内的灯条",
            "实现": "在边缘检测前应用颜色掩码"
        },
        {
            "优先级": "高",
            "改进点": "改进左右分组策略",
            "说明": "使用聚类算法或基于距离的配对，而不是简单按图像中心分割",
            "实现": "使用DBSCAN或K-means聚类，或基于距离和角度的配对算法"
        },
        {
            "优先级": "中",
            "改进点": "合并相近线段",
            "说明": "同一灯条可能被检测为多条线段，需要合并",
            "实现": "计算线段之间的距离和角度相似性，合并相近线段"
        },
        {
            "优先级": "中",
            "改进点": "自适应参数",
            "说明": "根据图像大小和特征动态调整参数",
            "实现": "threshold = image_width * 0.05, minLineLength根据图像高度调整"
        },
        {
            "优先级": "低",
            "改进点": "形态学优化",
            "说明": "对边缘检测结果进行形态学操作，去除小噪声",
            "实现": "使用开运算去除小的边缘片段"
        },
        {
            "优先级": "低",
            "改进点": "考虑灯条宽度",
            "说明": "灯条有一定宽度，可以检测灯条区域而不是单条线",
            "实现": "使用轮廓检测或区域生长算法"
        }
    ]
    
    for imp in improvements:
        print(f"\n[{imp['优先级']}优先级] {imp['改进点']}:")
        print(f"  说明: {imp['说明']}")
        print(f"  实现: {imp['实现']}")
    
    # 7. 综合评分
    print("\n【7. 综合评分】")
    print("-" * 80)
    
    scores = {
        "算法设计": {"得分": 7, "满分": 10, "说明": "基本算法合理，但缺少颜色信息结合"},
        "参数设置": {"得分": 6, "满分": 10, "说明": "参数固定，适应性较差"},
        "代码实现": {"得分": 8, "满分": 10, "说明": "代码逻辑正确，但分组策略简单"},
        "鲁棒性": {"得分": 6, "满分": 10, "说明": "对光照和噪声敏感，多目标处理不足"},
        "测试效果": {"得分": 7, "满分": 10, "说明": "基本能检测到灯条，但准确率有待提高"}
    }
    
    total_score = 0
    for aspect, score_info in scores.items():
        score = score_info['得分']
        total_score += score
        print(f"{aspect}: {score}/{score_info['满分']} - {score_info['说明']}")
    
    avg_score = total_score / len(scores)
    print(f"\n平均得分: {avg_score:.1f}/10")
    print(f"总体评价: ", end="")
    
    if avg_score >= 8:
        print("优秀 - 实现质量高，可以投入使用")
    elif avg_score >= 7:
        print("良好 - 基本满足要求，建议进行优化")
    elif avg_score >= 6:
        print("中等 - 需要改进，建议结合颜色信息和改进分组策略")
    else:
        print("需要改进 - 存在明显问题，需要重构")
    
    # 保存评估报告
    from datetime import datetime
    report = {
        "评估时间": datetime.now().isoformat(),
        "算法设计": algorithm_points,
        "参数设置": parameters,
        "代码质量": code_quality,
        "测试结果": results,
        "鲁棒性": robustness_analysis,
        "改进建议": improvements,
        "评分": {k: v['得分'] for k, v in scores.items()},
        "平均得分": avg_score
    }
    
    with open("output/task3_evaluation_report.json", "w", encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n评估报告已保存到: output/task3_evaluation_report.json")


if __name__ == "__main__":
    evaluate_task3_quality()

