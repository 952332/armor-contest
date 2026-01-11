"""
综合评估所有改进题目的效果
"""

import os
import json
import cv2
import numpy as np
from armor import (
    task1_image_preprocessing,
    task2_color_segmentation,
    task3_light_bar_extraction,
    task4_camera_calibration,
    task5_number_recognition
)
from task6_improved import task6_armor_detection_improved
from task7_improved import task7_pose_estimation_improved

def evaluate_all_improvements():
    """综合评估所有改进题目的效果"""
    
    print("=" * 80)
    print("所有改进题目综合效果评估")
    print("=" * 80)
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    template_dir = "digit_templates"
    calibration_dir = "calibration_images"
    output_dir = "output"
    
    all_results = {
        'task4': {'calibration': None},
        'task5': {'results': []},
        'task6': {'results': []},
        'task7': {'results': []}
    }
    
    # 1. 相机标定（Task 4）
    print("\n" + "=" * 80)
    print("【题目4：相机标定与畸变矫正】")
    print("=" * 80)
    
    if os.path.isdir(calibration_dir):
        result4 = task4_camera_calibration(
            calibration_dir,
            img_bgr=None,
            output_dir=output_dir,
            show_windows=False
        )
        all_results['task4']['calibration'] = {
            'has_calibration': True,
            'reprojection_error': result4.get('reprojection_error'),
            'validation': result4.get('validation')
        }
        
        print(f"重投影误差: {result4.get('reprojection_error', 'N/A')}")
        if result4.get('validation'):
            val = result4['validation']
            print(f"焦距验证: {'OK' if val['focal_length']['valid'] else 'FAIL'}")
            print(f"主点验证: {'OK' if val['principal_point']['valid'] else 'FAIL'}")
    else:
        all_results['task4']['calibration'] = {'has_calibration': False}
        print("未找到标定图像")
    
    # 2-7. 对每张图像进行完整流程测试
    print("\n" + "=" * 80)
    print("【完整流程测试】")
    print("=" * 80)
    
    for img_path in test_images:
        if not os.path.exists(img_path):
            continue
        
        img_name = os.path.basename(img_path)
        print(f"\n处理图像: {img_name}")
        
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
            result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir=None, show_windows=False)
            left_bars = result3["left_bars"]
            right_bars = result3["right_bars"]
            
            if len(left_bars) == 0 or len(right_bars) == 0:
                print(f"  跳过: 未检测到足够的灯条")
                continue
            
            # 数字识别（Task 5）
            result5 = task5_number_recognition(
                img_bgr, left_bars, right_bars,
                output_dir="output",
                template_dir=template_dir,
                show_windows=False
            )
            recognized_numbers = result5["recognized_numbers"]
            
            task5_result = {
                'image': img_name,
                'numbers_detected': len(recognized_numbers),
                'numbers': [{'digit': n['digit'], 'conf': n['confidence']} for n in recognized_numbers]
            }
            all_results['task5']['results'].append(task5_result)
            
            # 装甲板检测（Task 6）
            result6 = task6_armor_detection_improved(
                img_bgr, left_bars, right_bars,
                mask_red, mask_blue,
                recognized_numbers,
                output_dir=output_dir,
                min_score=50.0,
                require_number=False,
                show_windows=False
            )
            valid_armors = result6["valid_armors"]
            
            task6_result = {
                'image': img_name,
                'armors_detected': len(valid_armors),
                'armors': []
            }
            
            # 位姿解算（Task 7）
            task7_result = {
                'image': img_name,
                'pose_estimated': 0,
                'poses': []
            }
            
            if len(valid_armors) > 0:
                # 获取相机内参
                if all_results['task4']['calibration'] and all_results['task4']['calibration']['has_calibration']:
                    camera_matrix = result4["camera_matrix"]
                    dist_coeffs = result4["dist_coeffs"]
                else:
                    camera_matrix = np.array([[800, 0, 320], 
                                             [0, 800, 240], 
                                             [0, 0, 1]], dtype=np.float32)
                    dist_coeffs = np.zeros((4, 1))
                
                for armor in valid_armors:
                    armor_info = {
                        'color': armor['color'],
                        'number': armor['number'],
                        'score': armor['score'],
                        'number_confidence': armor['number_confidence']
                    }
                    task6_result['armors'].append(armor_info)
                    
                    # 位姿解算
                    result7 = task7_pose_estimation_improved(
                        armor,
                        camera_matrix,
                        dist_coeffs,
                        output_dir=output_dir,
                        visualize=False,
                        img_bgr=None
                    )
                    
                    if result7:
                        task7_result['pose_estimated'] += 1
                        pose_info = {
                            'distance_m': result7['distance_m'],
                            'euler_angles': result7['euler_angles'].tolist(),
                            'reprojection_error': result7['reprojection_error'],
                            'method': result7['method']
                        }
                        task7_result['poses'].append(pose_info)
            
            all_results['task6']['results'].append(task6_result)
            all_results['task7']['results'].append(task7_result)
            
        except Exception as e:
            print(f"  处理失败: {e}")
    
    # 生成综合报告
    print("\n" + "=" * 80)
    print("【综合效果评估】")
    print("=" * 80)
    
    # Task 4 评估
    print("\n【题目4：相机标定】")
    if all_results['task4']['calibration'] and all_results['task4']['calibration']['has_calibration']:
        rep_error = all_results['task4']['calibration']['reprojection_error']
        if rep_error:
            print(f"  重投影误差: {rep_error:.4f} 像素")
            if rep_error < 0.5:
                print(f"  质量评估: [EXCELLENT] 优秀")
            elif rep_error < 1.0:
                print(f"  质量评估: [GOOD] 良好")
            else:
                print(f"  质量评估: [FAIR] 一般")
    else:
        print("  状态: 未进行标定")
    
    # Task 5 评估
    print("\n【题目5：数字识别】")
    task5_results = all_results['task5']['results']
    if task5_results:
        total_numbers = sum(r['numbers_detected'] for r in task5_results)
        successful_numbers = sum(1 for r in task5_results 
                                for n in r['numbers'] if n['digit'] != '?')
        avg_confidence = np.mean([n['conf'] for r in task5_results 
                                 for n in r['numbers'] if n['digit'] != '?'])
        
        print(f"  识别到的数字区域: {total_numbers}")
        print(f"  成功识别数字: {successful_numbers}")
        print(f"  平均置信度: {avg_confidence:.3f}")
        
        if avg_confidence > 0.5:
            print(f"  质量评估: [GOOD] 良好")
        elif avg_confidence > 0.3:
            print(f"  质量评估: [FAIR] 一般")
        else:
            print(f"  质量评估: [NEEDS_IMPROVEMENT] 需要改进")
    
    # Task 6 评估
    print("\n【题目6：装甲板检测】")
    task6_results = all_results['task6']['results']
    if task6_results:
        total_armors = sum(r['armors_detected'] for r in task6_results)
        success_rate = len([r for r in task6_results if r['armors_detected'] > 0]) / len(task6_results) * 100
        avg_scores = [a['score'] for r in task6_results for a in r['armors']]
        
        print(f"  检测到的装甲板总数: {total_armors}")
        print(f"  检测成功率: {success_rate:.1f}%")
        if avg_scores:
            print(f"  平均评分: {np.mean(avg_scores):.1f}/100")
            print(f"  最高评分: {max(avg_scores):.1f}/100")
            print(f"  最低评分: {min(avg_scores):.1f}/100")
            
            if success_rate == 100 and np.mean(avg_scores) >= 80:
                print(f"  质量评估: [EXCELLENT] 优秀")
            elif success_rate >= 75 and np.mean(avg_scores) >= 70:
                print(f"  质量评估: [GOOD] 良好")
            else:
                print(f"  质量评估: [FAIR] 一般")
    
    # Task 7 评估
    print("\n【题目7：位姿解算】")
    task7_results = all_results['task7']['results']
    if task7_results:
        total_poses = sum(r['pose_estimated'] for r in task7_results)
        reproj_errors = [p['reprojection_error'] for r in task7_results 
                        for p in r['poses']]
        
        print(f"  成功解算的位姿数: {total_poses}")
        if reproj_errors:
            print(f"  平均重投影误差: {np.mean(reproj_errors):.4f} 像素")
            print(f"  最大重投影误差: {max(reproj_errors):.4f} 像素")
            print(f"  最小重投影误差: {min(reproj_errors):.4f} 像素")
            
            if np.mean(reproj_errors) < 1.0:
                print(f"  质量评估: [EXCELLENT] 优秀")
            elif np.mean(reproj_errors) < 2.0:
                print(f"  质量评估: [GOOD] 良好")
            else:
                print(f"  质量评估: [FAIR] 一般")
    
    # 整体评估
    print("\n" + "=" * 80)
    print("【整体评估】")
    print("=" * 80)
    
    improvements_summary = {
        'task4': {
            'name': '相机标定',
            'improvements': [
                '重投影误差计算',
                '标定质量验证',
                '支持真实棋盘格尺寸'
            ],
            'status': 'completed'
        },
        'task5': {
            'name': '数字识别',
            'improvements': [
                '多尺度模板匹配',
                '特征匹配',
                '轮廓匹配',
                '组合方法'
            ],
            'status': 'completed'
        },
        'task6': {
            'name': '装甲板检测',
            'improvements': [
                '评分机制',
                '降低数字置信度阈值',
                '支持无数字验证模式',
                '改进灯条配对算法'
            ],
            'status': 'completed'
        },
        'task7': {
            'name': '位姿解算',
            'improvements': [
                '使用灯条中心点',
                '重投影误差验证',
                '改进距离计算',
                '位姿可视化'
            ],
            'status': 'completed'
        }
    }
    
    print("\n改进题目总结:")
    for task_id, info in improvements_summary.items():
        print(f"\n{info['name']} ({task_id}):")
        for imp in info['improvements']:
            print(f"  [OK] {imp}")
    
    # 转换numpy类型为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    # 保存综合报告
    report_path = os.path.join(output_dir, "all_improvements_summary.json")
    serializable_results = convert_to_serializable(all_results)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': improvements_summary,
            'results': serializable_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n综合报告已保存: {report_path}")
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_all_improvements()

