"""
第六题改进版本：装甲板轮廓匹配与识别
主要改进：
1. 降低数字置信度阈值（更灵活）
2. 改进灯条配对算法（更智能的配对策略）
3. 添加装甲板评分机制（综合多个因素）
4. 改进颜色判断（更准确）
5. 支持无数字验证模式（仅基于灯条几何特征）
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional

def calculate_pair_score(left_bar: Dict, right_bar: Dict, 
                         recognized_numbers: List, 
                         mask_red: np.ndarray, mask_blue: np.ndarray) -> Tuple[float, Dict]:
    """
    计算灯条配对的综合评分
    
    参数:
        left_bar: 左灯条信息
        right_bar: 右灯条信息
        recognized_numbers: 数字识别结果列表
        mask_red: 红色掩码
        mask_blue: 蓝色掩码
    
    返回:
        (评分, 详细信息字典)
    """
    score = 0.0
    details = {
        'angle_score': 0.0,
        'length_score': 0.0,
        'distance_score': 0.0,
        'vertical_score': 0.0,
        'number_score': 0.0,
        'color_score': 0.0
    }
    
    # 1. 角度相似性评分（0-30分）
    angle_diff = abs(left_bar['angle'] - right_bar['angle'])
    if angle_diff <= 5:
        details['angle_score'] = 30.0
    elif angle_diff <= 10:
        details['angle_score'] = 20.0
    elif angle_diff <= 15:
        details['angle_score'] = 10.0
    else:
        details['angle_score'] = 0.0
    
    # 2. 长度比例评分（0-20分）
    length_ratio = min(left_bar['length'], right_bar['length']) / \
                   max(left_bar['length'], right_bar['length'])
    if length_ratio >= 0.9:
        details['length_score'] = 20.0
    elif length_ratio >= 0.8:
        details['length_score'] = 15.0
    elif length_ratio >= 0.7:
        details['length_score'] = 10.0
    elif length_ratio >= 0.6:
        details['length_score'] = 5.0
    else:
        details['length_score'] = 0.0
    
    # 3. 间距评分（0-20分）
    left_center_x = left_bar['center'][0]
    right_center_x = right_bar['center'][0]
    distance = abs(right_center_x - left_center_x)
    avg_length = (left_bar['length'] + right_bar['length']) / 2
    
    if avg_length > 0:
        distance_ratio = distance / avg_length
        if 1.0 <= distance_ratio <= 2.5:
            details['distance_score'] = 20.0
        elif 0.8 <= distance_ratio <= 3.0:
            details['distance_score'] = 15.0
        elif 0.5 <= distance_ratio <= 3.5:
            details['distance_score'] = 10.0
        else:
            details['distance_score'] = 0.0
    
    # 4. 垂直位置评分（0-10分）
    left_center_y = left_bar['center'][1]
    right_center_y = right_bar['center'][1]
    y_diff = abs(left_center_y - right_center_y)
    
    if avg_length > 0:
        y_ratio = y_diff / avg_length
        if y_ratio <= 0.2:
            details['vertical_score'] = 10.0
        elif y_ratio <= 0.3:
            details['vertical_score'] = 7.0
        elif y_ratio <= 0.5:
            details['vertical_score'] = 5.0
        else:
            details['vertical_score'] = 0.0
    
    # 5. 数字验证评分（0-15分）
    number_digit = None
    number_confidence = 0.0
    
    for num_info in recognized_numbers:
        num_x, num_y, num_w, num_h = num_info['bbox']
        num_center_x = num_x + num_w // 2
        
        # 检查数字是否在灯条之间
        if left_center_x < num_center_x < right_center_x:
            # 降低置信度阈值，更灵活
            conf = num_info['confidence']
            if conf > 0.3:  # 降低阈值到0.3
                details['number_score'] = min(15.0, conf * 15.0)
                number_digit = num_info['digit']
                number_confidence = conf
                break
            elif conf > 0.2:  # 即使置信度较低也给一些分数
                details['number_score'] = min(10.0, conf * 10.0)
                number_digit = num_info['digit']
                number_confidence = conf
    
    # 6. 颜色一致性评分（0-5分）
    # 计算灯条区域的颜色
    left_x1, left_y1, left_x2, left_y2 = left_bar['line']
    right_x1, right_y1, right_x2, right_y2 = right_bar['line']
    
    armor_x = min(left_x1, left_x2)
    armor_y = min(left_y1, left_y2, right_y1, right_y2)
    armor_x2 = max(right_x1, right_x2)
    armor_y2 = max(left_y1, left_y2, right_y1, right_y2)
    
    if armor_x2 > armor_x and armor_y2 > armor_y:
        armor_roi_red = mask_red[armor_y:armor_y2, armor_x:armor_x2]
        armor_roi_blue = mask_blue[armor_y:armor_y2, armor_x:armor_x2]
        
        red_pixels = np.sum(armor_roi_red > 0)
        blue_pixels = np.sum(armor_roi_blue > 0)
        total_pixels = (armor_x2 - armor_x) * (armor_y2 - armor_y)
        
        if total_pixels > 0:
            color_ratio = max(red_pixels, blue_pixels) / total_pixels
            if color_ratio > 0.3:  # 至少30%的区域有颜色
                details['color_score'] = 5.0
            elif color_ratio > 0.2:
                details['color_score'] = 3.0
    
    # 计算总分
    score = sum(details.values())
    details['total_score'] = score
    details['number_digit'] = number_digit
    details['number_confidence'] = number_confidence
    
    return score, details


def task6_armor_detection_improved(
    img_bgr: np.ndarray, 
    left_bars: List, 
    right_bars: List,
    mask_red: np.ndarray, 
    mask_blue: np.ndarray,
    recognized_numbers: List, 
    output_dir: str = "output",
    min_score: float = 50.0,
    require_number: bool = False,
    show_windows: bool = True
) -> Dict[str, any]:
    """
    第六题改进版：装甲板轮廓匹配与识别
    
    主要改进：
    1. 使用评分机制，更灵活地判断装甲板
    2. 降低数字置信度阈值
    3. 支持仅基于几何特征的识别（无数字验证）
    4. 改进颜色判断
    
    参数:
        img_bgr: BGR图像
        left_bars: 左灯条列表
        right_bars: 右灯条列表
        mask_red: 红色掩码
        mask_blue: 蓝色掩码
        recognized_numbers: 数字识别结果列表
        output_dir: 输出目录
        min_score: 最小评分阈值（默认50.0）
        require_number: 是否要求数字验证（默认False）
        show_windows: 是否显示窗口
    """
    print("=" * 60)
    print("题目6改进版：装甲板轮廓匹配与识别")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    img_result = img_bgr.copy()
    valid_armors = []
    all_pairs = []

    # 1. 对所有灯条配对进行评分
    for left_bar in left_bars:
        for right_bar in right_bars:
            score, details = calculate_pair_score(
                left_bar, right_bar, recognized_numbers, mask_red, mask_blue
            )
            
            # 如果要求数字验证，检查是否有数字
            if require_number and details['number_digit'] is None:
                continue
            
            # 如果评分达到阈值，添加到候选列表
            if score >= min_score:
                all_pairs.append({
                    'left_bar': left_bar,
                    'right_bar': right_bar,
                    'score': score,
                    'details': details
                })

    # 2. 按评分排序，选择最佳配对
    all_pairs.sort(key=lambda x: x['score'], reverse=True)
    
    # 3. 避免重复检测（如果多个配对指向同一个装甲板）
    used_left = set()
    used_right = set()
    
    for pair in all_pairs:
        left_bar = pair['left_bar']
        right_bar = pair['right_bar']
        
        # 检查是否已被使用
        left_id = id(left_bar)
        right_id = id(right_bar)
        
        if left_id in used_left or right_id in used_right:
            continue
        
        # 标记为已使用
        used_left.add(left_id)
        used_right.add(right_id)
        
        # 计算装甲板边界框
        left_x1, left_y1, left_x2, left_y2 = left_bar['line']
        right_x1, right_y1, right_x2, right_y2 = right_bar['line']
        
        armor_x = min(left_x1, left_x2)
        armor_y = min(left_y1, left_y2, right_y1, right_y2)
        armor_x2 = max(right_x1, right_x2)
        armor_y2 = max(left_y1, left_y2, right_y1, right_y2)
        
        armor_w = armor_x2 - armor_x
        armor_h = armor_y2 - armor_y
        
        # 颜色判断
        armor_roi_red = mask_red[armor_y:armor_y2, armor_x:armor_x2]
        armor_roi_blue = mask_blue[armor_y:armor_y2, armor_x:armor_x2]
        
        red_pixels = np.sum(armor_roi_red > 0)
        blue_pixels = np.sum(armor_roi_blue > 0)
        
        armor_color = "red" if red_pixels > blue_pixels else "blue"
        
        # 获取数字信息
        number_digit = pair['details']['number_digit'] or "?"
        number_confidence = pair['details']['number_confidence']
        
        valid_armors.append({
            'bbox': (armor_x, armor_y, armor_w, armor_h),
            'left_bar': left_bar,
            'right_bar': right_bar,
            'number': number_digit,
            'number_confidence': number_confidence,
            'color': armor_color,
            'score': pair['score'],
            'details': pair['details']
        })
        
        # 绘制装甲板
        color_bgr = (0, 0, 255) if armor_color == "red" else (255, 0, 0)
        cv2.rectangle(img_result, (armor_x, armor_y),
                      (armor_x2, armor_y2), color_bgr, 3)
        
        # 标注信息
        if number_digit != "?":
            label = f"{armor_color.upper()} {number_digit} ({number_confidence:.2f})"
        else:
            label = f"{armor_color.upper()} (Score:{pair['score']:.1f})"
        
        cv2.putText(img_result, label, (armor_x, armor_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)

    # 显示结果
    if show_windows:
        cv2.imshow("装甲板识别结果（改进版）", img_result)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite(os.path.join(output_dir, "task6_armor_detection_improved.jpg"), img_result)

    print(f"检测到 {len(valid_armors)} 个有效装甲板")
    for i, armor in enumerate(valid_armors):
        print(f"  装甲板 {i + 1}: {armor['color']} {armor['number']}, "
              f"评分: {armor['score']:.1f}, 数字置信度: {armor['number_confidence']:.2f}")

    return {
        "valid_armors": valid_armors,
        "result_image": img_result
    }

