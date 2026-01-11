"""
第五题改进版本：使用多种方法进行数字识别
包括：
1. 改进的模板匹配（多尺度、多角度）
2. 轮廓特征匹配（Hu矩、几何特征）
3. 基于数字结构的特征提取
4. OCR方法（可选）
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Optional

def extract_digit_features(roi_binary: np.ndarray) -> Dict:
    """
    提取数字的几何特征
    
    参数:
        roi_binary: 二值化的数字区域
    
    返回:
        特征字典
    """
    features = {}
    
    # 查找轮廓
    contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return features
    
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 1. 基本几何特征
    x, y, w, h = cv2.boundingRect(largest_contour)
    features['aspect_ratio'] = w / h if h > 0 else 0
    features['area'] = cv2.contourArea(largest_contour)
    features['perimeter'] = cv2.arcLength(largest_contour, True)
    features['extent'] = features['area'] / (w * h) if w * h > 0 else 0
    
    # 2. Hu矩（形状描述符）
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    features['hu_moments'] = hu_moments
    
    # 3. 凸包特征
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    features['solidity'] = features['area'] / hull_area if hull_area > 0 else 0
    
    # 4. 数字结构特征（简化版7段数码管特征）
    # 将数字区域分为上中下三部分
    h_third = h // 3
    top_region = roi_binary[y:y+h_third, x:x+w]
    mid_region = roi_binary[y+h_third:y+2*h_third, x:x+w]
    bottom_region = roi_binary[y+2*h_third:y+h, x:x+w]
    
    features['top_density'] = np.sum(top_region == 0) / (w * h_third) if w * h_third > 0 else 0
    features['mid_density'] = np.sum(mid_region == 0) / (w * h_third) if w * h_third > 0 else 0
    features['bottom_density'] = np.sum(bottom_region == 0) / (w * (h - 2*h_third)) if w * (h - 2*h_third) > 0 else 0
    
    # 5. 左右部分密度
    w_half = w // 2
    left_region = roi_binary[y:y+h, x:x+w_half]
    right_region = roi_binary[y:y+h, x+w_half:x+w]
    
    features['left_density'] = np.sum(left_region == 0) / (w_half * h) if w_half * h > 0 else 0
    features['right_density'] = np.sum(right_region == 0) / ((w - w_half) * h) if (w - w_half) * h > 0 else 0
    
    return features


def recognize_by_features(features: Dict) -> Tuple[str, float]:
    """
    基于特征识别数字
    
    参数:
        features: 特征字典
    
    返回:
        (识别数字, 置信度)
    """
    if not features:
        return "?", 0.0
    
    aspect_ratio = features.get('aspect_ratio', 0)
    solidity = features.get('solidity', 0)
    top_density = features.get('top_density', 0)
    mid_density = features.get('mid_density', 0)
    bottom_density = features.get('bottom_density', 0)
    left_density = features.get('left_density', 0)
    right_density = features.get('right_density', 0)
    
    # 基于特征的简单规则识别
    # 注意：这是简化版本，实际应用中需要更复杂的规则或机器学习
    
    # 数字1：很窄
    if aspect_ratio < 0.4:
        return "1", 0.7
    
    # 数字0：接近圆形，上下都有内容
    if 0.7 < aspect_ratio < 1.3 and top_density > 0.3 and bottom_density > 0.3:
        return "0", 0.6
    
    # 数字8：上下都有内容，中间也有
    if top_density > 0.3 and mid_density > 0.3 and bottom_density > 0.3:
        return "8", 0.6
    
    # 数字6：下面有内容，上面较少
    if bottom_density > 0.3 and top_density < 0.2:
        return "6", 0.5
    
    # 数字9：上面有内容，下面较少
    if top_density > 0.3 and bottom_density < 0.2:
        return "9", 0.5
    
    # 数字2：中间有内容，上下较少
    if mid_density > 0.3 and top_density < 0.2 and bottom_density < 0.2:
        return "2", 0.5
    
    # 数字3：右侧有内容
    if right_density > 0.4:
        return "3", 0.5
    
    # 数字4：中间和上面有内容
    if mid_density > 0.3 and top_density > 0.3:
        return "4", 0.5
    
    # 数字5：中间有内容，左侧较少
    if mid_density > 0.3 and left_density < 0.2:
        return "5", 0.5
    
    # 数字7：上面有内容，下面较少
    if top_density > 0.3 and bottom_density < 0.15:
        return "7", 0.5
    
    return "?", 0.3


def multi_scale_template_matching(roi_binary: np.ndarray, templates: Dict) -> Tuple[str, float]:
    """
    多尺度模板匹配
    
    参数:
        roi_binary: 二值化的数字区域
        templates: 模板字典
    
    返回:
        (识别数字, 置信度)
    """
    if not templates:
        return "?", 0.0
    
    best_match = -1
    best_confidence = 0.0
    
    # 尝试不同的缩放比例
    scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for scale in scales:
        # 缩放ROI
        h, w = roi_binary.shape
        new_w = int(w * scale)
        new_h = int(h * scale)
        roi_scaled = cv2.resize(roi_binary, (new_w, new_h))
        
        # 如果缩放后尺寸小于模板，跳过
        if new_w < 64 or new_h < 64:
            continue
        
        # 裁剪或填充到64x64
        if new_w >= 64 and new_h >= 64:
            # 裁剪中心部分
            start_x = (new_w - 64) // 2
            start_y = (new_h - 64) // 2
            roi_resized = roi_scaled[start_y:start_y+64, start_x:start_x+64]
        else:
            roi_resized = cv2.resize(roi_scaled, (64, 64))
        
        # 尝试两种二值化方向
        roi_variants = [roi_resized, cv2.bitwise_not(roi_resized)]
        
        for roi_variant in roi_variants:
            for digit, template in templates.items():
                # 模板匹配
                result = cv2.matchTemplate(roi_variant, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match = digit
    
    if best_match >= 0 and best_confidence > 0.2:
        return str(best_match), best_confidence
    else:
        return "?", 0.0


def recognize_by_contour_matching(roi_binary: np.ndarray, template_dir: str) -> Tuple[str, float]:
    """
    基于轮廓匹配的数字识别
    
    参数:
        roi_binary: 二值化的数字区域
        template_dir: 模板目录
    
    返回:
        (识别数字, 置信度)
    """
    # 提取ROI的轮廓特征
    roi_features = extract_digit_features(roi_binary)
    
    if not roi_features:
        return "?", 0.0
    
    best_match = -1
    best_similarity = 0.0
    
    # 加载模板并提取特征
    template_features = {}
    for digit in range(10):
        template_path = os.path.join(template_dir, f"digit_{digit}.jpg")
        if os.path.exists(template_path):
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                # 二值化模板
                _, template_binary = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
                template_features[digit] = extract_digit_features(template_binary)
    
    if not template_features:
        return "?", 0.0
    
    # 计算特征相似度
    for digit, tpl_features in template_features.items():
        if not tpl_features:
            continue
        
        # 计算Hu矩的相似度
        roi_hu = roi_features.get('hu_moments', np.zeros(7))
        tpl_hu = tpl_features.get('hu_moments', np.zeros(7))
        
        # Hu矩匹配（使用对数变换）
        hu_diff = np.sum(np.abs(np.log(np.abs(roi_hu) + 1e-10) - np.log(np.abs(tpl_hu) + 1e-10)))
        hu_similarity = 1.0 / (1.0 + hu_diff)
        
        # 计算宽高比相似度
        aspect_similarity = 1.0 - abs(roi_features.get('aspect_ratio', 0) - tpl_features.get('aspect_ratio', 0))
        
        # 综合相似度
        similarity = (hu_similarity * 0.6 + aspect_similarity * 0.4)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = digit
    
    if best_match >= 0 and best_similarity > 0.3:
        return str(best_match), best_similarity
    else:
        return "?", 0.0


def task5_number_recognition_improved(
    img_bgr: np.ndarray, 
    left_bars: List, 
    right_bars: List,
    output_dir: str = "output",
    template_dir: str = "digit_templates",
    method: str = "combined",
    show_windows: bool = True
) -> Dict[str, any]:
    """
    第五题改进版：装甲板数字识别
    
    支持多种识别方法：
    - "template": 模板匹配
    - "features": 特征匹配
    - "contour": 轮廓匹配
    - "combined": 组合方法（推荐）
    
    参数:
        img_bgr: BGR图像
        left_bars: 左灯条列表
        right_bars: 右灯条列表
        output_dir: 输出目录
        template_dir: 模板目录
        method: 识别方法
        show_windows: 是否显示窗口
    """
    print("=" * 60)
    print("题目5改进版：装甲板数字识别")
    print(f"使用方法: {method}")
    print("=" * 60)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. 定位数字区域（与原版相同）
    number_regions = []

    if len(left_bars) > 0 and len(right_bars) > 0:
        left_points = []
        for bar in left_bars:
            x1, y1, x2, y2 = bar['line']
            left_points.extend([(x1, y1), (x2, y2)])

        right_points = []
        for bar in right_bars:
            x1, y1, x2, y2 = bar['line']
            right_points.extend([(x1, y1), (x2, y2)])

        if left_points and right_points:
            left_points = np.array(left_points)
            right_points = np.array(right_points)

            left_x, left_y, left_w, left_h = cv2.boundingRect(left_points)
            right_x, right_y, right_w, right_h = cv2.boundingRect(right_points)

            number_x = left_x + left_w
            number_y = min(left_y, right_y)
            number_w = right_x - number_x
            number_h = max(left_h, right_h)

            if number_w > 0 and number_h > 0:
                number_regions.append({
                    'bbox': (number_x, number_y, number_w, number_h),
                    'roi': img_bgr[number_y:number_y + number_h, number_x:number_x + number_w]
                })

    if not number_regions:
        print("警告：未能定位到数字区域")
        return {"recognized_numbers": [], "result_image": img_bgr}

    # 2. 加载模板
    templates = {}
    if os.path.isdir(template_dir):
        for digit in range(10):
            template_path = os.path.join(template_dir, f"digit_{digit}.jpg")
            if os.path.exists(template_path):
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    templates[digit] = template
        
        if templates:
            print(f"加载了 {len(templates)} 个数字模板")

    # 3. 数字识别
    img_result = img_bgr.copy()
    recognized_numbers = []

    for region_info in number_regions:
        x, y, w, h = region_info['bbox']
        roi = region_info['roi']

        # 预处理
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, kernel)

        recognized_digit = "?"
        confidence = 0.0
        method_used = ""

        # 根据方法选择识别算法
        if method == "template":
            # 方法1：模板匹配
            recognized_digit, confidence = multi_scale_template_matching(roi_binary, templates)
            method_used = "模板匹配"
            
        elif method == "features":
            # 方法2：特征匹配
            features = extract_digit_features(roi_binary)
            recognized_digit, confidence = recognize_by_features(features)
            method_used = "特征匹配"
            
        elif method == "contour":
            # 方法3：轮廓匹配
            recognized_digit, confidence = recognize_by_contour_matching(roi_binary, template_dir)
            method_used = "轮廓匹配"
            
        elif method == "combined":
            # 方法4：组合方法（投票机制）
            results = []
            
            # 模板匹配
            if templates:
                digit1, conf1 = multi_scale_template_matching(roi_binary, templates)
                if digit1 != "?":
                    results.append((digit1, conf1, 1.0))  # 权重1.0
            
            # 特征匹配
            features = extract_digit_features(roi_binary)
            digit2, conf2 = recognize_by_features(features)
            if digit2 != "?":
                results.append((digit2, conf2, 0.8))  # 权重0.8
            
            # 轮廓匹配
            digit3, conf3 = recognize_by_contour_matching(roi_binary, template_dir)
            if digit3 != "?":
                results.append((digit3, conf3, 0.7))  # 权重0.7
            
            # 投票选择最佳结果
            if results:
                # 按数字分组，计算加权平均置信度
                digit_scores = {}
                for digit, conf, weight in results:
                    if digit not in digit_scores:
                        digit_scores[digit] = []
                    digit_scores[digit].append(conf * weight)
                
                # 选择得分最高的数字
                best_digit = "?"
                best_score = 0.0
                for digit, scores in digit_scores.items():
                    avg_score = np.mean(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_digit = digit
                
                recognized_digit = best_digit
                confidence = best_score
                method_used = f"组合方法({len(results)}种)"
            else:
                method_used = "组合方法(无结果)"

        recognized_numbers.append({
            'digit': recognized_digit,
            'confidence': confidence,
            'bbox': (x, y, w, h),
            'method': method_used
        })

        # 绘制结果
        color = (0, 255, 0) if recognized_digit != "?" else (0, 0, 255)
        cv2.rectangle(img_result, (x, y), (x + w, y + h), color, 2)
        label = f"{recognized_digit} ({confidence:.2f})"
        cv2.putText(img_result, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 显示结果
    if show_windows:
        cv2.imshow("数字识别结果（改进版）", img_result)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 保存结果
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, f"task5_improved_{method}.jpg"), img_result)

    print(f"识别到 {len(recognized_numbers)} 个数字区域")
    for num_info in recognized_numbers:
        print(f"  数字: {num_info['digit']}, 置信度: {num_info['confidence']:.2f}, "
              f"方法: {num_info['method']}")

    return {
        "recognized_numbers": recognized_numbers,
        "result_image": img_result
    }

