"""
第三题改进版本：装甲板灯条提取
主要改进：
1. 结合颜色信息（使用第二题的颜色分割结果）
2. 改进左右分组策略（使用聚类算法）
3. 合并相近线段
4. 自适应参数调整
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import os


def merge_nearby_lines(lines: List[Dict], distance_threshold: float = 15.0, 
                       angle_threshold: float = 5.0) -> List[Dict]:
    """
    合并相近的线段
    
    参数:
        lines: 线段列表，每个元素包含 'line', 'length', 'angle', 'center'
        distance_threshold: 距离阈值（像素）
        angle_threshold: 角度阈值（度）
    
    返回:
        合并后的线段列表
    """
    if not lines:
        return []
    
    merged = []
    used = [False] * len(lines)
    
    for i, line1 in enumerate(lines):
        if used[i]:
            continue
        
        # 找到与当前线段相近的所有线段
        group = [line1]
        used[i] = True
        
        for j, line2 in enumerate(lines):
            if used[j] or i == j:
                continue
            
            # 计算两条线段中心点之间的距离
            dist = np.sqrt((line1['center'][0] - line2['center'][0])**2 + 
                          (line1['center'][1] - line2['center'][1])**2)
            
            # 计算角度差
            angle_diff = abs(line1['angle'] - line2['angle'])
            
            # 如果距离和角度都相近，合并
            if dist < distance_threshold and angle_diff < angle_threshold:
                group.append(line2)
                used[j] = True
        
        # 合并组内的线段
        if len(group) > 1:
            # 计算合并后的线段（取所有端点的边界框）
            all_points = []
            total_length = 0
            avg_angle = 0
            
            for line_info in group:
                x1, y1, x2, y2 = line_info['line']
                all_points.extend([(x1, y1), (x2, y2)])
                total_length += line_info['length']
                avg_angle += line_info['angle']
            
            all_points = np.array(all_points)
            x_min, y_min = all_points.min(axis=0)
            x_max, y_max = all_points.max(axis=0)
            
            # 创建合并后的线段（垂直方向）
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            length = y_max - y_min
            
            merged_line = {
                'line': (center_x, y_min, center_x, y_max),
                'length': length,
                'angle': avg_angle / len(group),
                'center': (center_x, center_y)
            }
        else:
            merged_line = group[0]
        
        merged.append(merged_line)
    
    return merged


def cluster_bars(lines: List[Dict], image_width: int) -> Tuple[List[Dict], List[Dict]]:
    """
    使用简单的聚类方法将灯条分为左右两组
    
    参数:
        lines: 线段列表
        image_width: 图像宽度
    
    返回:
        (left_bars, right_bars)
    """
    if not lines:
        return [], []
    
    # 按x坐标排序
    sorted_lines = sorted(lines, key=lambda l: l['center'][0])
    
    # 如果只有一条线段，根据位置判断
    if len(sorted_lines) == 1:
        if sorted_lines[0]['center'][0] < image_width // 2:
            return sorted_lines, []
        else:
            return [], sorted_lines
    
    # 使用K-means简单实现（K=2）
    # 或者使用基于距离的方法
    
    # 方法1：找到x坐标的中间点作为分界
    x_coords = [l['center'][0] for l in sorted_lines]
    
    # 如果x坐标分布有明显的分界点，使用分界点
    # 否则使用图像中心
    if len(x_coords) >= 2:
        # 计算相邻x坐标的差值
        gaps = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        max_gap_idx = np.argmax(gaps)
        
        # 如果最大间隙足够大，使用间隙中点作为分界
        if gaps[max_gap_idx] > image_width * 0.15:  # 间隙大于图像宽度的15%
            threshold = (x_coords[max_gap_idx] + x_coords[max_gap_idx + 1]) // 2
        else:
            threshold = image_width // 2
    else:
        threshold = image_width // 2
    
    left_bars = [l for l in sorted_lines if l['center'][0] < threshold]
    right_bars = [l for l in sorted_lines if l['center'][0] >= threshold]
    
    return left_bars, right_bars


def task3_light_bar_extraction_improved(
    img_gray: np.ndarray, 
    img_bgr: np.ndarray,
    mask_red: Optional[np.ndarray] = None,
    mask_blue: Optional[np.ndarray] = None,
    output_dir: str = "output", 
    show_windows: bool = True
) -> Dict[str, any]:
    """
    题目3改进版：装甲板灯条提取
    
    主要改进：
    1. 结合颜色信息：只检测红色/蓝色区域内的灯条
    2. 合并相近线段：避免同一灯条被检测为多条线段
    3. 改进分组策略：使用基于距离的聚类方法
    4. 自适应参数：根据图像大小调整参数
    
    参数:
        img_gray: 灰度图像
        img_bgr: BGR格式的原始图像
        mask_red: 红色掩码（可选，如果提供则只检测红色区域）
        mask_blue: 蓝色掩码（可选，如果提供则只检测蓝色区域）
        output_dir: 输出目录
        show_windows: 是否显示窗口
    """
    print("=" * 60)
    print("题目3改进版：装甲板灯条提取")
    print("=" * 60)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    h, w = img_gray.shape
    
    # 自适应参数调整（稍微放宽，避免过度严格）
    canny_low = max(40, int(w * 0.04))  # 稍微提高低阈值
    canny_high = canny_low * 2.5  # 降低比例，避免阈值过高
    hough_threshold = max(25, int(w * 0.04))  # 稍微降低阈值
    min_line_length = max(15, int(h * 0.04))  # 稍微降低最小长度
    
    print(f"自适应参数: Canny({canny_low}, {canny_high}), "
          f"Hough threshold={hough_threshold}, min_length={min_line_length}")

    # 1. 边缘检测（Canny算法）
    edges = cv2.Canny(img_gray, canny_low, canny_high)
    
    # 2. 如果提供了颜色掩码，只检测颜色区域内的边缘
    edges_with_color = edges.copy()
    if mask_red is not None or mask_blue is not None:
        # 合并红色和蓝色掩码
        if mask_red is not None and mask_blue is not None:
            color_mask = cv2.bitwise_or(mask_red, mask_blue)
        elif mask_red is not None:
            color_mask = mask_red
        else:
            color_mask = mask_blue
        
        # 只保留颜色区域内的边缘
        edges_with_color = cv2.bitwise_and(edges, color_mask)
        
        # 形态学操作去除小噪声（使用较小的核，避免过度去除）
        kernel = np.ones((2, 2), np.uint8)
        edges_with_color = cv2.morphologyEx(edges_with_color, cv2.MORPH_CLOSE, kernel)
    
    # 使用颜色过滤后的边缘（如果可用），否则使用原始边缘
    edges_final = edges_with_color if (mask_red is not None or mask_blue is not None) else edges
    
    # 3. 霍夫直线检测
    lines = cv2.HoughLinesP(
        edges_final,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=10
    )

    # 4. 筛选符合灯条要求的直线
    valid_lines = []
    img_with_lines = img_bgr.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算线段长度
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 计算角度（转换为度数）
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # 标准化角度到[-90, 90]
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180

            # 筛选条件：长度和角度
            angle_threshold = 15  # 稍微缩小角度阈值，提高精度
            if length > min_line_length and abs(abs(angle) - 90) < angle_threshold:
                valid_lines.append({
                    'line': line[0],
                    'length': length,
                    'angle': angle,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })

                # 绘制检测到的直线
                cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print(f"初步检测到 {len(valid_lines)} 条有效灯条")

    # 5. 合并相近的线段
    if len(valid_lines) > 1:
        merged_lines = merge_nearby_lines(valid_lines, distance_threshold=15.0, angle_threshold=5.0)
        print(f"合并后剩余 {len(merged_lines)} 条灯条")
    else:
        merged_lines = valid_lines

    # 6. 灯条分组（左右灯条）
    left_bars, right_bars = cluster_bars(merged_lines, w)

    print(f"左灯条: {len(left_bars)}, 右灯条: {len(right_bars)}")

    # 7. 绘制灯条区域
    img_with_bars = img_bgr.copy()

    # 绘制左灯条区域
    if left_bars:
        left_points = []
        for bar in left_bars:
            x1, y1, x2, y2 = bar['line']
            left_points.extend([(x1, y1), (x2, y2)])
        if left_points:
            left_points = np.array(left_points)
            x, y, w_rect, h_rect = cv2.boundingRect(left_points)
            cv2.rectangle(img_with_bars, (x, y), (x + w_rect, y + h_rect), (255, 0, 0), 2)
            cv2.putText(img_with_bars, f"Left Bar ({len(left_bars)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 绘制右灯条区域
    if right_bars:
        right_points = []
        for bar in right_bars:
            x1, y1, x2, y2 = bar['line']
            right_points.extend([(x1, y1), (x2, y2)])
        if right_points:
            right_points = np.array(right_points)
            x, y, w_rect, h_rect = cv2.boundingRect(right_points)
            cv2.rectangle(img_with_bars, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 2)
            cv2.putText(img_with_bars, f"Right Bar ({len(right_bars)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 8. 显示结果（可选）
    if show_windows:
        cv2.imshow("边缘检测", edges)
        cv2.imshow("检测到的直线", img_with_lines)
        cv2.imshow("灯条区域（改进版）", img_with_bars)

        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 9. 保存结果
    if output_dir:
        cv2.imwrite(os.path.join(output_dir, "task3_improved_edges.jpg"), edges_final)
        cv2.imwrite(os.path.join(output_dir, "task3_improved_lines.jpg"), img_with_lines)
        cv2.imwrite(os.path.join(output_dir, "task3_improved_light_bars.jpg"), img_with_bars)

    return {
        "edges": edges_final,
        "edges_original": edges,  # 保留原始边缘用于对比
        "valid_lines": merged_lines,  # 返回合并后的线段
        "left_bars": left_bars,
        "right_bars": right_bars,
        "result_image": img_with_bars,
        "original_valid_lines": valid_lines  # 保留原始检测结果用于对比
    }

