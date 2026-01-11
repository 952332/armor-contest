"""
第六题可视化：装甲板轮廓匹配与识别
展示完整的检测流程和结果
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

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib未安装，将使用OpenCV进行可视化")

def visualize_task6(image_path: str, output_dir: str = "output", 
                     template_dir: str = "digit_templates",
                     save_individual: bool = True):
    """
    可视化第六题的检测过程
    
    参数:
        image_path: 输入图像路径
        output_dir: 输出目录
        template_dir: 数字模板目录
        save_individual: 是否保存单独的图像
    """
    
    print("=" * 80)
    print("第六题可视化：装甲板轮廓匹配与识别")
    print("=" * 80)
    
    if not os.path.exists(image_path):
        print(f"错误：图像文件不存在: {image_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 图像预处理
    print("\n步骤1: 图像预处理")
    result1 = task1_image_preprocessing(image_path, output_dir)
    img_bgr = result1["original_bgr"]
    img_gray = result1["gray"]
    
    # 2. 颜色分割
    print("\n步骤2: 颜色分割")
    result2 = task2_color_segmentation(img_bgr, output_dir, show_windows=False)
    mask_red = result2["mask_red"]
    mask_blue = result2["mask_blue"]
    
    # 3. 灯条提取
    print("\n步骤3: 灯条提取")
    result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir, show_windows=False)
    left_bars = result3["left_bars"]
    right_bars = result3["right_bars"]
    
    # 4. 数字识别
    print("\n步骤4: 数字识别")
    result5 = task5_number_recognition(
        img_bgr, left_bars, right_bars,
        output_dir=output_dir,
        template_dir=template_dir,
        show_windows=False
    )
    recognized_numbers = result5["recognized_numbers"]
    
    # 5. 装甲板检测（改进版）
    print("\n步骤5: 装甲板检测")
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
    
    # 创建可视化图像
    h, w = img_bgr.shape[:2]
    
    if HAS_MATPLOTLIB:
        # 使用matplotlib创建详细可视化
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 原始图像
        ax1 = plt.subplot(2, 3, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_rgb)
        ax1.set_title("1. 原始图像", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. 颜色分割结果
        ax2 = plt.subplot(2, 3, 2)
        mask_combined = cv2.bitwise_or(mask_red, mask_blue)
        mask_colored = np.zeros_like(img_rgb)
        mask_colored[mask_red > 0] = [255, 0, 0]  # 红色
        mask_colored[mask_blue > 0] = [0, 0, 255]  # 蓝色
        overlay = cv2.addWeighted(img_rgb, 0.6, mask_colored, 0.4, 0)
        ax2.imshow(overlay)
        ax2.set_title("2. 颜色分割结果", fontsize=14, fontweight='bold')
        ax2.axis('off')
        
        # 3. 灯条检测结果
        ax3 = plt.subplot(2, 3, 3)
        img_bars = img_rgb.copy()
        for bar in left_bars:
            x1, y1, x2, y2 = bar['line']
            cv2.line(img_bars, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.circle(img_bars, bar['center'], 5, (255, 0, 0), -1)
        for bar in right_bars:
            x1, y1, x2, y2 = bar['line']
            cv2.line(img_bars, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.circle(img_bars, bar['center'], 5, (0, 0, 255), -1)
        ax3.imshow(img_bars)
        ax3.set_title(f"3. 灯条检测 (左:{len(left_bars)}, 右:{len(right_bars)})", 
                      fontsize=14, fontweight='bold')
        ax3.axis('off')
        
        # 4. 数字识别结果
        ax4 = plt.subplot(2, 3, 4)
        img_numbers = img_rgb.copy()
        for num_info in recognized_numbers:
            x, y, w, h = num_info['bbox']
            cv2.rectangle(img_numbers, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{num_info['digit']} ({num_info['confidence']:.2f})"
            cv2.putText(img_numbers, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        ax4.imshow(img_numbers)
        ax4.set_title(f"4. 数字识别 ({len(recognized_numbers)}个)", 
                      fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        # 5. 装甲板检测结果（最终）
        ax5 = plt.subplot(2, 3, 5)
        img_result = cv2.cvtColor(result6['result_image'], cv2.COLOR_BGR2RGB)
        ax5.imshow(img_result)
        ax5.set_title(f"5. 装甲板检测结果 ({len(valid_armors)}个)", 
                      fontsize=14, fontweight='bold')
        ax5.axis('off')
        
        # 6. 评分信息
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        if valid_armors:
            info_text = "检测结果详情:\n\n"
            for i, armor in enumerate(valid_armors):
                info_text += f"装甲板 {i+1}:\n"
                info_text += f"  颜色: {armor['color']}\n"
                info_text += f"  数字: {armor['number']}\n"
                info_text += f"  数字置信度: {armor['number_confidence']:.2f}\n"
                info_text += f"  综合评分: {armor['score']:.1f}/100\n"
                
                details = armor['details']
                info_text += f"\n  评分明细:\n"
                info_text += f"    角度: {details['angle_score']:.1f}/30\n"
                info_text += f"    长度: {details['length_score']:.1f}/20\n"
                info_text += f"    间距: {details['distance_score']:.1f}/20\n"
                info_text += f"    垂直: {details['vertical_score']:.1f}/10\n"
                info_text += f"    数字: {details['number_score']:.1f}/15\n"
                info_text += f"    颜色: {details['color_score']:.1f}/5\n"
                info_text += "\n"
        else:
            info_text = "未检测到有效装甲板"
        
        ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存完整可视化
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"task6_visualization_{img_name}.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化结果已保存: {output_path}")
        
        plt.close()
    else:
        # 使用OpenCV创建组合可视化
        create_opencv_visualization(img_bgr, mask_red, mask_blue, left_bars, right_bars,
                                   recognized_numbers, result6, valid_armors, output_dir, image_path)
    
    # 保存单独的检测结果图像
    if save_individual:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        result_path = os.path.join(output_dir, f"task6_result_{img_name}.jpg")
        cv2.imwrite(result_path, result6['result_image'])
        print(f"检测结果图像已保存: {result_path}")


def create_opencv_visualization(img_bgr, mask_red, mask_blue, left_bars, right_bars,
                                recognized_numbers, result6, valid_armors, output_dir, image_path):
    """使用OpenCV创建可视化"""
    
    h, w = img_bgr.shape[:2]
    scale = 0.5  # 缩放比例
    
    # 1. 原始图像
    img1 = img_bgr.copy()
    cv2.putText(img1, "1. Original Image", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 2. 颜色分割结果
    mask_colored = np.zeros_like(img_bgr)
    mask_colored[mask_red > 0] = [0, 0, 255]  # 红色 (BGR)
    mask_colored[mask_blue > 0] = [255, 0, 0]  # 蓝色 (BGR)
    img2 = cv2.addWeighted(img_bgr, 0.6, mask_colored, 0.4, 0)
    cv2.putText(img2, f"2. Color Segmentation", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 3. 灯条检测结果
    img3 = img_bgr.copy()
    for bar in left_bars:
        x1, y1, x2, y2 = bar['line']
        cv2.line(img3, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img3, bar['center'], 5, (255, 0, 0), -1)
    for bar in right_bars:
        x1, y1, x2, y2 = bar['line']
        cv2.line(img3, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(img3, bar['center'], 5, (0, 0, 255), -1)
    cv2.putText(img3, f"3. Light Bars (L:{len(left_bars)}, R:{len(right_bars)})", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 4. 数字识别结果
    img4 = img_bgr.copy()
    for num_info in recognized_numbers:
        x, y, w, h = num_info['bbox']
        cv2.rectangle(img4, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{num_info['digit']} ({num_info['confidence']:.2f})"
        cv2.putText(img4, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img4, f"4. Number Recognition ({len(recognized_numbers)})", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 5. 最终检测结果
    img5 = result6['result_image'].copy()
    cv2.putText(img5, f"5. Armor Detection ({len(valid_armors)})", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 调整图像大小
    new_w, new_h = int(w * scale), int(h * scale)
    img1_small = cv2.resize(img1, (new_w, new_h))
    img2_small = cv2.resize(img2, (new_w, new_h))
    img3_small = cv2.resize(img3, (new_w, new_h))
    img4_small = cv2.resize(img4, (new_w, new_h))
    img5_small = cv2.resize(img5, (new_w, new_h))
    
    # 创建组合图像 (2行3列布局)
    # 第一行
    row1 = np.hstack([img1_small, img2_small, img3_small])
    # 第二行
    row2 = np.hstack([img4_small, img5_small, np.zeros((new_h, new_w, 3), dtype=np.uint8)])
    
    # 在空白区域添加文本信息
    info_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    y_offset = 30
    cv2.putText(info_img, "Detection Info:", (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if valid_armors:
        for i, armor in enumerate(valid_armors[:3]):  # 最多显示3个
            y_offset += 40
            text = f"Armor {i+1}: {armor['color']} {armor['number']}"
            cv2.putText(info_img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            text = f"Score: {armor['score']:.1f}/100"
            cv2.putText(info_img, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    row2 = np.hstack([img4_small, img5_small, info_img])
    
    # 组合所有图像
    combined = np.vstack([row1, row2])
    
    # 保存
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"task6_visualization_{img_name}.jpg")
    cv2.imwrite(output_path, combined)
    print(f"\n可视化结果已保存: {output_path}")
    
    return result6


def visualize_multiple_images():
    """可视化多张图像"""
    
    test_images = [
        "test_images/armor.jpg",
        "test_images/armor_001_normal.jpg",
        "test_images/armor_002_dark.jpg",
        "test_images/armor_005_angled.jpg"
    ]
    
    output_dir = "output"
    template_dir = "digit_templates"
    
    print("=" * 80)
    print("批量可视化第六题检测结果")
    print("=" * 80)
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n处理: {os.path.basename(img_path)}")
            try:
                visualize_task6(img_path, output_dir, template_dir, save_individual=True)
            except Exception as e:
                print(f"  处理失败: {e}")
                import traceback
                traceback.print_exc()


def create_comparison_visualization():
    """创建对比可视化（原版 vs 改进版）"""
    
    from armor import task6_armor_detection
    
    image_path = "test_images/armor.jpg"
    output_dir = "output"
    template_dir = "digit_templates"
    
    if not os.path.exists(image_path):
        print(f"图像不存在: {image_path}")
        return
    
    print("=" * 80)
    print("创建原版 vs 改进版对比可视化")
    print("=" * 80)
    
    # 预处理
    result1 = task1_image_preprocessing(image_path, output_dir)
    img_bgr = result1["original_bgr"]
    img_gray = result1["gray"]
    
    result2 = task2_color_segmentation(img_bgr, output_dir, show_windows=False)
    mask_red = result2["mask_red"]
    mask_blue = result2["mask_blue"]
    
    result3 = task3_light_bar_extraction(img_gray, img_bgr, output_dir, show_windows=False)
    left_bars = result3["left_bars"]
    right_bars = result3["right_bars"]
    
    result5 = task5_number_recognition(
        img_bgr, left_bars, right_bars,
        output_dir=output_dir,
        template_dir=template_dir,
        show_windows=False
    )
    recognized_numbers = result5["recognized_numbers"]
    
    # 原版
    result_original = task6_armor_detection(
        img_bgr, left_bars, right_bars,
        mask_red, mask_blue,
        recognized_numbers,
        output_dir=output_dir
    )
    
    # 改进版
    result_improved = task6_armor_detection_improved(
        img_bgr, left_bars, right_bars,
        mask_red, mask_blue,
        recognized_numbers,
        output_dir=output_dir,
        min_score=50.0,
        require_number=False,
        show_windows=False
    )
    
    if HAS_MATPLOTLIB:
        # 使用matplotlib创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 原版结果
        img_original = cv2.cvtColor(result_original['result_image'], cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_original)
        axes[0].set_title(f"原版方法\n检测到: {len(result_original['valid_armors'])} 个装甲板", 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 改进版结果
        img_improved = cv2.cvtColor(result_improved['result_image'], cv2.COLOR_BGR2RGB)
        axes[1].imshow(img_improved)
        axes[1].set_title(f"改进版方法\n检测到: {len(result_improved['valid_armors'])} 个装甲板", 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, "task6_comparison.jpg")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n对比可视化已保存: {output_path}")
        
        plt.close()
    else:
        # 使用OpenCV创建对比图
        img_original = result_original['result_image'].copy()
        img_improved = result_improved['result_image'].copy()
        
        # 添加标题
        cv2.putText(img_original, f"Original: {len(result_original['valid_armors'])} armors", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_improved, f"Improved: {len(result_improved['valid_armors'])} armors", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 水平拼接
        comparison = np.hstack([img_original, img_improved])
        
        output_path = os.path.join(output_dir, "task6_comparison.jpg")
        cv2.imwrite(output_path, comparison)
        print(f"\n对比可视化已保存: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "multiple":
            visualize_multiple_images()
        elif sys.argv[1] == "compare":
            create_comparison_visualization()
        else:
            visualize_task6(sys.argv[1])
    else:
        # 默认可视化单张图像
        visualize_task6("test_images/armor.jpg")
        # 创建对比可视化
        create_comparison_visualization()

