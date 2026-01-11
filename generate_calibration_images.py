"""
生成棋盘格标定图像
用于第四题：相机标定与畸变矫正
"""

import cv2
import numpy as np
import os

def generate_chessboard_image(output_path: str, 
                              chessboard_size: tuple = (9, 6),
                              square_size: int = 50,
                              image_size: tuple = (640, 480),
                              add_noise: bool = False,
                              add_distortion: bool = False):
    """
    生成棋盘格标定图像
    
    参数:
        output_path: 输出路径
        chessboard_size: 棋盘格内部角点数 (width, height)
        square_size: 每个方格的大小（像素）
        image_size: 图像尺寸 (width, height)
        add_noise: 是否添加噪声
        add_distortion: 是否添加模拟畸变
    """
    width, height = image_size
    img = np.ones((height, width), dtype=np.uint8) * 255
    
    # 计算棋盘格在图像中的位置（居中）
    board_width = chessboard_size[0] * square_size
    board_height = chessboard_size[1] * square_size
    start_x = (width - board_width) // 2
    start_y = (height - board_height) // 2
    
    # 绘制棋盘格
    for i in range(chessboard_size[1] + 1):
        for j in range(chessboard_size[0] + 1):
            x = start_x + j * square_size
            y = start_y + i * square_size
            
            # 交替绘制黑白方格
            if (i + j) % 2 == 0:
                # 绘制黑色方格
                if x < width and y < height:
                    end_x = min(x + square_size, width)
                    end_y = min(y + square_size, height)
                    img[y:end_y, x:end_x] = 0
    
    # 转换为BGR格式
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 添加噪声（可选）
    if add_noise:
        noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
        img_bgr = cv2.add(img_bgr, noise)
    
    # 添加模拟畸变（可选）
    if add_distortion:
        # 简单的径向畸变模拟
        h, w = img_bgr.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # 创建畸变映射
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # 计算相对于中心的距离
                dx = x - center_x
                dy = y - center_y
                r2 = dx*dx + dy*dy
                r4 = r2 * r2
                
                # 径向畸变系数
                k1, k2 = 0.1, 0.05
                
                # 应用畸变
                x_distorted = x + dx * (k1 * r2 + k2 * r4)
                y_distorted = y + dy * (k1 * r2 + k2 * r4)
                
                map_x[y, x] = x_distorted
                map_y[y, x] = y_distorted
        
        img_bgr = cv2.remap(img_bgr, map_x, map_y, cv2.INTER_LINEAR)
    
    # 保存图像
    cv2.imwrite(output_path, img_bgr)
    return img_bgr


def generate_calibration_dataset(output_dir: str = "calibration_images", 
                                 num_images: int = 10,
                                 chessboard_size: tuple = (9, 6)):
    """
    生成一组标定图像数据集
    
    参数:
        output_dir: 输出目录
        num_images: 生成图像数量
        chessboard_size: 棋盘格大小
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"生成 {num_images} 张棋盘格标定图像...")
    
    for i in range(num_images):
        # 随机变化参数
        square_size = np.random.randint(40, 60)
        image_size = (
            np.random.randint(600, 800),
            np.random.randint(450, 600)
        )
        
        # 随机添加噪声和畸变
        add_noise = np.random.random() > 0.5
        add_distortion = np.random.random() > 0.7  # 30%的概率添加畸变
        
        filename = f"chessboard_{i+1:03d}.jpg"
        output_path = os.path.join(output_dir, filename)
        
        generate_chessboard_image(
            output_path,
            chessboard_size=chessboard_size,
            square_size=square_size,
            image_size=image_size,
            add_noise=add_noise,
            add_distortion=add_distortion
        )
        
        print(f"  已生成: {filename}")
    
    print(f"\n完成！共生成 {num_images} 张标定图像，保存在: {output_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "dataset":
        # 生成数据集
        num_images = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        generate_calibration_dataset("calibration_images", num_images)
    else:
        # 生成单张图像
        output_path = "calibration_images/chessboard_001.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img = generate_chessboard_image(output_path, add_distortion=False)
        print(f"棋盘格图像已生成: {output_path}")
        
        # 显示图像
        cv2.imshow("Chessboard", img)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

