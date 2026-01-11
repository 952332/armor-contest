# 测试图像数据集说明

## 概述
这是一个用于测试机甲大师装甲板识别系统的模拟图像数据集。虽然网上没有找到公开的RoboMaster装甲板数据集，但我们创建了一个模拟数据集来测试各个功能模块。

## 图像生成器
使用 `generate_test_image.py` 可以生成多种场景的测试图像。

### 使用方法

#### 生成单张图像
```bash
python generate_test_image.py [scenario]
```

可选场景：
- `normal` - 正常场景（默认）
- `bright` - 明亮场景
- `dark` - 暗场景
- `multiple` - 多个装甲板
- `angled` - 倾斜的装甲板

#### 生成数据集
```bash
python generate_test_image.py dataset [数量]
```

例如：生成10张图像
```bash
python generate_test_image.py dataset 10
```

## 图像特征

### 装甲板结构
每个装甲板包含：
- **主体区域**：红色或蓝色矩形区域
- **左右灯条**：垂直的亮色灯条（用于定位）
- **数字区域**：中间的白色区域，显示数字0-9

### 场景类型

1. **normal（正常）**
   - 标准光照条件
   - 包含红色和蓝色装甲板各一个
   - 适合基础测试

2. **bright（明亮）**
   - 高亮度背景
   - 装甲板颜色更亮
   - 测试高光条件下的识别

3. **dark（暗）**
   - 低亮度背景
   - 装甲板颜色较暗
   - 测试低光条件下的识别

4. **multiple（多个）**
   - 包含4个装甲板
   - 不同颜色和数字
   - 测试多目标检测

5. **angled（倾斜）**
   - 装甲板有旋转角度
   - 测试角度变化下的识别

## 文件命名规则
- `armor.jpg` - 默认测试图像
- `armor_XXX_scenario.jpg` - 数据集图像（XXX为序号，scenario为场景类型）

## 使用建议

### 题目1：图像预处理
使用任意一张图像测试：
```python
from armor import task1_image_preprocessing
result = task1_image_preprocessing("test_images/armor.jpg")
```

### 题目2：颜色分割
使用包含红色和蓝色装甲板的图像：
```python
from armor import task2_color_segmentation
result = task2_color_segmentation(img_bgr)
```

### 题目3-6：完整识别流程
使用 `normal` 或 `multiple` 场景的图像进行测试。

### 题目7：位姿解算
需要相机标定数据，可以使用模拟的相机内参。

### 题目8-9：视频跟踪
需要视频文件，可以：
1. 使用OpenCV录制视频
2. 从生成的图像序列创建视频

## 注意事项

1. **模拟数据限制**：这些是模拟图像，与真实装甲板可能存在差异
2. **颜色范围**：实际应用中可能需要调整HSV颜色阈值
3. **数字识别**：当前数字是简单的文本绘制，实际识别可能需要更复杂的算法
4. **真实数据**：建议在实际项目中收集真实装甲板图像进行训练和测试

## 获取真实数据

如果条件允许，建议：
1. 从RoboMaster比赛视频中截图
2. 使用相机拍摄真实装甲板
3. 联系RoboMaster官方或社区获取测试数据

## 相关资源

- RoboMaster官网：https://www.robomaster.com
- GitHub开源项目：搜索 "RoboMaster" 或 "armor detection"

