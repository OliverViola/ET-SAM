import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2


class SAMDataset(Dataset):
    def __init__(self, csv_path, img_size=1024):
        self.df = pd.read_csv(csv_path)
        self.img_size = img_size

        # 一致性增强参数容器
        self.shared_params = {}

        # 第一阶段：一致性增强（所有图像共享相同参数）
        self.consistent_aug = T.Compose([
            self.ColorJitterWithConsistency(
                brightness=0.1, contrast=0.03, saturation=0.03
            ),
            self.RandomGrayscaleWithConsistency(p=0.05)
        ])

        # 第二阶段：非一致性增强
        self.non_consistent_aug = T.Compose([
            self.ColorJitterWithConsistency(
                brightness=0.1, contrast=0.05, saturation=0.05,
                consistent_transform=False
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 读取原始数据
        base_path = '/root/autodl-tmp/lung-segment'
        img_path = self.df.iloc[idx]['images']
        mask_path = self.df.iloc[idx]['masks']

        image = Image.open(os.path.join(base_path, img_path)).convert('RGB')
        mask = Image.open(os.path.join(base_path, mask_path)).convert('L')

        # 重置共享参数
        self.shared_params = {}

        # 应用一致性增强（仅图像）
        image = self.consistent_aug(image)

        # 应用非一致性增强（仅图像）
        image = self.non_consistent_aug(image)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # 处理mask生成SAM提示
        # points, boxes = self.find_entities(mask)
        # labels = np.ones(len(points), dtype=np.int64)
        mask = np.array(mask).astype(np.float32) / 255 # mask 二值化

        return np.array(image), mask 
    
    def split_lungs(self, mask):
        """分离左右肺区域"""
        mask = cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_CLOSE, 
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30))
        )
        # 寻找连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4, ltype=cv2.CV_32S)
        # print(len(stats), stats)
        # 筛选面积最大的两个区域（排除背景）
        sorted_area_indices = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
        if len(sorted_area_indices) < 2:
            return None, None
        
        # 按x坐标排序确定左右
        sorted_indices = np.argsort(centroids[1:, 0]) + 1  # +1跳过背景标签
        if centroids[sorted_area_indices[0], 0] < centroids[sorted_area_indices[1], 0]:
            left_lung = labels == sorted_area_indices[0]
            right_lung = labels == sorted_area_indices[1]
        else:
            left_lung = labels == sorted_area_indices[1]
            right_lung = labels == sorted_area_indices[0]

        # gray_np = ((left_lung | right_lung) * 255).astype("uint8")
        # plt.imshow(gray_np, cmap="gray")  # 必须指定 cmap="gray"
        # plt.axis("off")
        # plt.show()
        
        return left_lung.astype(np.uint8), right_lung.astype(np.uint8)
    
    def generate_paired_boxes(self, left_mask, right_mask):
        expand_pixels = 75
        # 左肺包围盒
        left_contour = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        x1, y1, w1, h1 = cv2.boundingRect(left_contour[0])
        
        # 右肺包围盒
        right_contour = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        x2, y2, w2, h2 = cv2.boundingRect(right_contour[0])
        
        return np.array([[x1-expand_pixels, y1-expand_pixels, x1+w1 + expand_pixels, y1+h1 + expand_pixels],
                        [x2-expand_pixels, y2-expand_pixels, x2+w2+expand_pixels, y2+h2+expand_pixels]])
    
    def paired_centroids(self, left_mask, right_mask):
        
        # 左肺质心
        M_left = cv2.moments(left_mask)
        cx_left = int(M_left["m10"] / M_left["m00"])
        cy_left = int(M_left["m01"] / M_left["m00"])
        
        # 右肺质心
        M_right = cv2.moments(right_mask)
        cx_right = int(M_right["m10"] / M_right["m00"])
        cy_right = int(M_right["m01"] / M_right["m00"])
        
        return np.array([[cx_left, cy_left], 
                        [cx_right, cy_right]])

    def generate_background_points(self, boxes):
        
        background_points = []
        labels = []

        for box in boxes:
            x1, y1, x2, y2 = box
            background_points.extend([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
            labels.extend([0,0,0,0])

        background_points = np.array(background_points) 
        labels = np.array(labels, dtype=np.int64)
        # print(f'background_points length {background_points.shape}')
        
        return background_points, labels

    def generate_sam_prompts(self, left, right):
        # 获取基础点
        centroids = self.paired_centroids(left, right)
        boxes = self.generate_paired_boxes(left, right)
        # bg_points, bg_labels = self.generate_background_points(boxes)
        
        # 合并所有点
        all_points = np.vstack([centroids])
        
        # 创建标签（0:背景，1:前景）
        labels = np.concatenate([
            np.ones(len(centroids)),          # 质心为前景      
        ])
        
        return all_points.astype(int), labels, boxes

    def find_entities(self, mask):
        # 转为Numpy数组
        if type(mask) == torch.Tensor:
            img_array = mask.cpu().detach().numpy()
        else:
            img_array = np.array(mask) 
        
        img_array = img_array > 0.5
        
        bs = img_array.shape[0]
        lung_boxes = []
        lung_points = []
        point_labels = []
        for i in range(bs):
            left, right = self.split_lungs(img_array[i])
            if left is None:
                return False
                
            points, labels, boxes = self.generate_sam_prompts(left, right)
            lung_boxes.append(boxes)
            lung_points.append(points)
            point_labels.append(labels)
        
        lung_boxes = np.array(lung_boxes)
        lung_points = np.array(lung_points)
        point_labels = np.array(point_labels)

        return torch.tensor(lung_points, dtype=torch.float32), torch.tensor(point_labels, dtype=torch.int64), torch.tensor(lung_boxes, dtype=torch.float32)

    class ColorJitterWithConsistency(T.ColorJitter):
        def __init__(self, consistent_transform=True, **kwargs):
            super().__init__(**kwargs)
            self.consistent_transform = consistent_transform
            self._params = None

        def forward(self, img):
            if self.consistent_transform:
                # 生成或复用参数
                if self._params is None:
                    self._params = self.get_params(
                        self.brightness, self.contrast,
                        self.saturation, self.hue
                    )
                fn_idx, brightness, contrast, saturation, hue = self._params
                for fn_id in fn_idx:
                    if fn_id == 0 and brightness is not None:
                        img = T.functional.adjust_brightness(img, brightness)
                    elif fn_id == 1 and contrast is not None:
                        img = T.functional.adjust_contrast(img, contrast)
                    elif fn_id == 2 and saturation is not None:
                        img = T.functional.adjust_saturation(img, saturation)
                    elif fn_id == 3 and hue is not None:
                        img = T.functional.adjust_hue(img, hue)
                return img
            else:
                return super().forward(img)

    class RandomGrayscaleWithConsistency(T.RandomGrayscale):
        def __init__(self, p=0.1, consistent_transform=True):
            super().__init__(p=p)
            self.consistent_transform = consistent_transform
            self._apply_grayscale = None

        def forward(self, img):
            if self.consistent_transform:
                if self._apply_grayscale is None:
                    self._apply_grayscale = torch.rand(1) < self.p
                if self._apply_grayscale:
                    return T.functional.rgb_to_grayscale(img, num_output_channels=3)
                return img
            else:
                return super().forward(img)