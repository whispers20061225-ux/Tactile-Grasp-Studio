"""
材质识别模块 - 基于视觉的物体材质分类
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class MaterialRecognitionResult:
    """材质识别结果"""
    material_class: str
    confidence: float
    texture_type: str  # 光滑、粗糙、纹理
    reflectance: float  # 反射率估计
    transparency: float  # 透明度估计
    properties: Dict[str, Any]  # 其他属性

class MaterialRecognizer:
    """
    基于深度学习的材质识别器
    结合图像特征和触觉数据进行多模态识别
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 材质分类模型
        self.material_classes = config.get('classes', [
            'metal', 'plastic', 'glass', 'ceramic',
            'wood', 'fabric', 'paper', 'rubber'
        ])
        self.use_object_prior = config.get('use_object_prior', True)
        
        # 加载预训练模型或训练自定义模型（可选）
        self.model = None
        self.model_ready = False
        try:
            self.model = self._load_material_model()
            self.model_ready = self.model is not None
        except Exception:
            self.model = None
            self.model_ready = False
        
        # 触觉特征提取器（结合Paxini Gen3传感器）
        self.tactile_processor = None
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_material_model(self):
        """加载材质识别模型"""
        # 可以使用的模型架构：
        # 1. ResNet-based material classifier
        # 2. Vision Transformer for material recognition
        # 3. Multi-modal network (vision + tactile)
        
        model_path = self.config.get('model_path')
        if not model_path:
            # 未提供权重则使用启发式材质估计
            return None

        # 避免联网下载权重，优先使用本地权重
        model = torchvision.models.resnet50(weights=None)
        # 修改最后一层为材质分类
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.material_classes))
        
        # 加载预训练权重（如果有）
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def recognize_from_vision(self, image: np.ndarray,
                            roi: Optional[List[int]] = None,
                            object_type: Optional[str] = None) -> MaterialRecognitionResult:
        """
        基于视觉的材质识别
        
        Args:
            image: RGB图像
            roi: 感兴趣区域 [x1, y1, x2, y2]
            object_type: 物体类别（如 bottle、cup 等）
            
        Returns:
            材质识别结果
        """
        # 提取ROI
        if roi:
            x1, y1, x2, y2 = roi
            patch = image[y1:y2, x1:x2]
        else:
            patch = image
        
        # 如果ROI为空，退回整幅图像
        if patch.size == 0:
            patch = image

        material_class = None
        confidence_value = 0.0
        use_model = self.model_ready and self.model is not None

        if use_model:
            try:
                # 图像预处理
                input_tensor = self.transform(patch).unsqueeze(0).to(self.device)
                
                # 推理
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, class_idx = torch.max(probabilities, dim=1)
                
                material_class = self.material_classes[class_idx.item()]
                confidence_value = confidence.item()
            except Exception:
                use_model = False

        # 分析材质属性（总是可用）
        properties = self._analyze_material_properties(patch)
        
        if not use_model or material_class is None:
            # 使用启发式方法给出材质类别
            obj_hint = object_type if self.use_object_prior else None
            material_class = self._infer_material_from_properties(properties, object_type=obj_hint)
            confidence_value = properties.get('heuristic_confidence', 0.3)
        
        return MaterialRecognitionResult(
            material_class=material_class,
            confidence=confidence_value,
            texture_type=properties['texture'],
            reflectance=properties['reflectance'],
            transparency=properties['transparency'],
            properties=properties
        )
    
    def recognize_multimodal(self, image: np.ndarray, 
                           tactile_data: Dict[str, Any]) -> MaterialRecognitionResult:
        """
        多模态材质识别（视觉 + 触觉）
        
        Args:
            image: 视觉图像
            tactile_data: 触觉传感器数据
            
        Returns:
            结合多模态信息的材质识别结果
        """
        # 视觉特征提取
        visual_features = self._extract_visual_features(image)
        
        # 触觉特征提取
        tactile_features = self._extract_tactile_features(tactile_data)
        
        # 特征融合和分类
        combined_features = torch.cat([visual_features, tactile_features], dim=1)
        
        # 使用融合模型进行预测
        # （这里需要实现一个多模态融合模型）
        
        # 返回结果
        pass
    
    def _analyze_material_properties(self, image_patch: np.ndarray) -> Dict[str, Any]:
        """
        分析材质属性
        """
        properties = {
            'texture': 'smooth',
            'reflectance': 0.0,
            'transparency': 0.0,
            'color_histogram': {},
            'edge_sharpness': 0.0,
        }

        if image_patch is None or image_patch.size == 0:
            return properties

        # 转换为灰度
        try:
            gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
        except Exception:
            try:
                if image_patch.ndim == 2:
                    gray = image_patch
                else:
                    gray = cv2.cvtColor(image_patch, cv2.COLOR_BGR2GRAY)
            except Exception:
                return properties
        
        try:
            # 1. 纹理分析
            properties['texture'] = self._analyze_texture(gray)
            # 2. 反射率估计
            properties['reflectance'] = self._estimate_reflectance(gray)
            # 3. 透明度估计
            properties['transparency'] = self._estimate_transparency(image_patch)
            # 4. 颜色分析
            properties['color_histogram'] = self._analyze_color(image_patch)
            # 5. 边缘特征
            properties['edge_sharpness'] = self._analyze_edges(gray)
        except Exception:
            pass
        
        return properties

    def _infer_material_from_properties(self, properties: Dict[str, Any], object_type: Optional[str] = None) -> str:
        """基于简单视觉特征+物体类型先验的材质估计（兜底方案）"""
        texture = properties.get('texture', 'smooth')
        reflectance = float(properties.get('reflectance', 0.0))
        transparency = float(properties.get('transparency', 0.0))
        edge_sharpness = float(properties.get('edge_sharpness', 0.0))

        color_stats = properties.get('color_histogram', {}) or {}
        mean_rgb = color_stats.get('mean_rgb')
        warm_color = False
        neutral_color = False
        if isinstance(mean_rgb, list) and len(mean_rgb) == 3:
            r, g, b = mean_rgb
            warm_color = (r > b + 15) and (r > 80) and (g > 60)
            neutral_color = abs(r - g) < 15 and abs(g - b) < 15

        materials = list(self.material_classes)
        scores = {m: 0.02 for m in materials}

        # 物体类型先验
        obj = (object_type or "").lower()
        obj_text = obj
        # 常见英文同义词/别名
        if "phone" in obj:
            obj_text += " cell phone mobile phone"
        if "tv" in obj or "television" in obj:
            obj_text += " tv monitor screen"
        if "sofa" in obj or "couch" in obj:
            obj_text += " couch sofa"
        if "mug" in obj:
            obj_text += " cup"
        if "can" in obj or "tin" in obj:
            obj_text += " can"
        if "laptop" in obj or "notebook" in obj:
            obj_text += " laptop computer"
        if "keyboard" in obj:
            obj_text += " keyboard"
        if "mouse" in obj:
            obj_text += " mouse"
        if "remote" in obj:
            obj_text += " remote"
        if "bottle" in obj:
            obj_text += " bottle"
        if "scissors" in obj:
            obj_text += " scissors"
        if "knife" in obj:
            obj_text += " knife"
        if "fork" in obj:
            obj_text += " fork"
        if "spoon" in obj:
            obj_text += " spoon"
        if "book" in obj:
            obj_text += " book paper"
        if "box" in obj:
            obj_text += " box"
        if "bag" in obj:
            obj_text += " bag"

        # 常见中文关键词（只做轻量映射）
        if "杯" in obj:
            obj_text += " cup"
        if "瓶" in obj:
            obj_text += " bottle"
        if "罐" in obj:
            obj_text += " can"
        if "手机" in obj:
            obj_text += " cell phone"
        if "电脑" in obj or "笔记本" in obj:
            obj_text += " laptop"
        if "键盘" in obj:
            obj_text += " keyboard"
        if "鼠标" in obj:
            obj_text += " mouse"
        if "遥控" in obj:
            obj_text += " remote"
        if "桌" in obj:
            obj_text += " table"
        if "椅" in obj:
            obj_text += " chair"
        if "沙发" in obj:
            obj_text += " sofa"
        if "剪刀" in obj:
            obj_text += " scissors"
        if "刀" in obj:
            obj_text += " knife"
        if "叉" in obj:
            obj_text += " fork"
        if "勺" in obj:
            obj_text += " spoon"
        if "书" in obj or "纸" in obj:
            obj_text += " book paper"
        if "包" in obj:
            obj_text += " bag"
        if "箱" in obj or "盒" in obj:
            obj_text += " box"

        # 若标签本身包含材质词，作为强提示
        if "wood" in obj or "木" in obj:
            scores["wood"] += 0.35
        if "metal" in obj or "金属" in obj:
            scores["metal"] += 0.35
        if "glass" in obj or "玻璃" in obj:
            scores["glass"] += 0.35
        if "plastic" in obj or "塑料" in obj:
            scores["plastic"] += 0.35
        if "paper" in obj or "纸" in obj:
            scores["paper"] += 0.35
        if "fabric" in obj or "布" in obj or "衣" in obj:
            scores["fabric"] += 0.35
        if "rubber" in obj or "橡胶" in obj:
            scores["rubber"] += 0.35
        if "ceramic" in obj or "陶瓷" in obj:
            scores["ceramic"] += 0.35
        priors = {
            "bottle": {"plastic": 0.55, "glass": 0.3, "metal": 0.1},
            "cup": {"ceramic": 0.35, "metal": 0.25, "plastic": 0.25, "glass": 0.15},
            "mug": {"ceramic": 0.4, "metal": 0.25, "plastic": 0.25},
            "wine glass": {"glass": 0.8, "plastic": 0.1},
            "can": {"metal": 0.8, "plastic": 0.1},
            "bowl": {"ceramic": 0.4, "plastic": 0.3, "glass": 0.2},
            "vase": {"ceramic": 0.4, "glass": 0.4, "plastic": 0.2},
            "thermos": {"metal": 0.7, "plastic": 0.2},
            "laptop": {"metal": 0.5, "plastic": 0.35, "glass": 0.15},
            "cell phone": {"glass": 0.4, "metal": 0.3, "plastic": 0.3},
            "keyboard": {"plastic": 0.7, "metal": 0.2},
            "mouse": {"plastic": 0.7, "rubber": 0.2, "metal": 0.1},
            "remote": {"plastic": 0.8, "rubber": 0.15},
            "tv": {"glass": 0.4, "plastic": 0.4, "metal": 0.2},
            "monitor": {"glass": 0.4, "plastic": 0.4, "metal": 0.2},
            "table": {"wood": 0.6, "metal": 0.2, "plastic": 0.15},
            "chair": {"wood": 0.4, "metal": 0.25, "plastic": 0.2, "fabric": 0.15},
            "sofa": {"fabric": 0.6, "wood": 0.2, "plastic": 0.1},
            "bed": {"fabric": 0.5, "wood": 0.3, "metal": 0.2},
            "door": {"wood": 0.6, "metal": 0.2},
            "cabinet": {"wood": 0.7, "metal": 0.2},
            "book": {"paper": 0.7, "wood": 0.2},
            "box": {"paper": 0.4, "plastic": 0.35, "wood": 0.2},
            "bag": {"fabric": 0.6, "plastic": 0.25},
            "bottle opener": {"metal": 0.8},
            "scissors": {"metal": 0.9},
            "knife": {"metal": 0.9},
            "fork": {"metal": 0.9},
            "spoon": {"metal": 0.9},
        }

        prior = None
        if obj_text:
            # 允许子串匹配
            for key in priors:
                if key in obj_text:
                    prior = priors[key]
                    break
            if prior is None and obj_text in priors:
                prior = priors[obj_text]

        if prior:
            for mat, weight in prior.items():
                if mat in scores:
                    scores[mat] += weight

        # 对特定物体类型做弱约束，避免不合理材质
        if "cup" in obj_text or "mug" in obj_text:
            scores["wood"] -= 0.15
            scores["fabric"] -= 0.1
        if any(key in obj_text for key in ("bottle", "can", "laptop", "keyboard", "mouse", "monitor", "tv", "remote")):
            scores["wood"] -= 0.12
            scores["fabric"] -= 0.08
        if any(key in obj_text for key in ("scissors", "knife", "fork", "spoon")):
            scores["wood"] -= 0.2

        # 视觉启发式调整
        if transparency > 0.6:
            scores["glass"] += 0.25
            scores["plastic"] += 0.05
        elif transparency < 0.2:
            scores["glass"] -= 0.05

        if reflectance > 0.6:
            scores["metal"] += 0.25
            scores["glass"] += 0.05
        elif reflectance < 0.3:
            if warm_color:
                scores["wood"] += 0.18
            scores["fabric"] += 0.15
            scores["paper"] += 0.12

        if texture == "rough":
            if warm_color:
                scores["wood"] += 0.22
            else:
                scores["wood"] += 0.05
            scores["fabric"] += 0.18
            scores["paper"] += 0.12
        elif texture == "slightly_textured":
            scores["plastic"] += 0.12
            if warm_color:
                scores["wood"] += 0.08
        else:  # smooth
            scores["plastic"] += 0.15
            scores["ceramic"] += 0.15

        # 若纹理不粗糙且偏冷色，降低木材概率
        if texture != "rough" and not warm_color:
            scores["wood"] -= 0.08

        # 颜色倾向
        if neutral_color:
            scores["metal"] += 0.12
            scores["ceramic"] += 0.08
            scores["plastic"] += 0.05
            scores["wood"] -= 0.05
        elif warm_color:
            scores["wood"] += 0.1

        # 边缘锐利度：偏向金属/陶瓷
        if edge_sharpness > 1500:
            scores["metal"] += 0.12
            scores["ceramic"] += 0.08

        # 防止负数
        for mat in scores:
            if scores[mat] < 0.0:
                scores[mat] = 0.0

        total = sum(scores.values()) or 1.0
        best_mat = max(scores, key=scores.get)
        properties['heuristic_confidence'] = float(scores[best_mat] / total)
        return best_mat
    
    def _analyze_texture(self, gray_image: np.ndarray) -> str:
        """分析纹理类型"""
        # 计算局部二值模式（LBP）或灰度共生矩阵（GLCM）
        lbp = self._compute_lbp(gray_image)
        
        # 分析纹理特征
        if lbp.std() < 10:
            return "smooth"
        elif lbp.std() < 30:
            return "slightly_textured"
        else:
            return "rough"
    
    def _estimate_reflectance(self, gray_image: np.ndarray) -> float:
        """估计反射率"""
        # 分析图像亮度分布
        mean_brightness = gray_image.mean() / 255.0
        
        # 考虑对比度
        contrast = gray_image.std() / 255.0
        
        # 简单估计反射率（高亮区域比例）
        highlight_ratio = np.sum(gray_image > 200) / gray_image.size
        
        reflectance = 0.5 * mean_brightness + 0.3 * contrast + 0.2 * highlight_ratio
        
        return float(reflectance)
    
    def _estimate_transparency(self, color_image: np.ndarray) -> float:
        """估计透明度（降低对低饱和度物体的误判）"""
        if color_image is None or color_image.size == 0:
            return 0.0

        try:
            hsv = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        except Exception:
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        saturation = hsv[:, :, 1].mean() / 255.0
        value = hsv[:, :, 2]
        value_mean = value.mean() / 255.0
        value_std = value.std() / 255.0

        # 边缘强度辅助（透明物体往往背景边缘更明显）
        gray = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY) if color_image.ndim == 3 else color_image
        edge_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        edge_score = min(1.0, edge_var / 1500.0)

        # 透明度估计：低饱和度 + 明显背景/边缘
        base = (1.0 - saturation)
        transparency = base * (0.3 + 0.7 * edge_score) * (0.5 + 0.5 * value_std)

        # 避免白色/低饱和度实心物体被判为透明
        if value_mean > 0.8 and value_std < 0.08:
            transparency *= 0.3

        return float(max(0.0, min(1.0, transparency)))

    def _compute_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """计算简单LBP纹理图（无需额外依赖）"""
        if gray_image is None or gray_image.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)
        if gray_image.ndim != 2:
            gray = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
        else:
            gray = gray_image
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return np.zeros_like(gray, dtype=np.uint8)

        center = gray[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        for idx, (dy, dx) in enumerate(offsets):
            neighbor = gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << (7 - idx))
        return lbp

    def _analyze_edges(self, gray_image: np.ndarray) -> float:
        """边缘清晰度估计"""
        if gray_image is None or gray_image.size == 0:
            return 0.0
        lap = cv2.Laplacian(gray_image, cv2.CV_64F)
        return float(lap.var())

    def _analyze_color(self, image_patch: np.ndarray) -> Dict[str, Any]:
        """颜色统计（均值/方差）"""
        if image_patch is None or image_patch.size == 0:
            return {}
        if image_patch.ndim == 2:
            mean_val = float(image_patch.mean())
            std_val = float(image_patch.std())
            return {'mean_gray': mean_val, 'std_gray': std_val}

        mean_rgb = image_patch.reshape(-1, 3).mean(axis=0).tolist()
        std_rgb = image_patch.reshape(-1, 3).std(axis=0).tolist()
        return {
            'mean_rgb': [float(v) for v in mean_rgb],
            'std_rgb': [float(v) for v in std_rgb],
        }

    def _extract_visual_features(self, image: np.ndarray) -> torch.Tensor:
        """简化视觉特征提取（占位，供多模态接口使用）"""
        if image is None or image.size == 0:
            return torch.zeros((1, 16), device=self.device)
        hist = cv2.calcHist([image], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
        feat = torch.tensor(hist.flatten()[:16], dtype=torch.float32, device=self.device).unsqueeze(0)
        return feat

    def _extract_tactile_features(self, tactile_data: Dict[str, Any]) -> torch.Tensor:
        """简化触觉特征提取（占位，供多模态接口使用）"""
        values = []
        if isinstance(tactile_data, dict):
            for key in sorted(tactile_data.keys()):
                try:
                    values.append(float(tactile_data[key]))
                except Exception:
                    continue
        if not values:
            return torch.zeros((1, 16), device=self.device)
        arr = np.array(values[:16], dtype=np.float32)
        if arr.size < 16:
            arr = np.pad(arr, (0, 16 - arr.size))
        return torch.tensor(arr, device=self.device).unsqueeze(0)
