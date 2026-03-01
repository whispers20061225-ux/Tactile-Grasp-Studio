import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import time
from pathlib import Path

from .grasp_predictor import GraspPredictor

logger = logging.getLogger(__name__)


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, model_path: str, config: Dict = None):
        self.config = config or {}
        
        # 设备
        self.device = torch.device(self.config.get('device', 
                                                  'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 推理统计
        self.inference_count = 0
        self.total_inference_time = 0
        
        # 缓存
        self.cache = {}
        
        logger.info(f"推理引擎初始化完成，设备: {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 根据文件扩展名判断模型类型
        if model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_class' in checkpoint:
                model_class = checkpoint['model_class']
                if model_class == 'GraspPredictor':
                    model_config = checkpoint.get('config', {})
                    model = GraspPredictor(model_config)
                else:
                    raise ValueError(f"未知模型类: {model_class}")
            else:
                # 尝试作为GraspPredictor加载
                model_config = checkpoint.get('config', {})
                model = GraspPredictor(model_config)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
        else:
            raise ValueError(f"不支持的模型格式: {model_path}")
        
        logger.info(f"模型已从 {model_path} 加载")
        return model
    
    def preprocess_visual(self, image: np.ndarray) -> np.ndarray:
        """预处理视觉输入"""
        # 调整大小
        target_size = self.config.get('image_size', (224, 224))
        if image.shape[:2] != target_size:
            from PIL import Image
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(target_size[::-1])  # (W, H) to (H, W)
            image = np.array(pil_image)
        
        # 归一化
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # 确保通道顺序为 (H, W, C)
        if image.shape[-1] != 3:
            raise ValueError(f"图像通道数应为3，实际为{image.shape[-1]}")
        
        return image
    
    def preprocess_tactile(self, tactile_data: np.ndarray) -> np.ndarray:
        """预处理触觉输入"""
        # 确保形状为 (192,)
        if tactile_data.size != 192:
            # 尝试重塑
            if tactile_data.size == 192:
                tactile_data = tactile_data.reshape(192)
            else:
                # 降采样或填充
                if tactile_data.size > 192:
                    # 降采样
                    tactile_data = tactile_data[:192]
                else:
                    # 填充零
                    padded = np.zeros(192)
                    padded[:tactile_data.size] = tactile_data.flatten()
                    tactile_data = padded
        
        # 归一化到 [0, 1]
        if tactile_data.max() > 1.0 or tactile_data.min() < 0:
            tactile_min = tactile_data.min()
            tactile_max = tactile_data.max()
            if tactile_max > tactile_min:
                tactile_data = (tactile_data - tactile_min) / (tactile_max - tactile_min)
        
        return tactile_data.astype(np.float32)
    
    def preprocess_arm_state(self, arm_state: Dict) -> np.ndarray:
        """预处理机械臂状态"""
        state_components = []
        
        # 关节角度 (6)
        joint_positions = arm_state.get('joint_positions', np.zeros(6))
        joint_positions_normalized = np.clip(joint_positions / np.pi, -1, 1)
        state_components.extend(joint_positions_normalized)
        
        # 关节速度 (6)
        joint_velocities = arm_state.get('joint_velocities', np.zeros(6))
        joint_velocities_normalized = np.tanh(joint_velocities)
        state_components.extend(joint_velocities_normalized)
        
        # 末端位姿 (7: 位置3 + 四元数4)
        end_effector_pose = arm_state.get('end_effector_pose', np.zeros(7))
        if len(end_effector_pose) == 7:
            position = end_effector_pose[:3] / 1.0  # 假设工作空间在1米内
            quaternion = end_effector_pose[3:]
            state_components.extend(position)
            state_components.extend(quaternion)
        elif len(end_effector_pose) == 6:
            # 位置 + 欧拉角
            position = end_effector_pose[:3] / 1.0
            euler = end_effector_pose[3:]
            state_components.extend(position)
            state_components.extend(np.sin(euler))
            state_components.extend(np.cos(euler))
        
        # 力/力矩 (6)
        force_torque = arm_state.get('force_torque', np.zeros(6))
        force_torque_normalized = np.tanh(force_torque / 10.0)  # 假设最大10N/Nm
        state_components.extend(force_torque_normalized)
        
        # 转换为numpy数组
        state_array = np.array(state_components, dtype=np.float32)
        
        # 确保维度为24
        if len(state_array) < 24:
            # 补零
            padding = np.zeros(24 - len(state_array))
            state_array = np.concatenate([state_array, padding])
        elif len(state_array) > 24:
            # 截断
            state_array = state_array[:24]
        
        return state_array
    
    def predict(self, visual_input: np.ndarray, 
                tactile_input: np.ndarray,
                arm_state: Dict,
                threshold: float = 0.7) -> Dict[str, Any]:
        """
        执行推理
        
        Args:
            visual_input: RGB图像 [H, W, 3]
            tactile_input: 触觉数据 [任意形状，会调整为192]
            arm_state: 机械臂状态字典
            threshold: 抓取成功概率阈值
        
        Returns:
            推理结果字典
        """
        start_time = time.time()
        
        try:
            # 预处理输入
            visual_processed = self.preprocess_visual(visual_input)
            tactile_processed = self.preprocess_tactile(tactile_input)
            arm_state_processed = self.preprocess_arm_state(arm_state)
            
            # 转换为张量
            visual_tensor = torch.FloatTensor(visual_processed).permute(2, 0, 1).unsqueeze(0)
            tactile_tensor = torch.FloatTensor(tactile_processed).unsqueeze(0)
            arm_tensor = torch.FloatTensor(arm_state_processed).unsqueeze(0)
            
            # 移动到设备
            visual_tensor = visual_tensor.to(self.device)
            tactile_tensor = tactile_tensor.to(self.device)
            arm_tensor = arm_tensor.to(self.device)
            
            # 推理
            with torch.no_grad():
                predictions = self.model(visual_tensor, tactile_tensor, arm_tensor)
            
            # 后处理
            result = self.postprocess(predictions, threshold)
            
            # 计算推理时间
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            # 添加元数据
            result['metadata'] = {
                'inference_time_ms': inference_time * 1000,
                'avg_inference_time_ms': (self.total_inference_time / self.inference_count) * 1000,
                'inference_count': self.inference_count,
                'timestamp': time.time(),
                'threshold': threshold
            }
            
            logger.debug(f"推理完成，时间: {inference_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            raise
    
    def postprocess(self, predictions: Dict[str, torch.Tensor], 
                   threshold: float) -> Dict[str, Any]:
        """后处理预测结果"""
        # 移动到CPU并转换为numpy
        grasp_pose = predictions['grasp_pose'].cpu().numpy()[0]
        grasp_quality = predictions['grasp_quality'].cpu().numpy()[0]
        force_pred = predictions['force_prediction'].cpu().numpy()[0]
        uncertainty = predictions.get('uncertainty', torch.zeros(1)).cpu().numpy()[0]
        
        # 分解位姿
        position = grasp_pose[:3]  # [-1, 1] 归一化位置
        orientation = grasp_pose[3:7]  # 归一化四元数
        gripper_width = grasp_pose[7]  # [0, 1] 开合度
        
        # 反归一化位置（假设工作空间为1米立方体）
        position_denorm = position * 0.5  # 转换为[-0.5, 0.5]米
        
        # 判断是否执行抓取
        success_prob = grasp_quality[0]
        should_grasp = success_prob > threshold and uncertainty < 0.5
        
        # 构建结果字典
        result = {
            'grasp_pose': {
                'position': position_denorm.tolist(),  # 米
                'position_normalized': position.tolist(),
                'orientation': orientation.tolist(),  # 四元数 [w, x, y, z]
                'gripper_width': float(gripper_width),  # 归一化开合度
                'gripper_width_mm': float(gripper_width * 100)  # 假设最大开合100mm
            },
            'grasp_quality': {
                'success_probability': float(grasp_quality[0]),
                'stability_score': float(grasp_quality[1]),
                'safety_score': float(grasp_quality[2]),
                'confidence': float(success_prob * (1 - uncertainty))
            },
            'force_prediction': {
                'grasp_force_normalized': force_pred[:3].tolist(),  # [Fx, Fy, Fz] 归一化
                'grasp_force_newton': (force_pred[:3] * 20).tolist(),  # 假设最大20N
                'slip_probability': float(force_pred[3]),
                'contact_balance': force_pred[4:6].tolist()  # 接触分布
            },
            'uncertainty': float(uncertainty),
            'decision': {
                'should_grasp': bool(should_grasp),
                'threshold': threshold,
                'reason': 'high_confidence' if should_grasp else 'low_confidence_or_high_uncertainty'
            }
        }
        
        return result
    
    def batch_predict(self, batch_inputs: List[Dict], 
                     threshold: float = 0.7) -> List[Dict[str, Any]]:
        """批量推理"""
        results = []
        
        for i, inputs in enumerate(batch_inputs):
            try:
                result = self.predict(
                    inputs['visual'],
                    inputs['tactile'],
                    inputs['arm_state'],
                    threshold
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"批量推理第{i}个样本失败: {e}")
                # 返回默认结果
                default_result = self._get_default_result()
                results.append(default_result)
        
        return results
    
    def _get_default_result(self) -> Dict[str, Any]:
        """获取默认推理结果"""
        return {
            'grasp_pose': {
                'position': [0, 0, 0],
                'orientation': [1, 0, 0, 0],
                'gripper_width': 0.5
            },
            'grasp_quality': {
                'success_probability': 0.0,
                'confidence': 0.0
            },
            'decision': {
                'should_grasp': False,
                'reason': 'inference_failed'
            },
            'metadata': {
                'inference_time_ms': 0,
                'error': True
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取推理统计"""
        return {
            'total_inferences': self.inference_count,
            'avg_inference_time_ms': (self.total_inference_time / max(self.inference_count, 1)) * 1000,
            'total_inference_time_s': self.total_inference_time,
            'device': str(self.device)
        }
    
    def warmup(self, num_iterations: int = 10):
        """预热模型（运行几次推理以避免首次推理延迟）"""
        logger.info("开始模型预热...")
        
        # 创建虚拟输入
        dummy_visual = np.random.rand(224, 224, 3).astype(np.float32) * 255
        dummy_tactile = np.random.rand(192).astype(np.float32)
        dummy_arm_state = {
            'joint_positions': np.zeros(6),
            'joint_velocities': np.zeros(6),
            'end_effector_pose': np.array([0, 0, 0, 1, 0, 0, 0])
        }
        
        for i in range(num_iterations):
            try:
                self.predict(dummy_visual, dummy_tactile, dummy_arm_state, 0.5)
                logger.debug(f"预热迭代 {i+1}/{num_iterations}")
            except Exception as e:
                logger.warning(f"预热迭代 {i+1} 失败: {e}")
        
        logger.info(f"模型预热完成，运行了 {num_iterations} 次推理")