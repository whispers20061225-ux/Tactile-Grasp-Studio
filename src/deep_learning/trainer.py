"""
模型训练器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings

from .grasp_predictor import GraspPredictor, GraspDataset
from .models import (
    TactileCNN,
    VisualEncoder,
    ForceControlNN,
    SlipDetector,
    ContactStateClassifier
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器 - 支持多种模型类型的训练"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 设备
        self.device = torch.device(config.get('device', 
                                            'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 创建模型
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 损失函数
        self.criterions = self._create_criterions()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # TensorBoard
        self.writer = None
        if config.get('use_tensorboard', False):
            log_dir = config.get('log_dir', f'runs/{time.strftime("%Y%m%d-%H%M%S")}')
            self.writer = SummaryWriter(log_dir)
        
        # 训练统计
        self.train_history = []
        self.val_history = []
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.global_step = 0
        
        logger.info(f"模型训练器初始化完成，设备: {self.device}")
        logger.info(f"模型类型: {config.get('model_type', 'unknown')}")
        logger.info(f"模型参数数量: {self._count_parameters():,}")
    
    def _count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _create_model(self) -> nn.Module:
        """根据配置创建模型"""
        model_type = self.config.get('model_type', 'GraspPredictor')
        model_config = self.config.get('model_config', {})
        
        logger.info(f"创建模型: {model_type}")
        
        if model_type == 'GraspPredictor':
            model = GraspPredictor(model_config)
            
        elif model_type == 'TactileCNN':
            input_channels = model_config.get('input_channels', 192)
            output_dim = model_config.get('output_dim', 128)
            model = TactileCNN(input_channels, output_dim)
            
        elif model_type == 'ForceControlNN':
            state_dim = model_config.get('state_dim', 100)
            hidden_dims = model_config.get('hidden_dims', [256, 128, 64])
            model = ForceControlNN(state_dim, hidden_dims)
            
        elif model_type == 'VisualEncoder':
            backbone = model_config.get('backbone', 'resnet50')
            pretrained = model_config.get('pretrained', True)
            model = VisualEncoder(backbone, pretrained)
            
        elif model_type == 'SlipDetector':
            input_dim = model_config.get('input_dim', 192)
            hidden_dim = model_config.get('hidden_dim', 128)
            lstm_layers = model_config.get('lstm_layers', 2)
            model = SlipDetector(input_dim, hidden_dim, lstm_layers)
            
        elif model_type == 'ContactStateClassifier':
            input_dim = model_config.get('input_dim', 192)
            num_classes = model_config.get('num_classes', 5)
            model = ContactStateClassifier(input_dim, num_classes)
            
        else:
            raise ValueError(f"未知模型类型: {model_type}")
        
        return model
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'Adam')
        lr = optimizer_config.get('learning_rate', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay,
                betas=optimizer_config.get('betas', (0.9, 0.999))
            )
        elif optimizer_type == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == 'SGD':
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=weight_decay,
                nesterov=optimizer_config.get('nesterov', True)
            )
        else:
            raise ValueError(f"未知优化器: {optimizer_type}")
        
        logger.info(f"使用优化器: {optimizer_type}, 学习率: {lr}")
        return optimizer
    
    def _create_criterions(self) -> Dict[str, nn.Module]:
        """创建损失函数集合"""
        model_type = self.config.get('model_type', 'GraspPredictor')
        criterions = {}
        
        if model_type == 'GraspPredictor':
            criterions['pose'] = nn.MSELoss()
            criterions['quality'] = nn.BCELoss()
            criterions['force'] = nn.MSELoss()
            
        elif model_type == 'ForceControlNN':
            criterions['force'] = nn.MSELoss()
            criterions['stiffness'] = nn.MSELoss()
            criterions['gains'] = nn.MSELoss()
            
        elif model_type in ['TactileCNN', 'VisualEncoder']:
            criterions['feature'] = nn.MSELoss()
            
        elif model_type == 'SlipDetector':
            criterions['classification'] = nn.BCELoss()
            criterions['regression'] = nn.MSELoss()
            
        elif model_type == 'ContactStateClassifier':
            criterions['classification'] = nn.CrossEntropyLoss()
            criterions['regression'] = nn.MSELoss()
            
        else:
            # 默认损失函数
            criterions['default'] = nn.MSELoss()
        
        return criterions
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', None)
        
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 5),
                verbose=True
            )
        elif scheduler_type == 'CosineAnnealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.config.get('epochs', 100)),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 20),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'OneCycleLR':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'],
                epochs=self.config.get('epochs', 100),
                steps_per_epoch=scheduler_config.get('steps_per_epoch', 100)
            )
        else:
            scheduler = None
        
        if scheduler:
            logger.info(f"使用学习率调度器: {scheduler_type}")
        
        return scheduler
    
    def _compute_loss(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失（根据模型类型）"""
        model_type = self.config.get('model_type', 'GraspPredictor')
        loss_values = {}
        total_loss = 0.0
        
        if model_type == 'GraspPredictor':
            # 位姿损失
            if 'grasp_pose' in outputs and 'grasp_label' in targets:
                pose_loss = self.criterions['pose'](outputs['grasp_pose'], targets['grasp_label'])
                loss_values['pose'] = pose_loss.item()
                total_loss += self.config.get('pose_weight', 1.0) * pose_loss
            
            # 质量损失
            if 'grasp_quality' in outputs and 'success' in targets:
                # 扩展标签以匹配输出维度
                quality_target = targets['success'].repeat(1, 3)
                quality_loss = self.criterions['quality'](outputs['grasp_quality'], quality_target)
                loss_values['quality'] = quality_loss.item()
                total_loss += self.config.get('quality_weight', 1.0) * quality_loss
            
            # 力控损失
            if 'force_prediction' in outputs and 'force_label' in targets:
                force_loss = self.criterions['force'](outputs['force_prediction'][:, :3], 
                                                     targets['force_label'])
                loss_values['force'] = force_loss.item()
                total_loss += self.config.get('force_weight', 0.5) * force_loss
        
        elif model_type == 'ForceControlNN':
            # 力预测损失
            if 'force' in outputs and 'force_target' in targets:
                force_loss = self.criterions['force'](outputs['force'], targets['force_target'])
                loss_values['force'] = force_loss.item()
                total_loss += force_loss
            
            # 刚度损失
            if 'stiffness' in outputs and 'stiffness_target' in targets:
                stiffness_loss = self.criterions['stiffness'](outputs['stiffness'], 
                                                             targets['stiffness_target'])
                loss_values['stiffness'] = stiffness_loss.item()
                total_loss += stiffness_loss
        
        elif model_type == 'SlipDetector':
            # 滑动分类损失
            if 'slip_probability' in outputs and 'slip_label' in targets:
                class_loss = self.criterions['classification'](outputs['slip_probability'], 
                                                              targets['slip_label'])
                loss_values['classification'] = class_loss.item()
                total_loss += class_loss
            
            # 滑动方向回归损失
            if 'slip_direction' in outputs and 'direction_label' in targets:
                reg_loss = self.criterions['regression'](outputs['slip_direction'], 
                                                        targets['direction_label'])
                loss_values['regression'] = reg_loss.item()
                total_loss += reg_loss
        
        elif model_type == 'ContactStateClassifier':
            # 分类损失
            if 'logits' in outputs and 'class_label' in targets:
                class_loss = self.criterions['classification'](outputs['logits'], 
                                                              targets['class_label'])
                loss_values['classification'] = class_loss.item()
                total_loss += class_loss
            
            # 力回归损失
            if 'force_estimate' in outputs and 'force_label' in targets:
                reg_loss = self.criterions['regression'](outputs['force_estimate'], 
                                                        targets['force_label'])
                loss_values['regression'] = reg_loss.item()
                total_loss += reg_loss
        
        else:
            # 默认损失计算
            if 'output' in outputs and 'target' in targets:
                default_loss = self.criterions['default'](outputs['output'], targets['target'])
                loss_values['default'] = default_loss.item()
                total_loss += default_loss
        
        loss_values['total'] = total_loss.item()
        return total_loss, loss_values
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {'total': 0.0}
        batch_count = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1} 训练")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 准备数据
            batch = self._prepare_batch(batch)
            
            # 前向传播
            outputs = self.model(*batch['inputs'])
            
            # 计算损失
            total_loss, batch_losses = self._compute_loss(outputs, batch['targets'])
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            grad_clip = self.config.get('grad_clip', 0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            # 更新学习率调度器（如果不是ReduceLROnPlateau）
            if self.scheduler and not isinstance(self.scheduler, 
                                               optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            
            # 记录损失
            batch_size = len(batch['inputs'][0]) if isinstance(batch['inputs'], tuple) else 1
            for key, value in batch_losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value * batch_size
            
            batch_count += batch_size
            
            # 更新进度条
            progress_bar.set_postfix({'loss': batch_losses['total']})
            
            # TensorBoard记录
            if self.writer is not None:
                for key, value in batch_losses.items():
                    self.writer.add_scalar(f'train/batch_{key}', value, self.global_step)
                
                # 记录学习率
                lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/learning_rate', lr, self.global_step)
            
            self.global_step += 1
        
        # 计算平均损失
        if batch_count > 0:
            for key in epoch_losses:
                epoch_losses[key] /= batch_count
        
        return epoch_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        
        val_losses = {'total': 0.0}
        batch_count = 0
        
        # 收集评估指标
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1} 验证")
            
            for batch_idx, batch in enumerate(progress_bar):
                # 准备数据
                batch = self._prepare_batch(batch)
                
                # 前向传播
                outputs = self.model(*batch['inputs'])
                
                # 计算损失
                _, batch_losses = self._compute_loss(outputs, batch['targets'])
                
                # 记录损失
                batch_size = len(batch['inputs'][0]) if isinstance(batch['inputs'], tuple) else 1
                for key, value in batch_losses.items():
                    if key not in val_losses:
                        val_losses[key] = 0.0
                    val_losses[key] += value * batch_size
                
                batch_count += batch_size
                
                # 收集预测结果用于计算指标
                self._collect_predictions(outputs, batch['targets'], 
                                         all_predictions, all_targets)
        
        # 计算平均损失
        if batch_count > 0:
            for key in val_losses:
                val_losses[key] /= batch_count
        
        # 计算评估指标
        metrics = self._compute_metrics(all_predictions, all_targets)
        val_losses.update(metrics)
        
        # TensorBoard记录
        if self.writer is not None:
            for key, value in val_losses.items():
                if key not in ['total', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
                    continue
                self.writer.add_scalar(f'val/{key}', value, self.epoch)
        
        return val_losses
    
    def _prepare_batch(self, batch: Dict) -> Dict:
        """准备批次数据（根据模型类型）"""
        model_type = self.config.get('model_type', 'GraspPredictor')
        
        inputs = []
        targets = {}
        
        if model_type == 'GraspPredictor':
            # 输入
            visual = batch['visual'].to(self.device)
            tactile = batch['tactile'].to(self.device)
            arm_state = batch['arm_state'].to(self.device)
            inputs = (visual, tactile, arm_state)
            
            # 目标
            targets['grasp_label'] = batch['grasp_label'].to(self.device)
            targets['success'] = batch['success'].to(self.device)
            if 'force_label' in batch:
                targets['force_label'] = batch['force_label'].to(self.device)
        
        elif model_type == 'ForceControlNN':
            # 输入：状态向量
            state = batch['state'].to(self.device)
            inputs = (state,)
            
            # 目标
            if 'force_target' in batch:
                targets['force_target'] = batch['force_target'].to(self.device)
            if 'stiffness_target' in batch:
                targets['stiffness_target'] = batch['stiffness_target'].to(self.device)
        
        elif model_type == 'SlipDetector':
            # 输入：时序触觉数据
            tactile_seq = batch['tactile_seq'].to(self.device)
            inputs = (tactile_seq,)
            
            # 目标
            if 'slip_label' in batch:
                targets['slip_label'] = batch['slip_label'].to(self.device)
            if 'direction_label' in batch:
                targets['direction_label'] = batch['direction_label'].to(self.device)
        
        elif model_type == 'ContactStateClassifier':
            # 输入：触觉数据
            tactile = batch['tactile'].to(self.device)
            inputs = (tactile,)
            
            # 目标
            if 'class_label' in batch:
                targets['class_label'] = batch['class_label'].to(self.device)
            if 'force_label' in batch:
                targets['force_label'] = batch['force_label'].to(self.device)
        
        else:
            # 默认处理
            inputs = (batch['input'].to(self.device),)
            targets = {'target': batch['target'].to(self.device)}
        
        return {'inputs': inputs, 'targets': targets}
    
    def _collect_predictions(self, outputs: Dict, targets: Dict, 
                           predictions_list: List, targets_list: List):
        """收集预测结果用于计算指标"""
        model_type = self.config.get('model_type', 'GraspPredictor')
        
        if model_type == 'GraspPredictor':
            if 'grasp_quality' in outputs and 'success' in targets:
                pred_probs = outputs['grasp_quality'][:, 0].cpu().numpy()
                true_labels = targets['success'].cpu().numpy().flatten()
                predictions_list.extend(pred_probs)
                targets_list.extend(true_labels)
        
        elif model_type in ['SlipDetector', 'ContactStateClassifier']:
            # 收集分类预测
            pass  # 根据具体需求实现
    
    def _compute_metrics(self, predictions: List, targets: List) -> Dict[str, float]:
        """计算评估指标"""
        if not predictions or not targets:
            return {}
        
        predictions_np = np.array(predictions)
        targets_np = np.array(targets)
        
        metrics = {}
        
        # 准确率（阈值0.5）
        if len(predictions_np.shape) == 1:  # 二元分类
            pred_binary = (predictions_np > 0.5).astype(int)
            accuracy = np.mean(pred_binary == targets_np)
            metrics['accuracy'] = accuracy
        
        # 尝试计算AUC
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
            
            if len(predictions_np.shape) == 1:
                # 二元分类指标
                auc = roc_auc_score(targets_np, predictions_np)
                f1 = f1_score(targets_np, pred_binary, zero_division=0)
                precision = precision_score(targets_np, pred_binary, zero_division=0)
                recall = recall_score(targets_np, pred_binary, zero_division=0)
                
                metrics.update({
                    'auc': auc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                })
                
        except (ImportError, ValueError) as e:
            warnings.warn(f"无法计算部分指标: {e}")
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = None) -> Dict[str, List[float]]:
        """完整训练流程"""
        if epochs is None:
            epochs = self.config.get('epochs', 100)
        
        logger.info(f"开始训练，总epoch数: {epochs}")
        logger.info(f"训练集大小: {len(train_loader.dataset)}")
        logger.info(f"验证集大小: {len(val_loader.dataset)}")
        
        for epoch in range(self.epoch, self.epoch + epochs):
            self.epoch = epoch + 1
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            self.train_history.append(train_losses)
            
            # 验证
            val_losses = self.validate(val_loader)
            self.val_history.append(val_losses)
            
            # 学习率调度（ReduceLROnPlateau）
            if self.scheduler and isinstance(self.scheduler, 
                                           optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_losses['total'])
            
            # 保存最佳模型
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pth')
                logger.info(f"💾 新的最佳模型保存，验证损失: {self.best_val_loss:.6f}")
            
            # 打印进度
            lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"📊 Epoch {self.epoch}/{self.epoch + epochs - 1}: "
                       f"训练损失={train_losses['total']:.6f}, "
                       f"验证损失={val_losses['total']:.6f}, "
                       f"准确率={val_losses.get('accuracy', 0):.4f}, "
                       f"学习率={lr:.6f}")
            
            # 定期保存检查点
            if self.epoch % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{self.epoch}.pth')
        
        # 训练完成
        logger.info("✅ 训练完成")
        self.save_checkpoint('final_model.pth')
        self.save_training_history()
        
        if self.writer is not None:
            self.writer.close()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config,
            'model_type': self.config.get('model_type', 'unknown')
        }
        
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, path)
        logger.info(f"💾 检查点已保存到 {path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        if not Path(path).exists():
            raise FileNotFoundError(f"检查点文件不存在: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        logger.info(f"📂 检查点已从 {path} 加载，epoch: {self.epoch}")
    
    def save_training_history(self, path: str = "training_history.json"):
        """保存训练历史"""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'config': self.config,
            'epochs': self.epoch,
            'best_val_loss': self.best_val_loss,
            'model_type': self.config.get('model_type', 'unknown')
        }
        
        # 转换numpy类型为Python原生类型
        def convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        
        history_serializable = convert(history)
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"📊 训练历史已保存到 {path}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """在测试集上评估模型"""
        self.model.eval()
        
        test_results = {
            'losses': {},
            'metrics': {},
            'predictions': [],
            'targets': []
        }
        
        total_loss = 0.0
        batch_count = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="测试评估"):
                # 准备数据
                batch = self._prepare_batch(batch)
                
                # 前向传播
                outputs = self.model(*batch['inputs'])
                
                # 计算损失
                batch_loss, _ = self._compute_loss(outputs, batch['targets'])
                
                # 记录
                batch_size = len(batch['inputs'][0]) if isinstance(batch['inputs'], tuple) else 1
                total_loss += batch_loss.item() * batch_size
                batch_count += batch_size
                
                # 收集预测结果
                self._collect_predictions(outputs, batch['targets'], 
                                         all_predictions, all_targets)
        
        # 计算平均损失
        if batch_count > 0:
            test_results['losses']['total'] = total_loss / batch_count
        
        # 计算指标
        if all_predictions and all_targets:
            metrics = self._compute_metrics(all_predictions, all_targets)
            test_results['metrics'] = metrics
        
        logger.info(f"📈 测试结果: 损失={test_results['losses'].get('total', 0):.6f}, "
                   f"准确率={test_results['metrics'].get('accuracy', 0):.4f}")
        
        return test_results


class MultiModelTrainer:
    """
    多模型训练器
    用于同时训练多个相关模型
    """
    
    def __init__(self, configs: Dict[str, Dict]):
        self.configs = configs
        self.trainers = {}
        
        # 为每个配置创建训练器
        for model_name, config in configs.items():
            self.trainers[model_name] = ModelTrainer(config)
    
    def train(self, dataloaders: Dict[str, Tuple[DataLoader, DataLoader]], 
              epochs: int = 100):
        """训练所有模型"""
        results = {}
        
        for model_name, trainer in self.trainers.items():
            if model_name in dataloaders:
                train_loader, val_loader = dataloaders[model_name]
                logger.info(f"开始训练模型: {model_name}")
                
                result = trainer.train(train_loader, val_loader, epochs)
                results[model_name] = result
        
        return results
    
    def save_all_checkpoints(self, base_path: str = "checkpoints"):
        """保存所有模型的检查点"""
        for model_name, trainer in self.trainers.items():
            path = f"{base_path}/{model_name}_checkpoint.pth"
            trainer.save_checkpoint(path)
    
    def load_all_checkpoints(self, base_path: str = "checkpoints"):
        """加载所有模型的检查点"""
        for model_name, trainer in self.trainers.items():
            path = f"{base_path}/{model_name}_checkpoint.pth"
            if Path(path).exists():
                trainer.load_checkpoint(path)


def create_dataloader(dataset, batch_size: int = 32, shuffle: bool = True, 
                      num_workers: int = 4) -> DataLoader:
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )


def get_default_config(model_type: str = 'GraspPredictor') -> Dict:
    """获取默认配置"""
    base_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 100,
        'batch_size': 32,
        'save_interval': 10,
        'use_tensorboard': True,
        'grad_clip': 1.0,
        'model_type': model_type
    }
    
    if model_type == 'GraspPredictor':
        base_config.update({
            'model_config': {
                'fusion_dim': 512
            },
            'optimizer': {
                'type': 'Adam',
                'learning_rate': 1e-3,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'factor': 0.5,
                'patience': 5
            },
            'pose_weight': 1.0,
            'quality_weight': 1.0,
            'force_weight': 0.5
        })
    
    elif model_type == 'ForceControlNN':
        base_config.update({
            'model_config': {
                'state_dim': 100,
                'hidden_dims': [256, 128, 64]
            },
            'optimizer': {
                'type': 'Adam',
                'learning_rate': 3e-4
            }
        })
    
    elif model_type == 'TactileCNN':
        base_config.update({
            'model_config': {
                'input_channels': 192,
                'output_dim': 128
            },
            'optimizer': {
                'type': 'Adam',
                'learning_rate': 1e-3
            }
        })
    
    return base_config