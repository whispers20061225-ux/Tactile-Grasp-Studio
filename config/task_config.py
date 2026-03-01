"""
任务配置文件
用于定义具体的机器人任务和实验流程
"""

class TaskConfig:
    """任务配置类"""
    
    # ========== 基本抓取任务 ==========
    BASIC_GRASP_TASK = {
        'name': 'basic_grasp',
        'description': '基本物体抓取任务',
        'steps': [
            {
                'action': 'move_to_ready',
                'params': {'speed': 30.0}
            },
            {
                'action': 'detect_object',
                'params': {'object_class': 'cube'}
            },
            {
                'action': 'plan_grasp',
                'params': {'method': 'top_down'}
            },
            {
                'action': 'execute_grasp',
                'params': {'force': 10.0}
            },
            {
                'action': 'move_to_place',
                'params': {'position': [0.5, 0.0, 0.3]}
            },
            {
                'action': 'release',
                'params': {}
            },
        ],
        'success_criteria': {
            'grasp_success': True,
            'object_lifted': True,
            'placement_accuracy': 0.02,  # 米
        },
    }
    
    # ========== 装配任务 ==========
    ASSEMBLY_TASK = {
        'name': 'peg_in_hole',
        'description': '轴孔装配任务',
        'steps': [
            {
                'action': 'locate_peg_and_hole',
                'params': {}
            },
            {
                'action': 'fine_alignment',
                'params': {'tolerance': 0.001}
            },
            {
                'action': 'insert_with_force_control',
                'params': {'max_force': 5.0}
            },
        ],
        'success_criteria': {
            'insertion_depth': 0.02,  # 米
            'max_force': 10.0,  # 牛
            'completion_time': 30.0,  # 秒
        },
    }
    
    # ========== 实验记录配置 ==========
    EXPERIMENT_RECORDING = {
        'enabled': True,
        'save_path': 'experiment_data/',
        'data_to_record': [
            'timestamps',
            'joint_positions',
            'joint_velocities',
            'joint_torques',
            'tactile_data',
            'camera_images',
            'grasp_success',
            'completion_time',
            'force_profiles',
        ],
        'video_recording': {
            'enabled': True,
            'resolution': [1280, 720],
            'fps': 30,
        },
        'metadata': {
            'operator': 'default',
            'environment': 'lab',
            'lighting': 'controlled',
            'object_properties': 'recorded',
        },
    }

def create_default_task_config():
    """创建默认的任务配置"""
    return TaskConfig()


# 测试代码
if __name__ == "__main__":
    config = create_default_task_config()
    print("任务配置测试:")
    print(f"基本抓取任务步骤数: {len(config.BASIC_GRASP_TASK['steps'])}")