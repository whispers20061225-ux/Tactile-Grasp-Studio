"""
仿真配置文件
用于配置PyBullet或CoppeliaSim仿真环境
"""

class SimulationConfig:
    """仿真配置类"""
    
    # ========== 仿真引擎配置 ==========
    ENGINE = {
        'type': 'pybullet',  # 'pybullet', 'coppeliasim', 'gazebo'
        'mode': 'gui',  # 'gui', 'direct', 'server'
        'physics_engine': 'bullet',  # 'bullet', 'ode', 'dart'
        'real_time_simulation': True,
        'update_frequency': 240,  # 赫兹
        'gravity': [0.0, 0.0, -9.81],  # 米/秒²
        'time_step': 1.0/240.0,  # 秒
    }
    
    # ========== 场景配置 ==========
    SCENE = {
        'floor_enabled': True,
        'floor_texture': 'checkerboard',
        'light_position': [2.0, 2.0, 2.0],
        'light_color': [1.0, 1.0, 1.0],
        'ambient_light': 0.6,
        'background_color': [0.8, 0.9, 1.0],  # RGB
        'workspace_bounds': {
            'x': [-1.0, 1.0],  # 米
            'y': [-1.0, 1.0],
            'z': [0.0, 1.0],
        },
    }
    
    # ========== 机械臂仿真配置 ==========
    ARM_SIMULATION = {
        'urdf_path': 'models/dofbot.urdf',
        'base_position': [0.0, 0.0, 0.0],
        'base_orientation': [0.0, 0.0, 0.0, 1.0],  # 四元数
        'global_scaling': 2.0,   #缩放倍数
        'initial_joint_positions_deg': [-90.0, 0.0, 0.0, 0.0, -90.0, 0.0],
        'joint_zero_offsets_deg': [-90.0, 0.0, 0.0, 0.0, -90.0, 0.0],
        'use_fixed_base': True,
        'collision_margin': 0.001,  # 米
        'contact_stiffness': 10000.0,
        'contact_damping': 100.0,
        'joint_control_mode': 'position',  # 'position', 'velocity', 'torque'
        'joint_friction': 0.1,
        'joint_damping': 0.1,
    }
    
    # ========== 夹爪仿真配置 ==========
    GRIPPER_SIMULATION = {
        'type': 'pneumatic',  # 'pneumatic', 'electric', 'custom'
        'urdf_path': 'models/gripper/gripper.urdf',
        'mount_position': [0.0, 0.0, 0.1],  # 相对于机械臂末端的偏移
        'finger_count': 2,
        'finger_length': 0.1,  # 米
        'max_opening': 0.12,  # 米
        'grasp_force': 20.0,  # 牛
        'contact_stiffness': 1000.0,
        'contact_damping': 10.0,
        'slider_crank_mechanism': True,  # 是否使用滑块曲柄机构
    }
    
    # ========== 触觉传感器仿真配置 ==========
    TACTILE_SIMULATION = {
        'enabled': True,
        'type': 'paxini_gen3',
        'sensor_count': 16,
        'taxel_spacing': 0.005,  # 米
        'force_range': [0.0, 10.0],  # 牛
        'noise_level': 0.1,  # 噪声级别（标准差）
        'update_rate': 100,  # 赫兹
        'visualization': {
            'enabled': True,
            'color_map': 'viridis',
            'max_force_color': [1.0, 0.0, 0.0],  # 红色
            'min_force_color': [0.0, 0.0, 1.0],  # 蓝色
        },
    }
    
    # ========== 物体仿真配置 ==========
    OBJECT_SIMULATION = {
        'default_density': 1000.0,  # 千克/立方米
        'default_friction': {
            'lateral': 0.5,
            'spinning': 0.1,
            'rolling': 0.01,
        },
        'default_restitution': 0.1,
        'object_types': {
            'cube': {
                'size_range': [0.02, 0.1],  # 米
                'mass_range': [0.01, 0.5],  # 千克
            },
            'sphere': {
                'radius_range': [0.01, 0.05],
                'mass_range': [0.01, 0.3],
            },
            'cylinder': {
                'radius_range': [0.01, 0.04],
                'height_range': [0.02, 0.1],
                'mass_range': [0.01, 0.4],
            },
        },
        'random_textures': True,
        'texture_path': 'textures/',
    }
    
    # ========== 物理引擎高级配置 ==========
    PHYSICS_ADVANCED = {
        'solver_iterations': 50,
        'solver_residual_threshold': 1e-7,
        'constraint_solver': 'sequential_impulse',
        'warm_start_factor': 0.9,
        'split_impulse': True,
        'contact_slop': 0.001,
        'enable_caching': True,
        'max_sub_steps': 10,
    }
    
    # ========== 可视化配置 ==========
    VISUALIZATION = {
        'camera_view': {
            'camera_distance': 1.5,  # 米
            'camera_yaw': 45.0,  # 度
            'camera_pitch': -30.0,  # 度
            'camera_target': [0.0, 0.0, 0.5],  # 米
        },
        'debug_lines': {
            'contact_points': False,
            'joint_axes': True,
            'coordinate_frames': True,
            'trajectory_path': True,
        },
        'rendering': {
            'shadows': True,
            'anti_aliasing': 4,
            'multi_samples': 4,
            'max_texture_size': 1024,
        },
    }
    
    # ========== 数据记录配置 ==========
    DATA_RECORDING = {
        'enabled': True,
        'record_video': False,
        'video_fps': 30,
        'video_path': 'simulation_videos/',
        'log_states': True,
        'log_frequency': 100,  # 赫兹
        'log_path': 'simulation_logs/',
        'log_variables': [
            'joint_positions',
            'joint_velocities',
            'joint_torques',
            'contact_forces',
            'object_positions',
            'gripper_force',
        ],
    }
    
    # ========== 性能配置 ==========
    PERFORMANCE = {
        'multithreading': True,
        'num_threads': 4,
        'parallel_solver': True,
        'use_gpu': False,
        'memory_allocation': 'dynamic',
        'cache_size': 1024,
    }

def create_default_sim_config():
    """创建默认的仿真配置"""
    return SimulationConfig()


# 测试代码
if __name__ == "__main__":
    config = create_default_sim_config()
    print("仿真配置测试:")
    print(f"物理引擎: {config.ENGINE['physics_engine']}")
    print(f"时间步长: {config.ENGINE['time_step']}")
