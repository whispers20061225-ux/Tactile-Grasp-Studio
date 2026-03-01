"""
LEARN机械臂配置文件
用于配置机械臂的连接参数、运动参数和安全限制
"""

class LearmArmConfig:
    """LEARN机械臂配置类"""
    
    # ========== 连接配置 ==========
    CONNECTION = {
        'type': 'serial',
        'serial_port': '/dev/cu.usbserial-110',
        'baud_rate': 115200,
        'ethernet_ip': '192.168.1.100',
        'ethernet_port': 502,
        'timeout': 3.0,  # 秒
        'retry_attempts': 3,
        # 关节零点偏移（度）：以你当前标定姿态作为 0°
        'joint_zero_offsets_deg': [86.73, 180.0, -54.25, 46.31, 270.0, 22.09],
        # 相机->夹爪平移（mm），来自你的实测值 0.0cm, 4.8cm, 12cm
        'camera_to_gripper_offset_mm': [0.0, 48.0, 120.0],
        # 关节5为0°时的相机姿态偏置（roll, pitch, yaw，单位：度）
        'camera_rotation_offset_rpy_deg': [0.0, 0.0, 0.0],
        # ??????5???????????????????????????False?
        'camera_use_joint5_rotation': False,
        # ?????????->????True=????????False=????????
        'camera_offset_is_gripper_to_camera': False,
        # 关节5旋转轴（相机坐标系）
        'camera_joint_axis': [0.0, 0.0, 1.0],
        # 平移向量是否跟随关节旋转（相机坐标系测量建议为True）
        'camera_translation_rotate': True,
        # 对指定关节禁用角度软限制（允许越过默认 0~180/270 映射）
        'joint_limit_disabled_ids': [1, 3],
    }
    
    # ========== 机械臂物理参数 ==========
    PHYSICAL = {
        'dof': 6,  # 自由度
        'max_payload': 3.0,  # 千克
        'reach': 900.0,  # 毫米
        'repeatability': 0.05,  # 毫米
        'joint_limits': {  # 关节角度限制（度）
            'joint1': [-360, 360],  # 放宽J1限制，便于测试负角度
            'joint2': [-120, 120],
            'joint3': [-360, 360],  # 放宽J3限制，便于测试负角度
            'joint4': [-120, 120],
            'joint5': [-170, 170],
            'joint6': [-360, 360],
        },
        'speed_limits': {  # 关节速度限制（度/秒）
            'joint1': 100.0,
            'joint2': 100.0,
            'joint3': 100.0,
            'joint4': 100.0,
            'joint5': 100.0,
            'joint6': 100.0,
        },
        'acceleration_limits': {  # 关节加速度限制（度/秒²）
            'joint1': 500.0,
            'joint2': 500.0,
            'joint3': 500.0,
            'joint4': 500.0,
            'joint5': 500.0,
            'joint6': 500.0,
        },
    }
    
    # ========== 运动控制配置 ==========
    MOTION = {
        'default_speed': 30.0,  # 默认速度百分比
        'default_acceleration': 30.0,  # 默认加速度百分比
        'blending_radius': 5.0,  # 毫米，路径混合半径
        'motion_mode': 'linear',  # 'joint', 'linear', 'circular'
        'coordinate_frame': 'world',  # 'world', 'tool', 'user'
        'home_position': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 关节空间回零位置
        'ready_position': [0.0, -90.0, 90.0, 0.0, 90.0, 0.0],  # 就绪位置
    }
    
    # ========== 工具坐标系配置 ==========
    TOOL = {
        'name': 'gripper_tool',
        'center_point': [0.0, 0.0, 100.0],  # TCP相对于法兰中心的偏移（毫米）
        'mass': 0.5,  # 千克
        'cog': [0.0, 0.0, 50.0],  # 重心位置
        'inertia': [0.001, 0.001, 0.001, 0.0, 0.0, 0.0],  # 惯性矩阵
    }
    
    # ========== 用户坐标系配置 ==========
    USER_FRAMES = {
        'table_frame': {
            'position': [500.0, 0.0, 0.0],  # 毫米
            'orientation': [0.0, 0.0, 0.0],  # 欧拉角（度）
        },
        'conveyor_frame': {
            'position': [300.0, 400.0, 0.0],
            'orientation': [0.0, 0.0, 45.0],
        },
    }
    
    # ========== 安全配置 ==========
    SAFETY = {
        'collision_detection': True,
        'torque_limit_factor': 0.8,  # 扭矩限制系数
        'emergency_stop_timeout': 0.5,  # 秒
        'workspace_limits': {  # 笛卡尔空间工作范围（毫米）
            'x': [-900, 900],
            'y': [-900, 900],
            'z': [0, 900],
        },
        'singularity_avoidance': True,
        'self_collision_check': True,
    }
    
    # ========== 通信协议配置 ==========
    PROTOCOL = {
        'command_format': 'json',  # 'json', 'modbus', 'binary'
        'polling_rate': 100,  # 赫兹，状态轮询频率
        'heartbeat_interval': 1.0,  # 秒
        'buffer_size': 1024,  # 字节
        'ack_timeout': 0.1,  # 秒
    }
    
    # ========== ROS配置（如使用） ==========
    ROS = {
        'enabled': False,
        'namespace': '/learm_arm',
        'joint_state_topic': '/joint_states',
        'command_topic': '/joint_command',
        'move_group_name': 'learm_arm',
        'planning_time': 5.0,  # 秒
        'planning_attempts': 3,
    }
    
    # ========== 仿真配置 ==========
    SIMULATION = {
        'physics_engine': 'pybullet',  # 'pybullet', 'gazebo', 'coppeliasim'
        'gravity': [0.0, 0.0, -9.81],
        'time_step': 1.0/240.0,  # 秒
        'real_time_factor': 1.0,
        'visualization': True,
        'urdf_path': 'models/learm_arm.urdf',
    }
    
    # ========== 校准配置 ==========
    CALIBRATION = {
        'auto_calibration': False,
        'zero_position_tolerance': 0.1,  # 度
        'tcp_calibration_points': 4,  # TCP标定点数量
        'joint_compensation': True,  # 关节补偿
    }
    
def create_default_learm_config():
    """创建默认的LEARN机械臂配置"""
    return LearmArmConfig()


# 测试代码
if __name__ == "__main__":
    config = create_default_learm_config()
    print("LEARN机械臂配置测试:")
    print(f"连接类型: {config.CONNECTION['type']}")
    print(f"自由度: {config.PHYSICAL['dof']}")
    print(f"工作空间范围: {config.SAFETY['workspace_limits']}")
