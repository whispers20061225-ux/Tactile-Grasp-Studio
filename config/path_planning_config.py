"""
路径规划配置文件
用于配置运动规划算法和参数
"""

class PathPlanningConfig:
    """路径规划配置类"""
    
    # ========== 规划器类型配置 ==========
    PLANNER_TYPE = {
        'global_planner': 'rrt_connect',  # 'rrt', 'rrt_star', 'rrt_connect', 'prm', 'est'
        'local_planner': 'trajopt',  # 'trajopt', 'cartesian', 'linear'
        'use_moveit': False,  # 是否使用MoveIt!规划器
    }
    
    # ========== RRT系列算法配置 ==========
    RRT_CONFIG = {
        'rrt': {
            'goal_bias': 0.05,  # 偏向目标的概率
            'max_iterations': 5000,
            'step_size': 0.05,  # 米或弧度
            'goal_tolerance': 0.01,  # 米或弧度
            'max_planning_time': 5.0,  # 秒
        },
        'rrt_star': {
            'rewire_factor': 1.1,
            'radius': 0.5,
            'delay_collision_checking': True,
        },
        'rrt_connect': {
            'range': 0.5,
            'bidirectional': True,
        },
    }
    
    # ========== PRM算法配置 ==========
    PRM_CONFIG = {
        'num_samples': 1000,
        'num_neighbors': 10,
        'roadmap_type': 'uniform',  # 'uniform', 'gaussian', 'bridge'
        'connection_radius': 0.5,
        'lazy_collision_checking': True,
    }
    
    # ========== 轨迹优化配置 ==========
    TRAJECTORY_OPTIMIZATION = {
        'method': 'trajopt',  # 'trajopt', 'chomp', 'stomp'
        'max_iterations': 100,
        'collision_cost_weight': 10.0,
        'smoothness_cost_weight': 1.0,
        'constraint_tolerance': 1e-3,
        'trust_region_radius': 0.1,
    }
    
    # ========== 笛卡尔路径规划配置 ==========
    CARTESIAN_PLANNING = {
        'waypoint_generation': {
            'method': 'linear',  # 'linear', 'circular', 'spline'
            'max_step_translation': 0.01,  # 米
            'max_step_rotation': 0.01,  # 弧度
            'jump_threshold': 1.5,
        },
        'orientation_interpolation': {
            'method': 'slerp',  # 'slerp', 'linear'
            'use_axis_angle': False,
        },
    }
    
    # ========== 碰撞检测配置 ==========
    COLLISION_DETECTION = {
        'enabled': True,
        'check_self_collision': True,
        'check_environment_collision': True,
        'collision_margin': 0.01,  # 米
        'continuous_collision_detection': True,
        'broadphase_algorithm': 'sweep_and_prune',  # 'sweep_and_prune', 'dynamic_aabb_tree'
        'collision_pairs': [
            'arm_link_1 - arm_link_3',
            'arm_link_2 - gripper_finger_1',
            # 添加其他需要检查的碰撞对
        ],
    }
    
    # ========== 工作空间约束配置 ==========
    WORKSPACE_CONSTRAINTS = {
        'position_limits': {
            'x': [-0.9, 0.9],  # 米
            'y': [-0.9, 0.9],
            'z': [0.0, 0.9],
        },
        'orientation_limits': {
            'roll': [-3.14, 3.14],  # 弧度
            'pitch': [-1.57, 1.57],
            'yaw': [-3.14, 3.14],
        },
        'joint_limits': {
            'joint1': [-2.97, 2.97],  # 弧度
            'joint2': [-2.09, 2.09],
            'joint3': [-2.97, 2.97],
            'joint4': [-2.09, 2.09],
            'joint5': [-2.97, 2.97],
            'joint6': [-6.28, 6.28],
        },
        'velocity_limits': {
            'joint1': 1.0,  # 弧度/秒
            'joint2': 1.0,
            'joint3': 1.0,
            'joint4': 1.0,
            'joint5': 1.0,
            'joint6': 2.0,
        },
        'acceleration_limits': {
            'joint1': 3.0,  # 弧度/秒²
            'joint2': 3.0,
            'joint3': 3.0,
            'joint4': 3.0,
            'joint5': 3.0,
            'joint6': 6.0,
        },
        'jerk_limits': {
            'joint1': 10.0,  # 弧度/秒³
            'joint2': 10.0,
            'joint3': 10.0,
            'joint4': 10.0,
            'joint5': 10.0,
            'joint6': 20.0,
        },
    }
    
    # ========== 抓取规划配置 ==========
    GRASP_PLANNING = {
        'approach_strategy': {
            'method': 'top_down',  # 'top_down', 'side', 'angle'
            'approach_distance': 0.05,  # 米
            'retreat_distance': 0.1,  # 米
            'approach_angle': 0.0,  # 弧度
        },
        'grasp_candidate_generation': {
            'method': 'sampling',  # 'sampling', 'learning_based', 'template'
            'num_candidates': 20,
            'max_sampling_attempts': 100,
            'force_closure_threshold': 0.5,
        },
        'grasp_selection': {
            'criteria': ['force_closure', 'collision_free', 'reachability'],
            'weights': [0.4, 0.3, 0.3],
            'quality_threshold': 0.6,
        },
        'grasp_execution': {
            'pregrasp_pause': 0.5,  # 秒
            'grasp_force': 10.0,  # 牛
            'settle_time': 0.2,  # 秒
        },
    }
    
    # ========== 避障配置 ==========
    OBSTACLE_AVOIDANCE = {
        'obstacle_inflation_radius': 0.05,  # 米
        'potential_field': {
            'enabled': True,
            'attractive_gain': 1.0,
            'repulsive_gain': 0.5,
            'influence_distance': 0.3,  # 米
        },
        'dynamic_obstacles': {
            'prediction_horizon': 2.0,  # 秒
            'safety_margin': 0.1,  # 米
        },
    }
    
    # ========== 轨迹生成配置 ==========
    TRAJECTORY_GENERATION = {
        'time_parameterization': {
            'method': 'trapz',  # 'trapz', 'cubic', 'quintic'
            'max_velocity_scaling': 0.5,
            'max_acceleration_scaling': 0.3,
            'blend_radius': 0.02,  # 米
        },
        'smoothing': {
            'enabled': True,
            'method': 'chomp',  # 'chomp', 'shortcut', 'bspline'
            'smoothing_iterations': 50,
        },
        'interpolation': {
            'points_per_segment': 10,
            'interpolation_method': 'cubic',  # 'linear', 'cubic', 'quintic'
        },
    }
    
    # ========== 运动学配置 ==========
    KINEMATICS = {
        'inverse_kinematics_solver': 'numeric',  # 'analytic', 'numeric', 'ikfast'
        'ik_timeout': 0.1,  # 秒
        'ik_attempts': 10,
        'ik_tolerance': 0.01,  # 米和弧度
        'redundancy_resolution': {
            'method': 'null_space',  # 'null_space', 'optimization'
            'preferred_configuration': [0.0, -1.57, 0.0, -1.57, 0.0, 0.0],
        },
    }
    
    # ========== 实时性配置 ==========
    REALTIME = {
        'planning_timeout': 10.0,  # 秒
        'execution_monitoring': True,
        'replanning_threshold': 0.02,  # 米
        'controller_frequency': 100,  # 赫兹
        'buffer_size': 10,  # 轨迹点缓冲区
    }
    
    # ========== 可视化配置 ==========
    VISUALIZATION = {
        'show_planned_path': True,
        'show_collision_points': True,
        'show_waypoints': True,
        'show_workspace': True,
        'color_scheme': {
            'planned_path': [0.0, 1.0, 0.0],  # 绿色
            'executed_path': [1.0, 0.0, 0.0],  # 红色
            'collision_points': [1.0, 0.0, 1.0],  # 紫色
            'waypoints': [0.0, 0.0, 1.0],  # 蓝色
        },
    }

def create_default_path_planning_config():
    """创建默认的路径规划配置"""
    return PathPlanningConfig()


# 测试代码
if __name__ == "__main__":
    config = create_default_path_planning_config()
    print("路径规划配置测试:")
    print(f"全局规划器: {config.PLANNER_TYPE['global_planner']}")
    print(f"碰撞检测: {config.COLLISION_DETECTION['enabled']}")