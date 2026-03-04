from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_world = PathJoinSubstitution(
        [FindPackageShare("tactile_sim"), "worlds", "phase6_tabletop.world"]
    )
    default_xacro = PathJoinSubstitution(
        [FindPackageShare("tactile_sim"), "urdf", "phase6_arm.urdf.xacro"]
    )

    world_arg = DeclareLaunchArgument(
        "world",
        default_value=default_world,
        description="Gazebo Sim world file",
    )
    xacro_file_arg = DeclareLaunchArgument(
        "xacro_file",
        default_value=default_xacro,
        description="Robot xacro file",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )
    entity_name_arg = DeclareLaunchArgument(
        "entity_name",
        default_value="phase6_arm",
        description="Spawned entity name",
    )

    world = LaunchConfiguration("world")
    xacro_file = LaunchConfiguration("xacro_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    entity_name = LaunchConfiguration("entity_name")

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
            )
        ),
        launch_arguments={"gz_args": ["-r ", world]}.items(),
    )

    robot_description = Command([FindExecutable(name="xacro"), " ", xacro_file])

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_phase6_arm",
        output="screen",
        arguments=[
            "-name",
            entity_name,
            "-topic",
            "robot_description",
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.42",
        ],
    )

    spawner_joint_state = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_joint_state_broadcaster",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "120",
        ],
    )

    spawner_joint_trajectory = Node(
        package="controller_manager",
        executable="spawner",
        name="spawner_joint_trajectory_controller",
        output="screen",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "120",
        ],
    )

    delay_joint_state = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_entity,
            on_exit=[spawner_joint_state],
        )
    )

    delay_joint_trajectory = RegisterEventHandler(
        OnProcessExit(
            target_action=spawner_joint_state,
            on_exit=[spawner_joint_trajectory],
        )
    )

    return LaunchDescription(
        [
            world_arg,
            xacro_file_arg,
            use_sim_time_arg,
            entity_name_arg,
            gazebo_launch,
            robot_state_publisher,
            spawn_entity,
            delay_joint_state,
            delay_joint_trajectory,
        ]
    )
