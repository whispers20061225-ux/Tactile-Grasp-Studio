from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    tactile_hardware_launch = PathJoinSubstitution(
        [FindPackageShare("tactile_bringup"), "launch", "tactile_hardware_only.launch.py"]
    )
    default_config = PathJoinSubstitution(
        [FindPackageShare("tactile_bringup"), "config", "tactile_hardware_only.yaml"]
    )

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description="Path to the tactile hardware-only parameter YAML",
    )
    start_web_gateway_arg = DeclareLaunchArgument(
        "start_web_gateway",
        default_value="true",
        description="Start the tactile web gateway together with the hardware nodes",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="ROS log level used by the tactile hardware-only stack",
    )

    config_file = LaunchConfiguration("config_file")
    start_web_gateway = LaunchConfiguration("start_web_gateway")
    log_level = LaunchConfiguration("log_level")

    return LaunchDescription(
        [
            config_file_arg,
            start_web_gateway_arg,
            log_level_arg,
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(tactile_hardware_launch),
                launch_arguments={
                    "config_file": config_file,
                    "start_web_gateway": start_web_gateway,
                    "log_level": log_level,
                }.items(),
            ),
        ]
    )
