from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description() -> LaunchDescription:
    param_file = PathJoinSubstitution(
        [FindPackageShare("tactile_bringup"), "config", "phase1_fake_chain.yaml"]
    )

    fake_tactile_publisher = Node(
        package="tactile_bringup",
        executable="fake_tactile_publisher",
        name="fake_tactile_publisher",
        output="screen",
        parameters=[param_file],
    )

    tactile_ui_subscriber = Node(
        package="tactile_ui_bridge",
        executable="tactile_ui_subscriber",
        name="tactile_ui_subscriber",
        output="screen",
        parameters=[param_file],
    )

    return LaunchDescription([fake_tactile_publisher, tactile_ui_subscriber])
