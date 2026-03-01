from setuptools import find_packages, setup


package_name = "tactile_ui_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="whisp",
    maintainer_email="whisp@users.noreply.github.com",
    description="ROS2 to UI bridge for tactile streams.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "tactile_ui_subscriber = tactile_ui_bridge.tactile_subscriber:main",
        ],
    },
)
