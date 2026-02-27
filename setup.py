from setuptools import setup, find_packages
import os

setup(
    name="yolo-auto-center",
    version="0.1.0",
    description="Real-time object auto-centering using YOLO with visual feedback and group centroid support.",
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "ultralytics",
        "opencv-python",
        "pyyaml",
    ],
)