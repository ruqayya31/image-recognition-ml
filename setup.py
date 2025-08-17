"""
Setup script for Image Recognition with Machine Learning Algorithm
Authors: MEHEK A, NASREEN T S, NAVJOT KAUR, SAYADA RUQAYYA
Institution: R.L. Jalappa Institute of Technology
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="image-recognition-ml",
    version="1.0.0",
    author="MEHEK A, NASREEN T S, NAVJOT KAUR, SAYADA RUQAYYA",
    author_email="contact@rljalappa.edu.in",
    description="Image Recognition using Machine Learning Algorithms - Mini Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/image-recognition-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="machine learning, image recognition, computer vision, classification, data science",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/image-recognition-ml/issues",
        "Source": "https://github.com/yourusername/image-recognition-ml",
        "Documentation": "https://github.com/yourusername/image-recognition-ml/blob/main/README.md",
    },
    package_data={
        "": [".md", ".txt"],
    },
    include_package_data=True,
)
