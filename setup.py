from setuptools import setup, find_packages

setup(
    name="arcface_contrastive",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "mtcnn",
        "numpy",
        "opencv-python",
        "Pillow"
    ],
)
