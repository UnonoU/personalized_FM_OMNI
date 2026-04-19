from setuptools import setup, find_packages

setup(
    name="rest-tuning",
    version="0.1.0",
    description="EEG Model Training and Fine-tuning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # 添加你的依赖项
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "tensorboardX",
        "h5py",
        "peft",
        "mne",
        "pyhealth",
        "timm",
    ],
)
