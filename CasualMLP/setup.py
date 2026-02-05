"""
CausalMLP Setup Script / Kịch bản Cài đặt CausalMLP

Install with / Cài đặt với:
    pip install .
    pip install -e .  # Development mode / Chế độ phát triển
    pip install .[dev]  # With dev dependencies / Với các phụ thuộc phát triển
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description / Đọc README cho mô tả dài
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Core dependencies / Các phụ thuộc cốt lõi
install_requires = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
]

# Optional dependencies / Các phụ thuộc tùy chọn
extras_require = {
    "full": [
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "networkx>=2.6.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.990",
        "isort>=5.10.0",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "myst-parser>=0.18.0",
    ],
}

# All optional dependencies / Tất cả các phụ thuộc tùy chọn
extras_require["all"] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

setup(
    name="causalmlp",
    version="1.0.0",
    author="CausalMLP Contributors",
    author_email="your.email@example.com",
    description="Advanced Hybrid Causal Discovery Model combining DECI and GraN-DAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/CausalMLP",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/CausalMLP/issues",
        "Documentation": "https://github.com/yourusername/CausalMLP#readme",
        "Source Code": "https://github.com/yourusername/CausalMLP",
    },
    license="MIT",
    
    # Package discovery / Khám phá gói
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    py_modules=["config", "train"],
    
    # Dependencies / Phụ thuộc
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    
    # Entry points / Điểm nhập
    entry_points={
        "console_scripts": [
            "causalmlp-train=train:main",
        ],
    },
    
    # Package data / Dữ liệu gói
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml"],
    },
    
    # Classifiers / Phân loại
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    
    # Keywords / Từ khóa
    keywords=[
        "causal-discovery",
        "causal-inference",
        "dag-learning",
        "structure-learning",
        "bayesian-networks",
        "deep-learning",
        "pytorch",
        "notears",
        "deci",
        "gran-dag",
    ],
)
