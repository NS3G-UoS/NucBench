from setuptools import setup, find_packages

setup(
    name="nucbench",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "litellm>=1.40.0",
        "streamlit>=1.35.0",
    ],
    entry_points={
        "console_scripts": [
            "nucbench=nucbench.cli:main",
        ],
    },
    python_requires=">=3.10",
)
