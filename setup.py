from setuptools import setup, find_packages

setup(
    name="poker_rl_bot",
    version="0.1.0",
    description="Texas Hold'em RL bot using PPO and self-play",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "gym==0.26.2",
        "stable-baselines3==2.2.1",
        "torch==2.1.0",
        "treys==0.1.8",
        "numpy==1.24.3",
        "pytest==7.4.3",
        "pyyaml==6.0.1",
    ],
    python_requires=">=3.8",
)