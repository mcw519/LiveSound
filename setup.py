from pathlib import Path

from setuptools import find_packages, setup

labtool_dir = Path(__file__).parent

install_requires = (labtool_dir / "requirements.txt").read_text().splitlines()

setup(
    name="LiveSound",
    version="1.0.0",
    python_requires=">=3.6.0",
    description="",
    author="Meng Wu",
    packages=find_packages(),
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "record-local=livesound.scripts.simple_recorder:recorder",
            "play-local=livesound.scripts.simple_player:player",
        ]
    },
)
