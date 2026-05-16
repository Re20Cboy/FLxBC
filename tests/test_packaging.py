import tomllib
from pathlib import Path


def test_core_install_keeps_heavy_runtime_dependencies_optional():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    core_dependencies = set(pyproject["project"]["dependencies"])
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    heavy_packages = {"flwr", "medmnist", "torch", "torchvision", "web3"}
    assert not any(
        dependency.split("[", 1)[0].split(">=", 1)[0] in heavy_packages
        for dependency in core_dependencies
    )
    assert {"app", "ml", "chain", "dev"}.issubset(optional_dependencies)
