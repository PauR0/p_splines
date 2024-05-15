import pathlib

from setuptools import find_packages, setup

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="p_splines",
    license="MIT",
    version="0.1.0",
    packages=find_packages(),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Pau Romero",
    author_email="pau.romero@uv.es",
    python_requires=">=3.11",
    setup_requires=["setuptools_scm"],
    install_requires=["matplotlib", "numpy", "pandas", "cpsplines", "scipy"],
    extras_require={
        "dev": ["black", "ipykernel >= 6.25.1", "pip-tools", "pytest", "ipywidgets"]
    },
)
