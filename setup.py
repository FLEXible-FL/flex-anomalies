from setuptools import find_packages, setup

setup(
    name="flexanomalies",
    version="0.0.1",
    authors="Ignacio Aguilera Martos and Beatriz Bello García",
    keywords="anomaly detection federated-learning flexible time series ",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=["flex", "numpy", "tensorflow" , "scikit-learn", "pandas"],
)