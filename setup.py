from setuptools import setup, find_packages

setup(
    name="credit-approval",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)