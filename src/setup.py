from setuptools import setup, find_packages

setup(
    name="iluvatar_s4_core",
    version="0.1.0",
    # Automatically finds the 'scripts' and 'src' folders and treats them as importable Python packages globally
    packages=find_packages(),
)
