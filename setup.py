from setuptools import setup, find_packages

REQUIREMENTS = ["numpy", "pandas", "sklearn", "matplotlib"]

with open("README.md", mode="r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="data_toolbox",
    version="0.0.2",
    author="Haoyin Xu, Jacob Desman, Muna Igboko, Qianqi Huang",
    author_email="haoyinxu@gmail.com",
    maintainer="Haoyin Xu",
    maintainer_email="haoyinxu@gmail.com",
    description="Team Cyan's function toolbox for data selection and modeling",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/PCM5/Data-Toolbox",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
    ],
    install_requires=REQUIREMENTS,
    packages=find_packages(),
)
