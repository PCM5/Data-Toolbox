from setuptools import setup, find_packages

REQUIREMENTS = ["numpy","pandas"]

with open("README.md", mode="r", encoding="utf8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="data_toolbox",
    version="0.0.1",
    author="Haoyin Xu, Jacob Desman",
    asuthor_email="haoyinxu@gmail.com",
    description="Team Cyan's function toolbox for exploring databases",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/PCM5/Data-Toolbox",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3",
    ],
    install_requires=REQUIREMENTS,
)
