from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pygaze",
    version="0.1.0",
	license="MIT",
    author="Jonas Freiknecht",
    author_email="j.freiknecht@googlemail.com",
    description="pygaze is a gaze estimation framework for python based on eth-xgaze.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/padmalcom/pygaze",
    packages=find_packages(exclude=("tests", "requirements.txt",)),
	include_package_data=True,
	install_requires=[
		"loguru>=0.6.0",
		"mediapipe>=0.9.0.1",
		"omegaconf>=2.3.0",
		"scipy>=1.9.3",
		"timm>=0.6.12",
		"torch>=1.13.0"
	],
    classifiers=[
        "Development Status :: 4 - Beta",
		"Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"Programming Language :: Python :: 3.10"
    ],
    python_requires='>=3.8',
)