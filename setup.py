from setuptools import setup, find_packages
with open("README.md", "r") as f:
    long_description = f.read()
setup(
    name='DetectVideoShotLength', # this is the package folder that contain the "main.py"
    description="A lightweight python package that analyzes shot lengths for video.",

    version='0.12', # to increment this version number for each new version of this package
    packages=find_packages(), 
    long_description=long_description,
    long_description_content_type="text/markdown",

    # install_requires=[ # dependencies for this package
    # # e.g. 'numpy>=1.11.1'
    # ],
    author="Rakib Mollah",
    author_email="rakib1703115@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["bson >= 0.5.10", "opencv-python", "numpy", "matplotlib", "tqdm", "scikit-image"],

)