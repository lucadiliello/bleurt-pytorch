import setuptools


def load_long_description():
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description


def load_requirements():
    requirements = []
    with open("requirements.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                requirements.append(line)
    return requirements


setuptools.setup(
    name='bleurt-pytorch',
    version='0.0.1',
    description='PyTorch porting of BLEURT',
    long_description=load_long_description(),
    long_description_content_type="text/markdown",
    url='https://github.com/lucadiliello/bleurt_pytorch.git',
    author='Luca Di Liello',
    author_email='luca.diliello@unitn.it',
    license='Apache License Version 2.0',
    packages=setuptools.find_packages(),
    install_requires=load_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License Version 2.0",
        "Operating System :: OS Independent",
    ]
)
