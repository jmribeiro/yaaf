from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='yaaf',
    version='1.0.1',
    description='YAAF: Yet Another Agents Framework',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/jmribeiro/yaaf',
    author='João Ribeiro',
    author_email='joao.g.ribeiro@tecnico.ulisboa.pt',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "gym",
        "torch",
        "torchvision",
        "tensorflow",
        "sklearn",
        "pyyaml",
        "pyparsing",
        "tqdm"
    ],
    zip_safe=False,
    keywords=[
        'Autonomous Agents',
        'Reinforcement Learning',
        "Deep Learning"
    ],
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
