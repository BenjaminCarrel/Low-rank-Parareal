import setuptools

with open("README.md", "r") as descr:
    long_description = descr.read()

setuptools.setup(
    name="Low-rank Parareal",
    version="1.0",
    author="Benjamin Carrel",
    author_email="benjamin.carrel@unige.ch",
    url="https://gitlab.unige.ch/Benjamin.Carrel/low-rank-parareal",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm"
    ],
)
