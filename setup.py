import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vest",  # Replace with your own username
    version="0.0.2",
    author="Vitor Cerqueira",
    author_email="cerqueira.vitormanuel@gmail.com",
    description="VEST: Vector of Statistics from Time Series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vcerqueira/vest-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)