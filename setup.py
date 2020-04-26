import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sskmeans",
    version="0.1.0",
    author="Eliot-M",
    author_email="unicorndancingoncodes@gmail.com",
    description="Simple package to create a same size clustering (inspired from a regular Kmeans with size constraints)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Eliot-M/sskmeans",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)