import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="keravis",
    version="1.0.1",
    author="Yara Halawi",
    author_email="yarahalawi12@gmail.com",
    description="A high-level API for ConvNet visualizations in Keras",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yarahal/keravis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
