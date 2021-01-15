import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyFrame-wrangelvid",
    version="0.0.1",
    author="David von Wrangel",
    author_email="wrangelvid@gmail.com",
    description="3D Frame and Truss analysis using the direct stiffness method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wrangelvid/pyFrame",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)