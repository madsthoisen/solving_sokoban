import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="solving_sokoban",
    version="0.0.1",
    author="Mads Thøisen",
    author_email="madsthoisen@users.noreply.github.com",
    description="Wrestling with Sokoban",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/madsthoisen/solving_sokoban",
    project_urls={
        "Bug Tracker": "https://github.com/madsthoisen/solving_sokoban/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
