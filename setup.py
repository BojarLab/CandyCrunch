import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CandyCrunch",
    version="0.4.0",
    author="Daniel Bojar",
    author_email="daniel.bojar@gu.se",
    description="Package for predicting glycan structure from LC-MS/MS data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BojarLab/CandyCrunch",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['*.csv', '*.pkl', '*.jpg', '*.pt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=["glycowork~=1.2.0", "regex", "networkx",
                      "scipy", "torch~=2.1", "numpy_indexed",
                      "seaborn", "pandas", "statsmodels",
                      "pymzml", "statsmodels", "pyteomics",
                      "lxml", "torchvision", "openpyxl"],
    extras_require={'draw':["glycowork[draw]~=1.2.0"]},
    entry_points={
        'console_scripts': [
            'candycrunch_predict=CandyCrunch.cli:main',
        ],
    },
)
