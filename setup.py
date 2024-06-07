from setuptools import setup, find_packages

setup(
    name="gridemissions",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.4",
    python_requires=">=3.7",
    install_requires=["requests", "pandas>=2.0", "matplotlib>=3.4.2"],
    extras_require={
        "all": [
            "dask[complete]",
            "joblib",
            "cmocean",
            "cvxpy>=1.4.3",
            "scipy>=1.10.1",
            "seaborn",
            "syrupy",
            "pytest",
        ]
    },
    entry_points={
        "console_scripts": [
            "ge_download=gridemissions.scripts.download:main",
            "ge_update=gridemissions.scripts.update:main",
            "ge_report=gridemissions.scripts.report:main",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
