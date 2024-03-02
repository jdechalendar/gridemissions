from setuptools import setup, find_packages

setup(
    name="gridemissions",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.3",
    python_requires=">=3.7",
    install_requires=["requests", "pandas>=2.0", "matplotlib>=3.4.2"],
    extras_require={
        "all": [
            "dask[complete]",
            "joblib",
            "cmocean",
            "cvxpy",
            "seaborn",
            "syrupy",
            "pytest",
        ]
    },
    entry_points={
        "console_scripts": [
            "gridemissions_download=gridemissions.scripts.download:main",
            "update_gridemissionsdata=gridemissions.scripts.update:main",
            "ba_report=gridemissions.scripts.ba_report:main",
            "ampd=gridemissions.scripts.ampd:main",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
