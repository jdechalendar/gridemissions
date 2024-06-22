from setuptools import setup, find_packages

setup(
    name="gridemissions",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.7",
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
            # As of June 22, 2024
            # numpy 2.0 breaks gridemissions because osqp is not yet compatible with it
            # see https://github.com/cvxpy/cvxpy/issues/2474
            # Can probably allow higher versions of numpy once that issue is closed
            "numpy==1.26.4",
        ]
    },
    entry_points={
        "console_scripts": [
            "ge_download=gridemissions.scripts.download:main",
            "ge_update_live_dataset=gridemissions.scripts.update_live_dataset:main",
            "ge_report=gridemissions.scripts.report:main",
        ],
    },
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
