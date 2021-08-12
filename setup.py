from setuptools import setup, find_packages

setup(
    name="gridemissions",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.1.0",
    python_requires=">=3.7",
    install_requires=[
        "requests",
        "pandas>=1.1.2"
    ],
    extras_require={
        "all":[
            'dask[complete]',
            'matplotlib>=3.4.2',
            'joblib',
            'cmocean',
            'cvxpy',
            'seaborn'
            ]
        },
    entry_points={
        "console_scripts": [
            "download_emissions=gridemissions.scripts.download:main",
            "update_gridemissionsdata=gridemissions.scripts.update:main",
            "ba_report=gridemissions.scripts.ba_report:main",
            "ampd=gridemissions.scripts.ampd:main",
        ],
    },
)
