import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hskl",
    version="0.0.2",
    author="Qian Cao",
    author_email="qcao.dev@gmail.com",
    description="Hyperspectral image analysis with scikit-learn.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiancao/hskl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy',
                      'scikit-learn',
                      'multimethod',
                      'matplotlib',
                      'spectral',
                      'h5py',
                      'scipy',
                      'tqdm',
                      'scikit-image']
)
