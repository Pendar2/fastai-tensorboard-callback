import setuptools

exec(open('fastai_tensorboard_callback/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastai-tensorboard-callback",
    version = __version__,
    author="Bryan Heffernan",
    license = "Apache License 2.0",
    description="A Tensorboard logging callback for fastai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords = 'fastai, deep learning, machine learning, tensorboard, tensorflow',
    url="https://github.com/Pendar2/fastai-tensorboard-callback",
    packages=setuptools.find_packages(),
    install_requires=['fastai', 'tensorboard', 'tensorboardX'],
    python_requires  = '==3.6.*',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
    ],
)
