from setuptools import setup, find_packages


setup(
    name='bpe_nmt',
    version='0.1.0',
    description='Tensor2tensor compatible BPE encoder',
    author='Haukur Barri Simonarson',
    license='MIT',
    classifiers=[
    'Intended Audience :: Developers',
    'Topic :: Text Processing',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    include_package_data=True
)
