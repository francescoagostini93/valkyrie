from setuptools import setup, find_packages

setup(
    name='data-verification-tool',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        'numpy',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'data_verification_tool=main:main',
        ],
    },
    description='A tool for verifying and managing image datasets.',
    author='Andrea Chinello',
    author_email='andrea.chinello@adaptica.com',
    url='',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
