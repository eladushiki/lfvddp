from setuptools import setup, find_packages

setup(
    name='SymmetrizedDDP',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your project's dependencies here
        # e.g., 'numpy', 'pandas', 'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'train=train.single_train:main',
            'submit_train=train.submit_train:main',
            'is_same_as_commit=frame.cluster.is_same_as_commit:main',
        ],
    },
    author='Elad Kliger',
    author_email='elad.kliger@weizmann.ac.il',
    description='Symmetrized DDP research project infrastructure and tools',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/inbarsavoray/LFVNN-symmetrized',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)