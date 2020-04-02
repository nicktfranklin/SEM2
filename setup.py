from setuptools import setup

setup(name='sem',
      version='2.0',
      description='The SEM model',
      url='https://github.com/nicktfranklin/sem',
      author='Nicholas Franklin',
      author_email='nthompsonfranklin@gmail.com',
      license='MIT',
      packages=['sem'],
      install_requires=[
            'numpy',
            'scipy',
            'tqdm',
            'tensorflow',
            'sklearn',
      ],
      zip_safe=False
      )