from setuptools import setup

setup(name='deep_morphology',
      version='0.1',
      description='PyTorch models for my experiments in morphology',
      url='https://github.com/juditacs/deep-morphology',
      author='Judit Acs',
      author_email='judit@sch.bme.hu',
      license='MIT',
      packages=['deep_morphology'],
      install_requires=[
          'recordclass',
          'pyyaml',
          'numpy',
          'torch',
          'torchvision',
      ],
      zip_safe=False)
