from distutils.core import setup

setup(name='pyvisir',
      version='0.1.0',
      description='VISIR reduction library',
      author='Klaus Pontoppidan',
      author_email='pontoppi@stsci.edu',
      url='http://www.stsci.edu/~pontoppi',
      packages=['pyvisir','utils'],
      package_data={'pyvisir': ['*.ini']}
      )

    
