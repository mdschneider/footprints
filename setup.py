from setuptools import setup

setup(name='footprints',
      version='0.1',
      description='Convert astronomical images to multi-epoch footprint files for sources of interest',
      url='https://github.com/mdschneider/footprints',
      author='Michael D. Schneider, William A. Dawson',
      author_email='mdschneider@me.com',
      license='MIT',
      packages=['footprints'],
      entry_points = {
      	'console_scripts': ['convert_to_footprint=footprints.convert_to_footprint:main',
      						'show_footprint=footprints.show_footprint:main']
      },
      zip_safe=False)
3