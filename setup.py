from setuptools import setup

setup(
  name='SplineSmooth3D',
  version='0.1.0',
  author='Ian Malone',
  author_email='i.malone@ucl.ac.uk',
  packages=['SplineSmooth3D'],
  scripts=['bin/measureprofile.py', 'bin/splinesmooth3d-promote-demo.py',
           'bin/splinesmooth3d-demo.py',
           'bin/splinesmooth3d-zerounsupported-demo.py',
           'bin/splinesmooth3d-j0-demo.py', 'bin/test-scipyspline-demo.py'],
  url='http://https://gitlab.drc.ion.ucl.ac.uk/malone/spline-experimentation/',
  license='BSD',
  description='3D B-spline smoother supporting thin-plate and multi-level basis splines',
  long_description=open('README.txt').read(),
  install_requires=[
    "numpy >= 1.17.2",
    "scipy >= 1.3.1",
  ],
)
