from distutils.core import setup

setup(
    name='pybench',
    version='0.0.0dev0',
    description='Automation Tools for Python Benchmarking',
    author='Peter Andreas Entschev',
    author_email='peter@entschev.com',
    url='https://github.com/pentschev/pybench',
    packages=['pybench'],
    install_requires=['pytest', 'pytest-benchmark'],
)
