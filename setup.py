from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='social_rl',
    version='1.0',
    packages=find_packages(),
    description='Your package description',
    author='Your Name',
    author_email='your@email.com',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=requirements
)
