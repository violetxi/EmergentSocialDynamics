# EmergentSocialDynamics
Context dependent intrinsic motivation allows diverse social dynamics to emerge from different environment.

## Installation

Install this repo on a remote cluster without full previlige, first install SWIG from source:

Download SWIG 
```
wget http://prdownloads.sourceforge.net/swig/swig-4.1.1.tar.gz
tar -xvf swig-4.1.1.tar.gz && cd swig-4.1.1/
```
Configure the makefile for SWIG. This is where you specify the prefix to a directory that you have write access to:
```
./configure --prefix=/path/to/your/directory/  --without-pcre
```
Build and install SWIG 
```
make && make install
```

Then add the `bin` directory to your ~/.bashrc
```
echo 'export PATH=/path/to/your/home/directory/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```
Install repo and dependency (using *venv* or *conda env* with `python=3.8`)
```
pip install -e .
```

