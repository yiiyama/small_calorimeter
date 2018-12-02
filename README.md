Requirements
============

- Python 3.6
- Numpy
- Tensorflow >= 1.10

Installation note
=================

If setGPU.py is not importable, copy it from /opt/anaconda3/lib/python3.6/site-packages/setGPU.py to your environment's site-packages.

Running
=======

source setup.sh # one time only just to set PYTHONPATH
./bin/train.py [--debug|--evaluate|--continue] <config file> <config name>

<config file>s are in configs/ directory.

Structure
=========

- All python modules are under lib
- train.py uses the Trainer object in lib/trainer.py. Trainer imports the model specified in the configuration file.
- All models are in lib/model/. Models inherit from the base class lib/model/model.py. Currently all models are classification tasks and inherit from lib/model/classification.py, which itself is a subclass of model.py.
