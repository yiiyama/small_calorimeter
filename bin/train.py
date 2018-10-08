#!/usr/bin/env python

import setGPU
from argparse import ArgumentParser

from trainer import Trainer

parser = ArgumentParser(description='Run training for recurrent cal')
parser.add_argument('input', help="Path to config file")
parser.add_argument('config', help="Config section within the config file")
parser.add_argument('--debug', '-D', action = 'store_true', dest = 'debug', help='Run the debug printouts')
parser.add_argument('--evaluate', '-E', action = 'store_true', dest = 'evaluate', help='Run the evaluation from saved training results')
parser.add_argument('--continue', '-C', action = 'store_true', dest = 'use_saved', help='Continue training from saved data')

args = parser.parse_args()

trainer = Trainer(args.input, args.config)

if args.debug:
    trainer.debug(trained = args.use_saved)
elif args.evaluate:
    trainer.evaluate()
else:
    trainer.train(from_scratch = (not args.use_saved))
