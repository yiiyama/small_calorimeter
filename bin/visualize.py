#!/usr/bin/env python

import sys
from argparse import ArgumentParser

parser = ArgumentParser(description='Visualize')
parser.add_argument('input', help="Path to input tree")
parser.add_argument('selection', nargs = '?', help="Selection")

args = parser.parse_args()

sys.argv = []

import ROOT
import root_numpy as rnp

source = ROOT.TFile.Open(args.input)
tree = source.Get('B4')

feature_branches = [
    'rechit_energy',
    'rechit_x',
    'rechit_y',
    'rechit_z',
    'rechit_layer',
    'rechit_varea',
    'rechit_vxy',
    'rechit_vz'
]
num_features = len(feature_branches)

#label_branches = [
#    'isElectron',
#    'isPionCharged',
#    'isMuon',
#    'isPionNeutral',
#    'isK0Long',
#    'isK0Short'
#]
label_branches = [
    'isGamma',
    'isPionNeutral'
]
num_labels = len(label_branches)

full_array = rnp.tree2array(tree, feature_branches + label_branches, args.selection)

grid = ROOT.TH3D('grid', '', 125, 0., 1875., 32, -150., 150., 32, -150., 150.) # using graphical X axis for Z to make visualization easier
canvas = ROOT.TCanvas('c1', 'c1', 800, 800)

ievent = 0
for event in full_array:
    ievent += 1

    grid.Reset()

    for ihit in range(len(event[0])):
        grid.Fill(event[3][ihit], event[1][ihit], event[2][ihit], event[0][ihit])

    for ib in range(num_labels):
        if event[num_features + ib]:
            grid.SetTitle(label_branches[ib])
        
    grid.Draw('BOX')
    canvas.Update()

    line = sys.stdin.readline()
    if 'q' in line:
        break
