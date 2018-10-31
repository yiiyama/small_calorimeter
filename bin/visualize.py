#!/usr/bin/env python

import sys
import array
from argparse import ArgumentParser

parser = ArgumentParser(description='Visualize')
parser.add_argument('input', nargs = '+', help="Path to input tree")
parser.add_argument('--filter', '-f', dest = 'filter', help="Selection")

args = parser.parse_args()

sys.argv = []

import ROOT

tree = ROOT.TChain('B4')
for path in args.input:
    tree.Add(path)

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

eventvar_branches = [
    'true_energy',
    'true_x',
    'true_y'
]
num_eventvars = len(eventvar_branches)

event = {}

for name in feature_branches:
    vec = ROOT.vector('double')()
    tree.SetBranchAddress(name, vec)
    event[name] = vec

for name in label_branches:
    lab = array.array('i', [0])
    tree.SetBranchAddress(name, lab)
    event[name] = lab

for name in eventvar_branches:
    var = array.array('d', [0])
    tree.SetBranchAddress(name, var)
    event[name] = var

grid = ROOT.TH3D('grid', '', 125, 0., 1875., 32, -150., 150., 32, -150., 150.) # using graphical X axis for Z to make visualization easier
canvas = ROOT.TCanvas('c1', 'c1', 800, 800)

if args.filter:
    tree.Draw('>>elist', args.filter, 'entrylist')
    elist = ROOT.gDirectory.Get('elist')
    tree.SetEntryList(elist)

ievent = 0
while True:
    ientry = tree.GetEntryNumber(ievent)
    tree.GetEntry(ientry)
    ievent += 1

    grid.Reset()

    for ihit in range(event['rechit_energy'].size()):
        grid.Fill(event['rechit_z'][ihit], event['rechit_x'][ihit], event['rechit_y'][ihit], event['rechit_energy'][ihit])

    for name in label_branches:
        if event[name][0] != 0:
            grid.SetTitle('%s E=%.2f GeV, (x,y)=(%.2f,%.2f)mm' % (name, event['true_energy'][0], event['true_x'][0], event['true_y'][0]))
        
    grid.Draw('BOX')
    canvas.Update()

    line = sys.stdin.readline()
    if 'q' in line:
        break
