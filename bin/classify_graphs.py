import sys
import array
import collections

import ROOT

source = ROOT.TFile.Open(sys.argv[1])
tree = source.Get('B4')

ax = array.array('d', [0.])
ay = array.array('d', [0.])
al = array.array('d', [0.])
ai = array.array('i', [0])

tree.SetBranchAddress('rechit_x', ax)
tree.SetBranchAddress('rechit_y', ay)
tree.SetBranchAddress('rechit_layer', al)
tree.SetBranchAddress('rechit_detid', ai)

tree.GetEntry(0)

classes = collections.defaultdict(list)

for ihit in range(len(ai)):
    
