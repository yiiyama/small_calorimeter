import sys
import array
import numpy as np
import h5py
import ROOT

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
canvas.SetGrid(True, True)
canvas.SetLeftMargin(0.15)
canvas.SetRightMargin(0.05)

source_dir = '/afs/cern.ch/work/y/yiiyama/small_calorimeter/gpi'

models = [
    ('featured_3d_conv_roc.npy', 'featured_3d_conv_ntuples.py', 'Binning', '1f77b4'),
    ('sparse_conv_single_neighbors_roc.npy', 'sparse_conv_single_neighbors_ntuples.py', 'Single neighbors', 'ff7f0e'),
    ('sparse_conv_hidden_aggregators_roc.npy', 'sparse_conv_hidden_aggregators_ntuples.py', 'Hidden aggregators', '2ca02c')
]

drawopt = 'AP'
grs = []
legend = ROOT.TLegend(0.17, 0.12, 0.55, 0.3)
legend.SetBorderSize(1)
legend.SetLineColor(ROOT.kGray)
legend.SetFillStyle(1001)
legend.SetFillColor(ROOT.kWhite)
legend.SetTextSize(0.033)

for roc, ntuples, title, color in models:
    with open(source_dir + '/' + roc, 'rb') as source:
        roc_data = np.load(source)

    idx = np.searchsorted(np.flip(roc_data[0][1], axis=-1), [0.1])[0]
    threshold = 0.005 * idx

    binning = [x + 0. for x in range(0, 100, 5)]

    numer = ROOT.TH1D('numer', '', len(binning) - 1, array.array('d', binning))
    denom = ROOT.TH1D('denom', '', len(binning) - 1, array.array('d', binning))

    with h5py.File(source_dir + '/' + ntuples) as source:
        for row in source['ntuples']:
            ispi0, prob, energy = row[:3]
            if (prob > threshold) == (ispi0 == 1.):
                numer.Fill(energy)
            denom.Fill(energy)

    col = ROOT.TColor.GetColor('#' + color)

    gr = ROOT.TGraphAsymmErrors(numer, denom)
    gr.SetLineColor(col)
    gr.SetLineWidth(2)
    gr.SetMarkerColor(col)
    gr.SetMarkerStyle(8)
    gr.Draw(drawopt)

    drawopt = 'P'

    gr.GetYaxis().SetTitle('True positive rate')
    gr.GetYaxis().SetRangeUser(0.5, 1.)
    gr.GetXaxis().SetTitle('Total energy (GeV)')

    grs.append(gr)

    legend.AddEntry(gr, title, 'LP')

    numer.Delete()
    denom.Delete()

legend.Draw()

canvas.Print('/afs/cern.ch/user/y/yiiyama/www/plots/small_calorimeter/20190121/energydep.png')
canvas.Print('/afs/cern.ch/user/y/yiiyama/www/plots/small_calorimeter/20190121/energydep.pdf')
