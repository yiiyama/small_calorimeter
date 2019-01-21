import sys
import array
import numpy as np
import ROOT

canvas = ROOT.TCanvas('c1', 'c1', 600, 600)
canvas.SetGrid(True, True)

for model in ['sparse_conv_single_neighbors', 'sparse_conv_hidden_aggregators']:
    for xname, xtitle in [('energy', 'E (GeV)'), ('sigma', '#sigma_{r} (mm)'), ('z', 'center of mass z (mm)')]:
        d = np.load('/afs/cern.ch/work/y/yiiyama/small_calorimeter/gpi/%s_%sdep.npy' % (model, xname))
        
        numer = ROOT.TH1D('numer', '', len(d) - 1, array.array('d', [x[0] for x in d]))
        denom = ROOT.TH1D('denom', '', len(d) - 1, array.array('d', [x[0] for x in d]))
        
        for e, n, d in d:
            for _ in range(int(n)):
                numer.Fill(e)
            for _ in range(int(d)):
                denom.Fill(e)
        
        gr = ROOT.TGraphAsymmErrors(numer, denom)
        gr.SetLineColor(ROOT.kBlack)
        gr.SetLineWidth(2)
        gr.SetMarkerColor(ROOT.kBlack)
        gr.SetMarkerStyle(8)
        gr.Draw('AP')

        gr.GetYaxis().SetRangeUser(0.5, 1.)
        gr.GetXaxis().SetTitle(xtitle)

        canvas.Print('/afs/cern.ch/user/y/yiiyama/www/plots/small_calorimeter/20190115/%s_%s.png' % (model, xname))
        canvas.Print('/afs/cern.ch/user/y/yiiyama/www/plots/small_calorimeter/20190115/%s_%s.pdf' % (model, xname))

        numer.Delete()
        denom.Delete()
