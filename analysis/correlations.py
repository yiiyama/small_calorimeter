import numpy as np
import h5py
import ROOT

outfile = ROOT.TFile.Open('/tmp/yiiyama/correlations.root', 'recreate')

gamma_depth = ROOT.TH1D('gamma_depth', ';<z>/E_{tot} (mm/GeV)', 100, 0., 5.)
pi0_depth = ROOT.TH1D('pi0_depth', ';<z>/E_{tot} (mm/GeV)', 100, 0., 5.)
prob_v_depth = ROOT.TH2D('prob_v_depth', ';<z>/E_{tot} (mm/GeV);P(#pi^{0})', 100, 0., 5., 100, 0., 1.)
gamma_prob_v_depth = ROOT.TH2D('gamma_prob_v_depth', ';<z>/E_{tot} (mm/GeV);P(#pi^{0})', 100, 0., 5., 100, 0., 1.)
pi0_prob_v_depth = ROOT.TH2D('pi0_prob_v_depth', ';<z>/E_{tot} (mm/GeV);P(#pi^{0})', 100, 0., 5., 100, 0., 1.)
prob_v_width = ROOT.TH2D('prob_v_width', ';#sigma_r (mm);P(#pi^{0})', 100, 0., 60., 100, 0., 1.)
gamma_prob_v_width = ROOT.TH2D('gamma_prob_v_width', ';#sigma_r (mm);P(#pi^{0})', 100, 0., 60., 100, 0., 1.)
pi0_prob_v_width = ROOT.TH2D('pi0_prob_v_width', ';#sigma_r (mm);P(#pi^{0})', 100, 0., 60., 100, 0., 1.)

for conf in ['sparse_conv_single_neighbors']:
    with h5py.File('/afs/cern.ch/work/y/yiiyama/small_calorimeter/gpi/%s_ntuples.py' % conf) as source:
        ntuples = source['ntuples']
        for ispi0, prob, energy, sigma_r, z in ntuples:
            prob_v_depth.Fill(z / energy, prob)
            prob_v_width.Fill(sigma_r, prob)
            if ispi0 > 0.5:
                pi0_depth.Fill(z / energy)
                pi0_prob_v_depth.Fill(z / energy, prob)
                pi0_prob_v_width.Fill(sigma_r, prob)
            else:
                gamma_depth.Fill(z / energy)
                gamma_prob_v_depth.Fill(z / energy, prob)
                gamma_prob_v_width.Fill(sigma_r, prob)

outfile.cd()
outfile.Write()
