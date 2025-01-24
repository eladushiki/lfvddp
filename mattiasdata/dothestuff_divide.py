import os
import sys
from ROOT import *
gSystem.Load("/srv01/agrp/mattiasb/runners/PhenoOutput/scripts/kinvars_root620/kinVars_h.so")
def main():
    input_path="/srv01/agrp/mattiasb/runners/PhenoOutput/pheno-yesbrem20/merge_outputs/"
    input_files=get_input_files(input_path,skipZ=True) ## Skipping Drell-Yann else takes very long
    channels=['em','me','ee','mm']
    variables=['Lep0Pt','Lep1Pt','MLL']
    histdct=create_histograms(variables,channels)
    fill_histograms(input_files,histdct,channels,variables)
    for var in variables:
      plot_histograms(histdct,var,'em','me',save=True) ## Set 'save' to True to create plot as png file
      plot_histograms(histdct,var,'ee','mm',save=True)

## Make plots
def plot_histograms(histdct,var,ch0,ch1,save=True):
    ## Get hists
    h0=histdct["%s_%s"%(ch0,var)]
    h1=histdct["%s_%s"%(ch1,var)]
    ratio=create_ratio(h0, h1, ch0, ch1)
    ## Plot options
    title="%s_vs_%s_%s"%(ch0,ch1,var)
    h0.SetTitle(title)
    h0.SetMaximum(1.3*max(h0.GetMaximum(),h1.GetMaximum()))
    h0.SetLineColor(kRed)
    h1.SetLineColor(kBlue)
    h0.SetMarkerStyle(20)
    h0.SetLineWidth(1)
    c=TCanvas("c","")
    upper_pad=TPad("upper_pad", "", 0, 0.35, 1, 1)
    c.cd()
    lower_pad=TPad("lower_pad", "", 0, 0, 1, 0.35)
    upper_pad.Draw()
    lower_pad.Draw()
    upper_pad.cd()
    h0.Draw()
    h1.Draw("same")
    lower_pad.cd()
    ratio.Draw("ep")
    ratiobkg = TF1("one", "1", 0, 150e3)
    ratiobkg.SetLineColor(kRed)
    ratiobkg.SetLineStyle(2)
    ratiobkg.SetLineWidth(2)
    ratiobkg.SetMinimum(-125)
    ratiobkg.SetMaximum(250)
    ratiobkg.GetXaxis().SetLabelSize(0.08)
    ratiobkg.GetXaxis().SetTitleSize(0.12)
    ratiobkg.GetXaxis().SetTitleOffset(1.0)
    ratiobkg.GetYaxis().SetLabelSize(0.08)
    ratiobkg.GetYaxis().SetTitleSize(0.09)
    ratiobkg.GetYaxis().SetTitle("Data - Bkg.")
    ratiobkg.GetYaxis().CenterTitle()
    ratiobkg.GetYaxis().SetTitleOffset(0.7)
    ratiobkg.GetYaxis().SetNdivisions(503, False)
    ratiobkg.GetYaxis().ChangeLabel(-1, -1, 0)
    ratiobkg.Draw("same")
    if save:
        c.SaveAs(title+".png")
        l=TList()
        l.Add(h0)        
        l.Add(h1)
        l.Add(ratio)
        f=TFile(title+"_histograms.root","RECREATE")
        l.Write(title+"_histograms",TObject.kSingleKey)
        f.ls()        
    else:
        print("Press Enter to continue...")
        raw_input()

## Create ratio
def create_ratio(h1, h2, ch0, ch1):
   h3 = h1.Clone("h3")
   h3.SetLineColor(kBlack)
   h3.SetMarkerStyle(20)
   h3.SetTitle("")
   h3.SetMinimum(0.6)
   h3.SetMaximum(1.4)
   # Set up plot for markers and errors
   h3.Sumw2()
   h3.SetStats(0)
   h3.Divide(h2)
   
   # Adjust y-axis settings
   y = h3.GetYaxis()
   y.SetTitle("ratio %s/%s "%(ch0,ch1))
   y.SetNdivisions(505)
   y.SetTitleSize(15)
   y.SetTitleFont(43)
   y.SetTitleOffset(1.5)
   y.SetLabelFont(43)
   y.SetLabelSize(15)
   
   # Adjust x-axis settings
   x = h3.GetXaxis()
   x.SetTitle("pT")
   x.SetTitleSize(15)
   x.SetTitleFont(43)
   x.SetTitleOffset(1.5)
   x.SetLabelFont(43)
   x.SetLabelSize(15)
   
   return h3




## Combine all files to single tree

# tree=TChain("PhenoOutput")
#     for i,f in enumerate(input_files):
#         tree.Add(f)
#     n_events=tree.GetEntries()
#     print("Filling histograms, going to process %s events"%n_events)
   


## Fill histogram
def fill_histogram(input_file,histdct,channels,variables):
    
    tree=TChain("PhenoOutput")
    tree.Add(input_file)
    ## Loop on events
    for i,event in enumerate(tree):
        if i>0 and i%100000==0:
            print("- Processed event %s/%s"%(i,n_events))
        ## Get event channel (ee/mm/em/me)
        channel=event.reco.channel
        ## Apply selection (example)
        if not event.reco.has2leps:
            continue
        if channel not in channels:
            continue
        if event.reco.Lep0Pt<15e3 or event.reco.Lep1Pt<15e3: ## check both since pT ordering is from true events, sometimes inverted in reco evet
            continue
        if abs(event.reco.Lep0Eta)>2.3 or abs(event.reco.Lep1Eta)>2.3:
            continue
        ## Apply isolation constraint
        iso0=0.15 if channel[0]=="e" else 0.25
        iso1=0.15 if channel[1]=="e" else 0.25
        if event.reco.Lep0Iso>iso0 or event.reco.Lep1Iso>iso1:
            continue
        ## Fill histograms
        for var in variables:
            histdct["%s_%s"%(channel,var)].Fill(getattr(event.reco,var))

## Create empty histograms
def create_histograms(variables,channels):
    dct={}
    for ch in channels:
        for var in variables:
            label="%s_%s"%(ch,var)
            dct[label]=TH1F(label,label,30,0,150e3)
            dct[label].SetDirectory(0) ## Needed so python takes ownership of histogram else disappears
    return dct

## Get list of input files
def get_input_files(input_path,skipZ=False):
    files=[]
    for f in os.listdir(input_path):
        if skipZ and f.startswith('Z_'):
            continue
        if f.endswith('_dump.root'):
            files.append(input_path+f)
    return files

if __name__=="__main__":
    main()
