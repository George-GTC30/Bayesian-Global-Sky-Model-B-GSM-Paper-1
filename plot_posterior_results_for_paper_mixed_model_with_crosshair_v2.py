from random import sample
import matplotlib as mpl
#mpl.use('Agg')
from itertools import combinations_with_replacement
from itertools import product
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LogNorm
import os
import pandas
from numpy import pi, log, sqrt
#import pix_by_pix_mk19b as likelihood

import scipy.optimize as so
import math
import scipy

import matplotlib.cm as cm
from scipy.optimize import minimize

try:
    import pypolychord
    from pypolychord.settings import PolyChordSettings
    from pypolychord.priors import UniformPrior
    from pypolychord.priors import GaussianPrior
except:
    pass
try:
    from anesthetic import NestedSamples
except ImportError:
    pass
try:
    from anesthetic.weighted_pandas import WeightedDataFrame
except ImportError:
    pass
from scipy.optimize import minimize
import gen_EDGES_beams as gen_beams_and_T_vs_LST_v2
from line_profiler import LineProfiler
import matplotlib
from fgivenx import plot_contours, plot_lines
from fgivenx import samples_from_getdist_chains
from astropy.modeling.powerlaws import LogParabola1D
unobs_marker = -32768
p=os.getcwd()+"/"
Nside = 32

no_of_comps = 2
#=======================================================================
fit_curved=[True,True]
calibrate_all = True#False
calibrate_all_but_45_150 = False#True

rezero_prior_std=2000
subtract_CMB = 0
#params for the prior on the spectra
spec_min, spec_max = -3.5,1 #the range for the prior on the spectral indexes
curvature_mean, curvature_std = 0,3 #the range for the prior on the spectral index curvature parameter

#params for fitting the reference frequency 

fixed_f0 = 150 #if you dont fit a seperate reference freq for each comp then we fix f0 to this value


#params for the prior on the true maps
map_prior_variance_spec_index = -2.6
map_prior_variance_f0 = 408#fixed_f0
map_prior_std=300

no_to_fit_curve = np.sum(fit_curved)

#=======================================================================
#true spectral parameters
true_specs=[[-2.6,0],[-2.1,-0.5]]
#true components
true_c1, true_c2 = np.loadtxt(p+"mock_data_file/true_comp1"), np.loadtxt(p+"mock_data_file/true_comp2")
true_c1 += 50-np.nanmin(true_c1)
true_c2 += 10-np.nanmin(true_c2)

#true calibration values
true_alphas = [1,0.95,1.05,1.06,1.1,1.15,1,1.18,0.95]
true_betas = [0,600,400,300,-200,-300,0,-30,5]

#reference frequency for the spectra
f0=150
#freqs for the map comparison
fmaps_for_test = np.array([47.5,75,100,140,200,250,300,350])#[45.0,50.0,60.0,70.0,74.0,80.0,150.0,159.0,408.0]#[47.5,75,100,140,200,250,300,350]

#freqs for the T_vs_LST comparison
fT_vs_LST = [55,75,95,120,140,160,180,220,300,380]
LSTs = np.linspace(0,24,49)

#freqs for the mean_temps comparisons
fmeans = [55,75,95,120,140,160,180,220,300,380]

main_label = "_petur:True_"+str(no_of_comps)+"_comp_cal:True_rezro_pri_std:"+str(rezero_prior_std)+"_CMB="+str(subtract_CMB)+"_map_pri_std:"+str(map_prior_std)+"_mu:0_map_pri_std_spec_ind="+str(map_prior_variance_spec_index)+"_map_pri_f0="+str(map_prior_variance_f0)+"_cond_no_thres=9.0_crv_N_std="+str(curvature_std)+"_spec="+str(spec_min)+"_to:"+str(spec_max)#+"_rej_crit="+str(reject_criterion)#+"_nlive="+str(nlive)+"_nrept="+str(nrepeat)+"_precision_criterion="+str(precision_criterion)

if calibrate_all==True:
        file_name="uni_EDGES_v4_data_mk24_no_of_curved:"+str(no_to_fit_curve)+"_cal_all_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label
else:
    if calibrate_all_but_45_150==True:
        file_name="uni_EDGES_v4_dat_mk24_no_curve:"+str(no_to_fit_curve)+"_no_cal_45_150_f0="+str(fixed_f0)+main_label
#file_name = "v3b_data_mk24_LSTs:72_f0=150_perturbed_dataset:True_2_comp_cal_all_inc_408:True_rezro_pri_std:2000_CMB=0_map_pri_std:500_mu:0_map_pri_spec_ind=-2.6_map_pri_f0=408_cond_no_thres=9.0_crv_N_std=2_spec=-3.5_to:1"

root = p+file_name
print ("================================================================================")
print (file_name)
print ("================================================================================")

#plot the true compontnets and the spectra used to generate the synthetic dataset
#================================================================================
plt_freqs = np.logspace(1.653,2.7,200)

fig=plt.figure(figsize=(7,4.5))
ax1 = plt.subplot2grid((2,2),(0,0))
plt.axes(ax1)
hp.mollview(true_c1,title="Component 1 true",hold=True,notext=True,norm="log")
ax2 = plt.subplot2grid((2,2),(1,0))
plt.axes(ax2)
hp.mollview(true_c2,title="Component 2 true",hold=True,notext=True,norm="log")

ax3 = plt.subplot2grid((2,2),(0,1))
ax3.set_title("Spectrum Component 1",fontsize=12)
ax3.plot(plt_freqs,LogParabola1D(1,150,2.6,0)(plt_freqs))
ax3.set_ylabel("scaling",fontsize=12)
ax3.set_yticks([0.01,0.1,1,10,100],[0.01,0.1,1,10,100])
ax3.set_ylim(0.01,100)
ax3.set_xscale('log')
ax3.set_yscale('log')

ax4 = plt.subplot2grid((2,2),(1,1))
ax4.set_title("Spectrum Component 2",fontsize=12)
ax4.plot(plt_freqs,LogParabola1D(1,150,2.1,0.5)(plt_freqs))
ax4.set_ylabel("scaling",fontsize=12)
ax4.set_xlabel("Freq (MHz)",fontsize=12)
ax4.set_yticks([0.01,0.1,1,10,100],[0.01,0.1,1,10,100])
ax4.set_ylim(0.01,100)
ax4.set_xscale('log')
ax4.set_yscale('log')
plt.rcParams.update({'font.size':12})
plt.subplots_adjust(wspace=0.5)
plt.subplots_adjust(hspace=0.5)
#plt.tight_layout()
#plt.rcParams['font.size'] = 15
#plt.show()
plt.savefig(root+"/true_comps_and_specs.pdf")

#produce a plot showing the production of the synthtic EDGES data
#=====================================================================================
true_sky45 = np.loadtxt(p+"/mock_data_file/mock_dataset_v4/true_sky_45.0")
true_TvsLST45 = np.loadtxt(p+"/mock_data_file/mock_dataset_v4/true_noisy_TvsLST_45.0")
beam45_at_18h = gen_beams_and_T_vs_LST_v2.gen_EDGES_beams_at_LSTs(45,[18],32).flatten()
beam45_at_18h[(beam45_at_18h<0)]=0
true_sky45[(true_sky45<=0)]=float("NaN")


fig=plt.figure(figsize=(6,4))
ax1 = plt.subplot2grid((1,1),(0,0))
plt.axes(ax1)
hp.mollview(true_sky45,title="Synthetic Sky 45MHz\n(no calibration uncertainty)",hold=True,notext=True,norm="log")
plt.savefig(root+"/syth_sky_45.pdf")

fig=plt.figure(figsize=(6,4))
ax2 = plt.subplot2grid((1,1),(0,0))
plt.axes(ax2)
hp.mollview(beam45_at_18h,title="Beam Model 45MHz\n(for LST=18h)",hold=True,notext=True)#,norm="log")
plt.savefig(root+"/syth_beam_45.pdf")

fig=plt.figure(figsize=(6.5,4.5))
ax3 = plt.subplot2grid((1,1),(0,0))
ax3.set_title("Synthetic EDGES data for 45MHz",fontsize=15)
ax3.plot(np.linspace(0,24,73)[:-1],true_TvsLST45)
ax3.set_ylabel("Antenna Temp (K)",fontsize=15)
ax3.set_xlabel("LST (hours)")
plt.savefig(root+"/syth_EDGES_45.pdf")
plt.show()
#========================================================
#define the nested samping labels and no of dimensions
if calibrate_all==True:
    nDims = no_of_comps + no_to_fit_curve + 18
    if no_to_fit_curve == 2:
        param_names = [r"$\beta_1$",r"$\beta_2$",r"$\gamma_1$",r"$\gamma_2$",r"$b_{45}$",r"$b_{50}$",r"$b_{60}$",r"$b_{70}$",r"$b_{74}$",r"$b_{80}$",r"$b_{150}$",r"$b_{159}$",r"$b_{408}$",r"$a_{45}$",r"$a_{50}$",r"$a_{60}$",r"$a_{70}$",r"$a_{74}$",r"$a_{80}$",r"$a_{150}$",r"$a_{159}$",r"$a_{408}$"]
        true_param_vals_for_all = [-2.6,-2.1,0,-0.5,0,600,400,300,-200,-300,0,-30,5,1,0.95,1.05,1.06,1.1,1.15,1,1.18,0.95]
    if no_to_fit_curve == 1:
        param_names = [r"$\beta_1$",r"$\beta_2$",r"$\gamma_2$",r"$b_{45}$",r"$b_{50}$",r"$b_{60}$",r"$b_{70}$",r"$b_{74}$",r"$b_{80}$",r"$b_{150}$",r"$b_{159}$",r"$b_{408}$",r"$a_{45}$",r"$a_{50}$",r"$a_{60}$",r"$a_{70}$",r"$a_{74}$",r"$a_{80}$",r"$a_{150}$",r"$a_{159}$",r"$a_{408}$"]
        true_param_vals_for_all = [-2.6,-2.1,-0.5,0,600,400,300,-200,-300,0,-30,5,1,0.95,1.05,1.06,1.1,1.15,1,1.18,0.95]
if calibrate_all_but_45_150==True:
    nDims = no_of_comps + no_to_fit_curve + 14

    if no_to_fit_curve == 2:
        param_names = [r"$\beta_1$",r"$\beta_2$",r"$\gamma_1$",r"$\gamma_2$",r"$b_{50}$",r"$b_{60}$",r"$b_{70}$",r"$b_{74}$",r"$b_{80}$",r"$b_{159}$",r"$b_{408}$",r"$a_{50}$",r"$a_{60}$",r"$a_{70}$",r"$a_{74}$",r"$a_{80}$",r"$a_{159}$",r"$a_{408}$"]
        true_param_vals_for_all = [-2.6,-2.1,0,-0.5,600,400,300,-200,-300,-30,5,0.95,1.05,1.06,1.1,1.15,1.18,0.95]
    if no_to_fit_curve == 1:
        param_names = [r"$\beta_1$",r"$\beta_2$",r"$\gamma_2$",r"$b_{50}$",r"$b_{60}$",r"$b_{70}$",r"$b_{74}$",r"$b_{80}$",r"$b_{159}$",r"$b_{408}$",r"$a_{50}$",r"$a_{60}$",r"$a_{70}$",r"$a_{74}$",r"$a_{80}$",r"$a_{159}$",r"$a_{408}$"]
        true_param_vals_for_all = [-2.6,-2.1,-0.5,600,400,300,-200,-300,-30,5,0.95,1.05,1.06,1.1,1.15,1.18,0.95]
        upper_lower_bounds=[[-2.62,-2.598],[-2.11,-2.09],[-0.51,-0.48]]
#load the posterior samples
samples = NestedSamples(root= p+"chains" + '/' + file_name)
column_names_in_dataframe = list(samples.columns.values)
print ("column names for dataframe:", column_names_in_dataframe)
nested_samples = []
for i in range(nDims):
    print (i,column_names_in_dataframe[i])
    nested_samples.append(list(samples.loc[:,column_names_in_dataframe[i]]))
    
nested_samples = np.array(nested_samples).T#samples.loc[:,par_llamo].to_numpy()

weights = samples.weight
samps_with_weights = WeightedDataFrame(nested_samples,weight=weights)
mean_params = samps_with_weights.mean().to_numpy()
std_params = samps_with_weights.std().to_numpy()
no_of_samples = nested_samples.shape[0]
print ("=========================================================")
print ("mean param solultions")
print (mean_params)
print ("stds")
print (std_params)
print ("mean param solutions (with no weighting of samples)")
print (np.mean(nested_samples,axis=0))
print ("=========================================================")


#===========================================================
#plot a histogram of the Log evidence values for the posterior
samples.gui()
plt.show()
logZ = samples.logZ() #average log evidence for the model
print (type(logZ),logZ)
logL = samples.logL.to_numpy()

plt.hist(logL,bins=np.linspace(-1000000,-800000,100),weights=weights)

#samples.logL.plot(kind="hist",weights=weights)
plt.show()
plt.hist(logL,bins=np.linspace(-883000,-881000,1000),weights=weights)
plt.xlabel("log(L)")
plt.ylabel("occurrence")
plt.show()

#plt.hist(logL,bins=np.linspace(-809000,-808900,1000))#,weights=weights)
#plt.show()
#===========================================================
#produce a corner plot for the posterior samples

#define the axis labels for the posterior plot
tick_labels = []
upper_lower_bounds = []
for i in range(len(mean_params)):
    #define the axis labels to be at the posterior mean value and +/- 5 sigma from this
    lower_lab, mean_lab ,upper_lab = mean_params[i] - 2.5*std_params[i], mean_params[i], mean_params[i] + 2.5*std_params[i]

    lower_lab, mean_lab, upper_lab = np.round(lower_lab,3), np.round(mean_lab,3), np.round(upper_lab,3)
    lower_b,upper_b = mean_params[i] - 5*std_params[i], mean_params[i] + 5*std_params[i]

    lower_b, upper_b = np.round(lower_b,3), np.round(upper_b,3)

    if std_params[i]>=1:
        lower_lab, mean_lab, upper_lab = np.round(lower_lab,0), np.round(mean_lab,0), np.round(upper_lab,0)
    
    if no_to_fit_curve==2:
        if i == 1:
            upper_lower_bounds.append([-2.105,-2.085])
            lower_lab, upper_lab = -2.1,-2.09
        if i == 2:
            upper_lower_bounds.append([-0.012,0.002])
            lower_lab, upper_lab = -0.009, -0.001
        if i!=1:
            if i!=2:
                upper_lower_bounds.append([lower_b,upper_b])
    if no_to_fit_curve==1:
        if i==2:
            upper_lower_bounds.append([-0.504,-0.487])
        else:
            upper_lower_bounds.append([lower_b,upper_b])
    tick_labels.append([lower_lab,upper_lab])
print (tick_labels)
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 

fig, axes = samples.plot_2d(['p%i' %i for i in range(1,nDims+1)])

fig.set_size_inches(8, 8)
for i in range(nDims):
    #set the axis labels for the y and x axis to the correct variable name
    axes.iloc[i,0].set_ylabel(param_names[i], fontsize=15,rotation=0,labelpad=22,horizontalalignment="center", verticalalignment="center",rotation_mode="anchor")
    axes.iloc[int(nDims-1),i].set_xlabel(param_names[i], fontsize=15,rotation=90)
    #set the tick labels for the subplots
    axes.iloc[i,0].set_yticks(tick_labels[i])
    axes.iloc[int(nDims-1),i].set_xticks(tick_labels[i])
    axes.iloc[i,0].set_yticklabels(tick_labels[i])
    axes.iloc[int(nDims-1),i].set_xticklabels(tick_labels[i],rotation=90)
    #set axis limits
    axes.iloc[int(nDims-1),i].set_xlim(upper_lower_bounds[i][0],upper_lower_bounds[i][1])
    axes.iloc[i,0].set_ylim(upper_lower_bounds[i][0],upper_lower_bounds[i][1])

#plot the true values for the params as a crossed set of lines
for i in range(nDims):
    for j in range(nDims):
        par_i_true = true_param_vals_for_all[i]
        par_j_true = true_param_vals_for_all[j]

        if j>i:
            pass
        else:
            if j==i:
                axes.iloc[i,j].plot(par_j_true*np.ones(2),[-400,700],c="red",linestyle="--")
            else:
                axes.iloc[i,j].plot([-400,700],par_i_true*np.ones(2),c="red",linestyle="--")
                axes.iloc[i,j].plot(par_j_true*np.ones(2),[-400,700],c="red",linestyle="--")
plt.tight_layout()
#plt.show()
fig.savefig(root+'/sampled_posterior_with_cross.pdf')



#============================================================================
#plot the component map posterior
#load the posterior component maps
post_c1_mean, post_c2_mean = np.loadtxt(root+"/posterior_mean_for_comp_1"), np.loadtxt(root+"/posterior_mean_for_comp_2")
post_c1_std, post_c2_std = np.loadtxt(root+"/posterior_std_for_comp_1"),np.loadtxt(root+"/posterior_std_for_comp_2")
post_c1_mean[(post_c1_mean<=0)]=float("NaN")
post_c2_mean[(post_c2_mean<=0)]=float("NaN")
fig=plt.figure(figsize=(6,8.5))

ax1 = plt.subplot2grid((4,2),(0,0))
#ax1.suptitle("Component 1")
plt.axes(ax1)
hp.mollview(true_c1,title="Component 1\ntrue",hold=True,notext=True,norm="log")#,min=-10,max=10)#np.max(c_map_1))

ax2 = plt.subplot2grid((4,2),(1,0))
plt.axes(ax2)
try:
    hp.mollview(post_c1_mean,title="posterior mean",hold=True,notext=True,norm="log")
except:
    hp.mollview(post_c1_mean,title="posterior mean",hold=True,notext=True)
ax3 = plt.subplot2grid((4,2),(2,0))
plt.axes(ax3)
hp.mollview(post_c1_std,title="posterior std",hold=True,notext=True,norm="log")

ax4 = plt.subplot2grid((4,2),(3,0))
plt.axes(ax4)
nm1=(post_c1_mean-true_c1)/post_c1_std
hp.mollview(nm1,title="norm resid",hold=True,notext=True,min=-5,max=5)

ax1b = plt.subplot2grid((4,2),(0,1))
#ax1b.suptitle("Component 2")
plt.axes(ax1b)
hp.mollview(true_c2,title="Component 2\ntrue",hold=True,notext=True,norm="log")#,min=-10,max=10)#np.max(c_map_1))

ax2b = plt.subplot2grid((4,2),(1,1))
plt.axes(ax2b)
try:
    hp.mollview(post_c2_mean,title="posterior mean",hold=True,notext=True,norm="log")
except:
    hp.mollview(post_c2_mean,title="posterior mean",hold=True,notext=True)
ax3b = plt.subplot2grid((4,2),(2,1))
plt.axes(ax3b)
hp.mollview(post_c2_std,title="posterior std",hold=True,notext=True,norm="log")

ax4b = plt.subplot2grid((4,2),(3,1))
plt.axes(ax4b)
nm2=(post_c2_mean-true_c2)/post_c2_std
hp.mollview(nm2,title="norm resid",hold=True,notext=True,min=-5,max=5)

plt.subplots_adjust(hspace=0.3)
plt.tight_layout()
plt.rcParams.update({'font.size':12})
plt.savefig(root+"/posterior_component_comparison.pdf")

nm1_for_hist=nm1[~np.isnan(nm1)]
nm2_for_hist=nm2[~np.isnan(nm2)]
print ("for component 1 norm resids: mean",np.nanmean(nm1),"std",np.nanstd(nm1))
print ("for component 2 norm resids: mean",np.nanmean(nm2),"std",np.nanstd(nm2))

fig = plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.hist(nm1_for_hist,bins=40)
plt.title("Component 1\n"+r"$\mu=$"+" "+str(np.round(np.nanmean(nm1),2))+" "+r"$\sigma=$"+" "+str(np.round(np.nanstd(nm1),2)))
plt.xlabel("norm resid")
plt.ylabel("occurrence")
plt.xlim(-4,4)
plt.subplot(1,2,2)
plt.hist(nm2_for_hist,bins=40)
plt.title("Component 2\n"+r"$\mu=$"+" "+str(np.round(np.nanmean(nm2),2))+" "+r"$\sigma=$"+" "+str(np.round(np.nanstd(nm2),2)))
plt.xlabel("norm resid")
plt.xlim(-4,4)
plt.tight_layout()
plt.savefig(root+"/posterior_hist_for_comps.pdf")
np.savetxt(root+"/comp1_norm_resids_for_hist",nm1_for_hist,delimiter=",")
np.savetxt(root+"/comp2_norm_resids_for_hist",nm2_for_hist,delimiter=",")
#plt.show()

#=========================================================================
#plot the posterior sky predictions
fig = plt.figure(figsize=(9,13.5))
test_freqs = fmaps_for_test
norm_resid_vals = []
for i in range(len(test_freqs)):
    f = test_freqs[i]

    post_sky = np.loadtxt(root+"/bayesian_pred_"+str(f)+"MHz")
    post_sky_err = np.loadtxt(root+"/bayesian_errs_"+str(f)+"MHz")

    true_sky = np.loadtxt(p+"/mock_data_file/mock_dataset_v4/true_sky_"+str(float(f)))

    true_sky[(true_sky==unobs_marker)]=float("NaN")

    norm_resids = (post_sky-true_sky)/post_sky_err
    if i == 0:

        ax1 = plt.subplot2grid((len(test_freqs),4),(i,0))
        plt.axes(ax1)
        hp.mollview(true_sky,title="true "+str(f)+" MHz",hold=True,notext=True,norm="log")

        ax2 = plt.subplot2grid((len(test_freqs),4),(i,1))
        plt.axes(ax2)
        hp.mollview(post_sky,title="post mean",hold=True,notext=True,norm="log")

        ax3 = plt.subplot2grid((len(test_freqs),4),(i,2))
        plt.axes(ax3)
        hp.mollview(post_sky_err,title="post std",hold=True,notext=True,norm="log")

        ax4 = plt.subplot2grid((len(test_freqs),4),(i,3))
        plt.axes(ax4)
        hp.mollview(norm_resids,title="norm resid",hold=True,notext=True,min=-5,max=5)
    else:

        ax1 = plt.subplot2grid((len(test_freqs),4),(i,0))
        plt.axes(ax1)
        hp.mollview(true_sky,title="true "+str(f)+" MHz",hold=True,notext=True,norm="log")

        ax2 = plt.subplot2grid((len(test_freqs),4),(i,1))
        plt.axes(ax2)
        hp.mollview(post_sky,title="post mean",hold=True,notext=True,norm="log")

        ax3 = plt.subplot2grid((len(test_freqs),4),(i,2))
        plt.axes(ax3)
        hp.mollview(post_sky_err,title="post std",hold=True,notext=True,norm="log")

        ax4 = plt.subplot2grid((len(test_freqs),4),(i,3))
        plt.axes(ax4)
        hp.mollview(norm_resids,title="norm resid",hold=True,notext=True,min=-5,max=5)

    norm_resid_vals.append(norm_resids)
plt.subplots_adjust(hspace=0.6)
plt.tight_layout()
plt.rcParams.update({'font.size':18})
plt.savefig(root+"/posterior_sky_map_comparison.pdf")
#plt.show()

norm_resid_vals=np.array(norm_resid_vals).flatten()
norm_resids_for_hist = norm_resid_vals[~np.isnan(norm_resid_vals)]

mean_norm_resid,std_norm_resid= np.nanmean(norm_resids_for_hist), np.nanstd(norm_resids_for_hist)
print ("mean norm_resid =",mean_norm_resid," std = ",std_norm_resid)

fig=plt.figure(figsize=(5.5,5))
plt.hist(norm_resids_for_hist,bins=50)
plt.ylabel("occurrence")
plt.xlabel("norm resid")
plt.title("Sky Posterior (all freqs)\n"+r"$\mu=$"+" "+str(np.round(mean_norm_resid,2))+" "+r"$\sigma=$"+" "+str(np.round(std_norm_resid,2)))
plt.savefig(root+"/posterior_sky_maps_norm_resid_histo.pdf")

np.savetxt(root+"/norm_resids_for_hist",norm_resids_for_hist,delimiter=",")
#=============================================================================
#plot the posterior of the TvsLST
test_freqs_for_TvsLST = [45.0,50.0,60.0,70.0,74.0,80.0,150.0,159.0,408.0]#[45.0,50.0,60.0,70.0,74.0,80.0]#,150.0,159.0,408.0]
test_LSTs = np.linspace(0,24,73)[:-1]



#fig=plt.figure(figsize=(12,8.5))
TvsLST_true =[]
TvsLST_uncal = []
TvsLST_post =[]
print ("========================================================")
for i in range(len(test_freqs_for_TvsLST)):
    true_TvsLST = np.loadtxt(p+"/mock_data_file/mock_dataset_v4/true_noisy_TvsLST_"+str(test_freqs_for_TvsLST[i]))
    uncal_dataset_TvsLST = np.loadtxt(p+"/mock_data_file/mock_dataset_v4/peturbed_TvsLST_"+str(test_freqs_for_TvsLST[i]))

    TvsLST_true.append(true_TvsLST)
    TvsLST_uncal.append(uncal_dataset_TvsLST)

    #plt.subplot(3,3,i+1)
    #plt.title(str(test_freqs_for_TvsLST[i])+" MHz")
    #plt.plot(test_LSTs,true_TvsLST,c="red",label="true")
    #plt.plot(test_LSTs,uncal_dataset_TvsLST,c="green",label="uncal dataset")


    #generate EDGES beams set for this freq
    beams = gen_beams_and_T_vs_LST_v2.gen_EDGES_beams_at_LSTs(test_freqs_for_TvsLST[i],test_LSTs,Nside)
    beams = (1/(4*np.pi))*beams*hp.nside2pixarea(nside=Nside)

    #load the posterior sky
    post_sky = np.loadtxt(root+"/bayesian_pred_"+str(test_freqs_for_TvsLST[i])+"MHz")
    #generate a TvsLST for the posterior sky
    post_mean_TvsLST = np.array([np.nansum(beams[:,i]*post_sky) for i in range(len(test_LSTs))])
    
    
    TvsLST_post.append(post_mean_TvsLST)
    print ("for freq:",test_freqs_for_TvsLST[i])
    print ("mean resid (for posterior):",np.mean(post_mean_TvsLST-true_TvsLST))
    print ("rms resid (uncal):",np.sqrt(np.mean((uncal_dataset_TvsLST-true_TvsLST)**2)))
    print ("rms resid (posterior):",np.sqrt(np.mean((post_mean_TvsLST-true_TvsLST)**2)))
    #plt.plot(test_LSTs,post_mean_TvsLST,c="blue",linestyle="--",label="posterior mean")

    #plt.legend(loc="upper left")
#plt.tight_layout()
#plt.savefig(root+"/posterior_TvsLST_comp_all_freqs.png")
#plt.show()
plt.rcParams.update({'font.size':14})
fig=plt.figure(figsize=(8.5,13))

#row 1
ax1 = plt.subplot2grid((9,3),(0,0),rowspan=2)
ax1.plot(test_LSTs,TvsLST_true[0],label="true",c="red",linewidth=2)
ax1.plot(test_LSTs,TvsLST_uncal[0],label="uncal data",c="green",linewidth=2)
ax1.plot(test_LSTs,TvsLST_post[0],label="post mean",linewidth=2,linestyle="--",c="blue")
ax1.legend(loc="upper left")
ax1.set_title(r"45MHz")
ax1.set_ylabel(r"Ant Temp (K)")

ax2 = plt.subplot2grid((9,3),(2,0),rowspan=1,sharex=ax1)
#ax2.plot(test_LSTs,TvsLST_true[0]-TvsLST_uncal[0],label="true - uncal data",c="green",linewidth=2)
ax2.plot(test_LSTs,-TvsLST_true[0]+TvsLST_post[0],label="true - post mean",linewidth=2,c="blue")
#ax2.legend(loc="upper left")
ax2.set_ylabel(r"$\Delta$ Ant"+"\nTemp (K)")

ax1b = plt.subplot2grid((9,3),(0,1),rowspan=2)
ax1b.plot(test_LSTs,TvsLST_true[1],label="true",c="red",linewidth=2)
ax1b.plot(test_LSTs,TvsLST_uncal[1],label="uncal data",c="green",linewidth=2)
ax1b.plot(test_LSTs,TvsLST_post[1],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1b.legend(loc="upper left")
ax1b.set_title(r"50MHz")
#ax1b.set_ylabel(r"Ant Temp (K)")

ax2b = plt.subplot2grid((9,3),(2,1),rowspan=1,sharex=ax1b)
#ax2b.plot(test_LSTs,TvsLST_true[1]-TvsLST_uncal[1],label="true - uncal data",c="green",linewidth=2)
ax2b.plot(test_LSTs,-TvsLST_true[1]+TvsLST_post[1],label="true - post mean",linewidth=2,c="blue")
#ax2b.legend(loc="upper left")
#ax2b.set_ylabel(r"$\Delta$ Ant Temp (K)")

ax1c = plt.subplot2grid((9,3),(0,2),rowspan=2)
ax1c.plot(test_LSTs,TvsLST_true[2],label="true",c="red",linewidth=2)
ax1c.plot(test_LSTs,TvsLST_uncal[2],label="uncal data",c="green",linewidth=2)
ax1c.plot(test_LSTs,TvsLST_post[2],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1c.legend(loc="upper left")
ax1c.set_title(r"60MHz")
#ax1c.set_ylabel(r"Ant Temp (K)")

ax2c = plt.subplot2grid((9,3),(2,2),rowspan=1,sharex=ax1c)
#ax2c.plot(test_LSTs,TvsLST_true[2]-TvsLST_uncal[2],label="true - uncal data",c="green",linewidth=2)
ax2c.plot(test_LSTs,-TvsLST_true[2]+TvsLST_post[2],label="true - post mean",linewidth=2,c="blue")
#ax2c.legend(loc="upper left")
#ax2c.set_ylabel(r"$\Delta$ Ant Temp (K)")

#row 2
ax1 = plt.subplot2grid((9,3),(3,0),rowspan=2)
ax1.plot(test_LSTs,TvsLST_true[3],label="true",c="red",linewidth=2)
ax1.plot(test_LSTs,TvsLST_uncal[3],label="uncal data",c="green",linewidth=2)
ax1.plot(test_LSTs,TvsLST_post[3],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1.legend(loc="upper left")
ax1.set_title(r"70MHz")
ax1.set_ylabel(r"Ant Temp (K)")

ax2 = plt.subplot2grid((9,3),(5,0),rowspan=1,sharex=ax1)
#ax2.plot(test_LSTs,TvsLST_true[3]-TvsLST_uncal[3],label="true - uncal data",c="green",linewidth=2)
ax2.plot(test_LSTs,-TvsLST_true[3]+TvsLST_post[3],label="true - post mean",linewidth=2,c="blue")
#ax2.legend(loc="upper left")
ax2.set_ylabel(r"$\Delta$ Ant"+"\nTemp (K)")

ax1b = plt.subplot2grid((9,3),(3,1),rowspan=2)
ax1b.plot(test_LSTs,TvsLST_true[4],label="true",c="red",linewidth=2)
ax1b.plot(test_LSTs,TvsLST_uncal[4],label="uncal data",c="green",linewidth=2)
ax1b.plot(test_LSTs,TvsLST_post[4],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1b.legend(loc="upper left")
ax1b.set_title(r"74MHz")
#ax1b.set_ylabel(r"Ant Temp (K)")

ax2b = plt.subplot2grid((9,3),(5,1),rowspan=1,sharex=ax1b)
#ax2b.plot(test_LSTs,TvsLST_true[4]-TvsLST_uncal[4],label="true - uncal data",c="green",linewidth=2)
ax2b.plot(test_LSTs,-TvsLST_true[4]+TvsLST_post[4],label="true - post mean",linewidth=2,c="blue")
#ax2b.legend(loc="upper left")
#ax2b.set_ylabel(r"$\Delta$ Ant Temp (K)")

ax1c = plt.subplot2grid((9,3),(3,2),rowspan=2)
ax1c.plot(test_LSTs,TvsLST_true[5],label="true",c="red",linewidth=2)
ax1c.plot(test_LSTs,TvsLST_uncal[5],label="uncal data",c="green",linewidth=2)
ax1c.plot(test_LSTs,TvsLST_post[5],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1c.legend(loc="upper left")
ax1c.set_title(r"80MHz")
#ax1c.set_ylabel(r"Ant Temp (K)")

ax2c = plt.subplot2grid((9,3),(5,2),rowspan=1,sharex=ax1c)
#ax2c.plot(test_LSTs,TvsLST_true[5]-TvsLST_uncal[5],label="true - uncal data",c="green",linewidth=2)
ax2c.plot(test_LSTs,-TvsLST_true[5]+TvsLST_post[5],label="true - post mean",linewidth=2,c="blue")
#ax2c.legend(loc="upper left")
#ax2c.set_ylabel(r"$\Delta$ Ant Temp (K)")

#row 3
ax1 = plt.subplot2grid((9,3),(6,0),rowspan=2)
ax1.plot(test_LSTs,TvsLST_true[6],label="true",c="red",linewidth=2)
ax1.plot(test_LSTs,TvsLST_uncal[6],label="uncal data",c="green",linewidth=2)
ax1.plot(test_LSTs,TvsLST_post[6],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1.legend(loc="upper left")
ax1.set_title(r"150MHz")
ax1.set_ylabel(r"Ant Temp (K)")

ax2 = plt.subplot2grid((9,3),(8,0),rowspan=1,sharex=ax1)
#ax2.plot(test_LSTs,TvsLST_true[6]-TvsLST_uncal[6],label="true - uncal data",c="green",linewidth=2)
ax2.plot(test_LSTs,-TvsLST_true[6]+TvsLST_post[6],label="true - post mean",linewidth=2,c="blue")
#ax2.legend(loc="upper left")
ax2.set_ylabel(r"$\Delta$ Ant"+"\nTemp (K)")
ax2.set_xlabel(r"LST (hours)")

ax1b = plt.subplot2grid((9,3),(6,1),rowspan=2)
ax1b.plot(test_LSTs,TvsLST_true[7],label="true",c="red",linewidth=2)
ax1b.plot(test_LSTs,TvsLST_uncal[7],label="uncal data",c="green",linewidth=2)
ax1b.plot(test_LSTs,TvsLST_post[7],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1b.legend(loc="upper left")
ax1b.set_title(r"159MHz")
#ax1b.set_ylabel(r"Ant Temp (K)")

ax2b = plt.subplot2grid((9,3),(8,1),rowspan=1,sharex=ax1b)
#ax2b.plot(test_LSTs,TvsLST_true[7]-TvsLST_uncal[7],label="true - uncal data",c="green",linewidth=2)
ax2b.plot(test_LSTs,-TvsLST_true[7]+TvsLST_post[7],label="true - post mean",linewidth=2,c="blue")
#ax2b.legend(loc="upper left")
#ax2b.set_ylabel(r"$\Delta$ Ant Temp (K)")
ax2b.set_xlabel(r"LST (hours)")

ax1c = plt.subplot2grid((9,3),(6,2),rowspan=2)
ax1c.plot(test_LSTs,TvsLST_true[8],label="true",c="red",linewidth=2)
ax1c.plot(test_LSTs,TvsLST_uncal[8],label="uncal data",c="green",linewidth=2)
ax1c.plot(test_LSTs,TvsLST_post[8],label="post mean",linewidth=2,linestyle="--",c="blue")
#ax1c.legend(loc="upper left")
ax1c.set_title(r"408MHz")
#ax1c.set_ylabel(r"Ant Temp (K)")

ax2c = plt.subplot2grid((9,3),(8,2),rowspan=1,sharex=ax1c)
#ax2c.plot(test_LSTs,TvsLST_true[8]-TvsLST_uncal[8],label="true - uncal data",c="green",linewidth=2)
ax2c.plot(test_LSTs,-TvsLST_true[8]+TvsLST_post[8],label="true - post mean",linewidth=2,c="blue")
#ax2c.legend(loc="upper left")
#ax2c.set_ylabel(r"$\Delta$ Ant Temp (K)")
ax2c.set_xlabel(r"LST (hours)")

plt.subplots_adjust(hspace=0.97)
plt.subplots_adjust(wspace=0.28)
plt.savefig(root+"/posterior_TvsLST_comp_all_freqs_v2.pdf")

#plt.show()


#========================================================
#plot the posterior of the spectra

#define the spectra functional form
def spec_func(fs,params):
    #print (params)
    #t = (fs/f0)**(params[0]+params[1]*np.log(fs/f0))
    t = LogParabola1D(1,f0,-params[0],-params[1])(fs)
    #print (t)
    #plt.plot(fs,t)
    #plt.show()
    return t

#select the samples with non zero weight
sammps_set2 = nested_samples[(weights!=0),:]
nested_samples = sammps_set2
print ("no of samps with non-zero weight is:",nested_samples.shape)
#plot a posterior of the spectra for each component
print ("plotting posterior of the spectra")
plt.rcParams['font.size'] = 14
fig = plt.figure(figsize=(8,4))
plt_freqs = np.logspace(1.653,2.7,200)
for i in range(no_of_comps):
    spec_pars = np.empty(shape=(nested_samples.shape[0],2))
    if no_to_fit_curve == no_of_comps:
        spec_pars[:,0] = nested_samples[:,i]
        spec_pars[:,1] = nested_samples[:,no_of_comps+i]
    else:
        if no_to_fit_curve==1:
            if fit_curved[i] == True:
                spec_pars[:,0] = nested_samples[:,i]
                spec_pars[:,1] = nested_samples[:,2]
            else:
                spec_pars[:,0] = nested_samples[:,i]
                spec_pars[:,1] = 0
        if no_to_fit_curve ==0:
            spec_pars[:,0] = nested_samples[:,i]
            spec_pars[:,1] = 0
    print ("unweighted mean of spec pars is:",np.mean(spec_pars,axis=0))

    plt.subplot(1,no_of_comps,i+1)
    #plot the posterior samples
    plot_lines(spec_func, plt_freqs, spec_pars)
    #plot the true spectra
    print ("true spectral params are:",true_specs[i])
    plt.plot(plt_freqs,spec_func(plt_freqs,true_specs[i]),linestyle="--",c="red",linewidth=1.5,label="true:"+r"$\beta=$"+str(true_specs[i][0])+"  "+r"$\gamma=$"+str(true_specs[i][1]))
    print ("mean post pars:",[mean_params[2*i],mean_params[2*i+1]])
    #plt.plot(plt_freqs,spec_func(plt_freqs,[mean_params[2*i],mean_params[2*i+1]]),linestyle="--",c="purple",label="posterior mean")
    
    plt.legend(loc="upper right")
    plt.xlabel("freq (MHz)",fontsize=15)
    if i==0:
        plt.ylabel("spectral scaling",fontsize=15)
    plt.yticks(ticks=[0.01,0.1,1,10,100],labels=[0.01,0.1,1,10,100])
    plt.ylim(0.01,100)
    plt.xscale('log')
    plt.yscale("log")
    



#plt.ylabel("spectral scaling")

plt.tight_layout()
plt.savefig(root+"/spectral_posterior.pdf")
#plt.show()