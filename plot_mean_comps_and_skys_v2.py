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
import pix_by_pix_mk24_final2 as likelihood

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
#import gen_beams_and_T_vs_LST as gen_beams_and_T_vs_LST_v2
from line_profiler import LineProfiler
import matplotlib
from astropy.modeling.powerlaws import LogParabola1D
from csv import writer

#========================================================================================#
#compute mean posterior component amplitude maps
calc_mean_comps = False
#compute mean posterior sky predictions
calc_skys = True

#========================================================================================#
#a very large number (~60 thousand) posterior samples have very low weighting <10^-50
#setting this to True means we only use the more highly weighted posterior samples
use_limited_post = True


#specify the frequencies we will generate posterior predictions at
#========================================================================================#
#test_freqs = np.array([45.0,50,60])
#test_freqs = np.array([70.0,74,80])
#test_freqs = np.array([150.0,159,408])
#test_freqs = np.array([47.5,75,100,140])
#test_freqs = np.array([200.0,250.0])
test_freqs = np.array([300.0,350.0])
#test_freqs = np.array([400.0])

print ("test freqs are:",test_freqs)
#declare a random seed
np.random.seed(0)
use_perturbed_dataset = True #do we want the input dataset to have calibration errors

#===================================================================#
#| SET PARAMS FOR THE SIMULATED DATA AND NESTED SAMPLING
Max_Nside=32 #the Nside at which to generate the set of maps
Max_m = hp.nside2npix(Max_Nside)
no_of_comps = 2
fit_curved=[True,True]


no_to_fit_curve = np.sum(fit_curved)
rezero_prior_std=2000

#params for the prior on the spectra
spec_min, spec_max = -3.5,1 #the range for the prior on the spectral indexes
curvature_mean, curvature_std = 0,3 #the range for the prior on the spectral index curvature parameter

#params for fitting the reference frequency 

fixed_f0 = 150 #if you dont fit a seperate reference freq for each comp then we fix f0 to this value


#params for the prior on the true maps
map_prior_variance_spec_index = -2.6
map_prior_variance_f0 = 408#fixed_f0
map_prior_std=300

calibrate = True
calibrate_all_but_45_150 = False#True #calibrate all maps in the dataset but the 45 and 150 MHz maps
calibrate_all = True#False #calibrate every map in the dataset


use_equal_spaced_LSTs = True
fit_haslam_noise = False
subtract_CMB = 0#-2.726
print_vals_as_calc = False



reject_criterion = 1e-3#None #how close can two spectral idexes be in value before being rejected
cond_no_threshold =1e+9




test_LSTs = np.linspace(0,24,73)[:-1]#np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])#np.array([0,2,4,6,8,10,12,14,16,18,20,22]) #the LSTs in hours at which we will make comparison between the mean sky and EDGES for likelihood calls
#test_freqs = [47.5,75]#,250,300,350]#[45,50.005,59.985,70.007,73.931,79.960,100,125,150,159,200,408]
#test_freqs = [100,140,200]
#test_freqs = [250,300,350]

#np.array([45.0,50.0,60.0,70,74,80,150,159,408])
#test_freqs = [70.0,74.0,80.0]
#test_freqs = [150.0,159.0,408.0]



unobs_marker = -32768

nlive = 500#2500*no_of_comps

precision_criterion = 1e-3


#specify what to fit for each map
if calibrate_all_but_45_150 == True:
    freqs_to_calibrate = np.array([False,True,True,True,True,True,False,True,True]) #calibrate all the maps except the 45 and 150 MHz
    freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])

if calibrate_all == True:
    freqs_to_calibrate = np.array([True,True,True,True,True,True,True,True,True]) #calibrate all the maps except the 45 and 150 MHz
    freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])

if calibrate == False:
    freqs_to_calibrate = np.array([False,False,False,False,False,False,False,False,False])#np.array([True,True,True,True,True,True,False,False])
    freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])


main_label = "_petur:"+str(use_perturbed_dataset)+"_"+str(no_of_comps)+"_comp_cal:"+str(calibrate)+"_rezro_pri_std:"+str(rezero_prior_std)+"_CMB="+str(subtract_CMB)+"_map_pri_std:"+str(map_prior_std)+"_mu:0_map_pri_std_spec_ind="+str(map_prior_variance_spec_index)+"_map_pri_f0="+str(map_prior_variance_f0)+"_cond_no_thres="+str(np.round(np.log10(cond_no_threshold),1))+"_crv_N_std="+str(curvature_std)+"_spec="+str(spec_min)+"_to:"+str(spec_max)#+"_rej_crit="+str(reject_criterion)#+"_nlive="+str(nlive)+"_nrept="+str(nrepeat)+"_precision_criterion="+str(precision_criterion)

if use_equal_spaced_LSTs==True:
    #LSTs_for_comparison = np.array([2,4,6,8,10,12,14,15,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,20,21,22])#np.array([0,2,4,6,8,10,12,14,16,18,20,22])#np.array([2.5,18]) #the LSTs in hours at which we will make comparison between the mean sky and EDGES for likelihood calls
    LSTs_for_comparison = np.linspace(0,24,73)[:-1]
    print (LSTs_for_comparison)
    print ("no of LSTs is:",len(LSTs_for_comparison))
    if calibrate_all==True:
        root="uni_EDGES_v4_data_mk24_no_of_curved:"+str(no_to_fit_curve)+"_cal_all_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label
    else:
        if calibrate_all_but_45_150==True:
            root="uni_EDGES_v4_dat_mk24_no_curve:"+str(no_to_fit_curve)+"_no_cal_45_150_f0="+str(fixed_f0)+main_label
else:
    #LSTs_for_comparison = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11.25,11.5,11.75,12,12.25,12.5,12.75,13,13.25,13.5,13.75,14,14.25,14.5,14.75,15,15.25,15.5,15.75,16,16.25,16.5,16.75,17,17.1,17.2,17.3,17.4,17.5,17.6,17.7,17.8,17.9,18,18.25,18.5,18.75,19,19.25,19.5,19.75,20,20.25,20.5,20.75,21,21.25,21.5,21.75,22,22.25,22.5,22.75,23,23.25,23.5,23.75])
    LSTs_for_comparison = np.array([0,2,4,6,8,10,12,14,15,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,20,21,22])
    print (LSTs_for_comparison)
    print ("no of LSTs is:",len(LSTs_for_comparison))
    
    root="mk24_extra_uneq_LSTs:"+str(len(LSTs_for_comparison))+"_fixed_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label


p=os.getcwd()+"/"
path = p+root+"/"

#load the marginal samples and comp map samples
print ("loading marginal samples and weights")
marg_samps = np.loadtxt(path+"post_samples_marginal.csv",delimiter=",")
weights = np.loadtxt(path+"post_samples_marginal_weights.csv",delimiter=",")

import gzip

if use_limited_post == True:
    #we throw away the 60000 lowest weighted samples
    #this significantly reduces memory requierments 
    weights = weights[60000:]
    marg_samps = marg_samps[60000:,:]

    print ("loading the component amplitude samples")
    start = 7
else:
    #we use all posterior samples 
    start = 1
for i in range(start,10):
    name = path+"post_samples_chunk_"+str(int(i))+"_comp_maps.npy.gz"

    try:
        print ("opening chunk:",i)
        f = gzip.GzipFile(name,"r")

        chunk = np.load(f)
        f.close()

        if i ==start:
            maps = chunk
        else:
            maps = np.append(maps,chunk,axis=0)
    except:
        print ("no file for this chunk")
    print (maps.shape)

no_of_pixels = int(maps.shape[1]/no_of_comps)
no_of_samps = maps.shape[0]

if calc_mean_comps==True:
    samples_of_maps_DF = WeightedDataFrame(maps,weight=weights)#np.array(samples_of_maps)
    print (samples_of_maps_DF)

    posterior_mean = samples_of_maps_DF.mean() #,axis=0)
    posterior_std = samples_of_maps_DF.std()#np.nanstd(samples_of_maps,axis=0)#/np.sqrt(nested_samples.shape[0])


    #convert dataframe to np array
    posterior_mean = posterior_mean.to_numpy()
    posterior_std = posterior_std.to_numpy()
    #print (posterior_mean.shape)

    recovered_map = np.empty((no_of_pixels,no_of_comps))
    err_map = np.empty((no_of_pixels,no_of_comps))
    bool_arr_template = np.zeros(no_of_comps,dtype=bool)
    #comps_arr = np.empty((no_of_comps,no_of_comps,no_of_pixels))
    for c in range(no_of_comps):
        bool_arr_temp = np.copy(bool_arr_template)
        bool_arr_temp[c] = 1

        bool_arr = np.tile(bool_arr_temp,no_of_pixels)

        recovered_map[:,c]=posterior_mean[bool_arr]
        err_map[:,c]=posterior_std[bool_arr]#[c*no_of_pixels:(c+1)*no_of_pixels]
    #    print (recovered_map)

#    comps_arr[:,c,:] = maps[:,bool_arr]

    print (recovered_map)
    #save the mean componetent maps and their errors
    for i in range(no_of_comps):
        rec_comp, rec_comp_errs = recovered_map[:,i], err_map[:,i]

        np.savetxt(path+'posterior_mean_for_comp_'+str(i+1),rec_comp,delimiter=",")
        np.savetxt(path+'posterior_std_for_comp_'+str(i+1),rec_comp_errs,delimiter=",")
    #plot the component maps
    fig = plt.figure(figsize=(15,10))

    for i in range(no_of_comps):
        ax = plt.subplot2grid((2,no_of_comps),(0,i))

        plt.axes(ax)
        hp.mollview(recovered_map[:,i],title="recovered mean comp "+str(i+1),hold=True,notext=True)#,min=-10,max=10)#np.max(c_map_1))

    for i in range(no_of_comps):
        ax = plt.subplot2grid((2,no_of_comps),(1,i))

        plt.axes(ax)
        hp.mollview(err_map[:,i],title="errs mean comp "+str(i+1),hold=True)#,notext=True,min=-10,max=10)#np.max(c_map_1))

    plt.savefig(path+"recovered_comp_comparison.png")

#gen the skys
bool_arr_template = np.zeros(no_of_comps,dtype=bool)
comps_arr = np.empty((no_of_samps,no_of_comps,no_of_pixels))
for c in range(no_of_comps):
    bool_arr_temp = np.copy(bool_arr_template)
    bool_arr_temp[c] = 1

    bool_arr = np.tile(bool_arr_temp,no_of_pixels)
    comps_arr[:,c,:] = maps[:,bool_arr]

def gen_samp_sky(index):

    spec_params = marg_samps[index,:3*no_of_comps]

    comp_maps_samp = comps_arr[index]

    specs = np.empty((len(test_freqs),no_of_comps))
    for i in range(no_of_comps):
        sp = spec_params[3*i:3*(i+1)]
    #    print (sp)
        spec = LogParabola1D(1,sp[0],-sp[1],-sp[2])(test_freqs)
    #    print (spec)
        specs[:,i]=spec
    specs = np.array(specs)
    #print (specs)
    s1 = specs @ comp_maps_samp

    #print (s1)
    #print (s1.flatten())
    return s1.flatten()

if calc_skys == True:
    print ("generating sky samples")
    sky_samps = []
    for i in range(no_of_samps):
        sky_samps.append(gen_samp_sky(i))

    sky_samps = np.array(sky_samps)

    samples_of_maps_DF = WeightedDataFrame(sky_samps,weight=weights)#np.array(samples_of_maps)
    print (samples_of_maps_DF)

    posterior_mean = samples_of_maps_DF.mean() #,axis=0)
    posterior_std = samples_of_maps_DF.std()#np.nanstd(samples_of_maps,axis=0)#/np.sqrt(nested_samples.shape[0])


    #convert dataframe to np array
    posterior_mean = posterior_mean.to_numpy()
    posterior_std = posterior_std.to_numpy()

    for i in range(len(test_freqs)):
        f=test_freqs[i]
        sky = posterior_mean[i*no_of_pixels:(i+1)*no_of_pixels]
        sky_err = posterior_std[i*no_of_pixels:(i+1)*no_of_pixels]

        np.savetxt(path+"bayesian_pred_"+str(f)+"MHz",sky,delimiter=",")
        np.savetxt(path+"bayesian_errs_"+str(f)+"MHz",sky_err,delimiter=",")

        hp.mollview(sky,title=str(f))
        plt.savefig(path+str(f)+".png")

        hp.mollview(sky_err,title=str(f))
        plt.savefig(path+str(f)+"_err.png")

