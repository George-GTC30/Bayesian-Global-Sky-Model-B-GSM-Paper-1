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
import gen_EDGES_beams as gen_beams_and_T_vs_LST_v2
from line_profiler import LineProfiler
import matplotlib
from astropy.modeling.powerlaws import LogParabola1D
from csv import writer

#RUN PARAMS
#================================================================================================
save_no = 50
chunksize=100
if save_no!=1:
    how_many_post_points = [-save_no*chunksize,-(save_no-1)*chunksize] #which posterior points to use
else:
    how_many_post_points = [-save_no*chunksize,None]
print ("running for the posteior points between indexes:",how_many_post_points)
how_many_maps = 10 #for each posterior point how many component maps will we use

#MODEL PARAMS
#================================================================================================
#declare a random seed

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
curvature_mean, curvature_std = 0,1 #the range for the prior on the spectral index curvature parameter

#params for fitting the reference frequency 

fixed_f0 = 150 #if you dont fit a seperate reference freq for each comp then we fix f0 to this value


#params for the prior on the true maps
map_prior_variance_spec_index = -2.8
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

test_freqs = [45.0,50.0,60.0,70,74,80,150,159,408]
#test_freqs = [70.0,74.0,80.0]
#test_freqs = [150.0,159.0,408.0]



unobs_marker = -32768

nlive = 500#2500*no_of_comps

precision_criterion = 1e-3


f0=150 #ref freq used for some fitting of spec indexes for plots (not used during any model fitting)

freqs = np.array([45.0,50.0,60.0,70.0,74.0,80.0,150.0,159.0,408.0]) #the frequencies in MHz of maps used to generate the model
#specify what to fit for each map
if calibrate_all == True:
    freqs_to_calibrate = np.array([True,True,True,True,True,True,True,True,True]) #calibrate all the maps except the 45 and 150 MHz
    freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])

if calibrate_all_but_45_150 == True:
    freqs_to_calibrate = np.array([False,True,True,True,True,True,False,True,True]) #calibrate all the maps except the 45 and 150 MHz
    freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])
if calibrate==False:
    freqs_to_calibrate = np.array([False,False,False,False,False,False,False,False,False])#np.array([True,True,True,True,True,True,False,False])
    freqs_to_fit_noise = np.array([False,False,False,False,False,False,False,fit_haslam_noise,fit_haslam_noise])

freqs_for_T_v_LST_comp = np.array([40.0,45.0,50.0,55.0,60.0,65.0,70.0,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200.0])


main_label = "_petur:"+str(use_perturbed_dataset)+"_"+str(no_of_comps)+"_comp_cal:"+str(calibrate)+"_rezro_pri_std:"+str(rezero_prior_std)+"_CMB="+str(subtract_CMB)+"_map_pri_std:"+str(map_prior_std)+"_mu:0_map_pri_std_spec_ind="+str(map_prior_variance_spec_index)+"_map_pri_f0="+str(map_prior_variance_f0)+"_cond_no_thres="+str(np.round(np.log10(cond_no_threshold),1))+"_crv_N_std="+str(curvature_std)+"_spec="+str(spec_min)+"_to:"+str(spec_max)#+"_rej_crit="+str(reject_criterion)#+"_nlive="+str(nlive)+"_nrept="+str(nrepeat)+"_precision_criterion="+str(precision_criterion)

if use_equal_spaced_LSTs==True:
    #LSTs_for_comparison = np.array([2,4,6,8,10,12,14,15,15.5,15.75,16,16.25,16.5,16.75,17,17.25,17.5,17.75,18,18.25,18.5,18.75,19,19.25,19.5,20,21,22])#np.array([0,2,4,6,8,10,12,14,16,18,20,22])#np.array([2.5,18]) #the LSTs in hours at which we will make comparison between the mean sky and EDGES for likelihood calls
    LSTs_for_comparison = np.linspace(0,24,73)[:-1]
    print (LSTs_for_comparison)
    print ("no of LSTs is:",len(LSTs_for_comparison))
    if calibrate_all==True:
        #root="uni_EDGES_v4_data_mk24_no_of_curved:"+str(no_to_fit_curve)+"_cal_all_f0="+str(fixed_f0)+main_label#"very_unequal_LST_lots_freq_vSTRG_BIAS"+main_label#"real_data_mk19_EDGES_"+main_label
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
    

#specify what to fit for each map


#set the prior for the noise (on the Haslam map)
noise_prior_lower, noise_prior_upper = np.array([0.01]),np.array([50])

print ("noise prior is from:",noise_prior_lower,"to",noise_prior_upper,"Kelvin")

n_spec_pars = 3*no_of_comps #the number of parameters for the spectra (for each comp we have: break_freq, spec_index1, spec_index2)


nv=len(freqs) #the number of freqs that have maps
no_of_fitted_noise = np.sum(freqs_to_fit_noise) #the number of freqs at which we fit noise level
no_of_calibrated = np.sum(freqs_to_calibrate) #the number of freqs at which we fit the calibration

if np.sum(freqs_to_calibrate)!=0:
    zero_lev_prior_std = rezero_prior_std*np.ones(np.sum(freqs_to_calibrate))#200*((np.array(freqs)[freqs_to_calibrate]/100)**-2.5)
    zero_lev_prior_means = np.zeros(np.sum(freqs_to_calibrate))


    #set the prior params for the scale corrections
    scale_prior_lower=0.85
    scale_prior_upper=1.25
    

    print ("zero level prior is gauss with mean 0K, stds (Kelvin):")
    print (zero_lev_prior_std)
    print ("temp scale prior is uniform from:",scale_prior_lower,"to",scale_prior_upper)






#CREATE A DIR TO STORE RESULTS
#====================================================================#


#make a dir to store the results
p=os.getcwd()+"/"
path = p+root+"/"
try:
    os.mkdir(root)
except:
    pass
#make a dir to store the results as we run
root2 = path+"running_results/"
try:
    os.mkdir(root2)
except:
    pass

#LOAD THE DATASET AND THE ERROR MAPS
#====================================================================#

obs_maps = []
inv_err_maps = []
data_err_maps = []
load_path = p+"mock_data_file/mock_dataset_v4/"
for i in range(len(freqs)):
    f=freqs[i]

    if use_perturbed_dataset==True:
        fname1 = "noisy_perturbed_sky_"+str(f)
        err_fname = "perturbed_err_map_"+str(f)
    else:
        fname1 = "noisy_sky_"+str(f)
        err_fname = "err_map_"+str(f)
    
    #fname2 = "noise_"+str(f)

    if freqs_to_fit_noise[i]==False:
        try:
            #m1, err_m = np.loadtxt(load_path+fname1), np.loadtxt(load_path+fname2)
            m1 = np.loadtxt(load_path+fname1)

            err_m = np.loadtxt(load_path+err_fname)
        except:
            print ("cant find the files for freq:",f)

        
        #mask out any pixels with negative temps
        bool_arr = m1<=0
        err_m[bool_arr] = unobs_marker
        m1[bool_arr] = unobs_marker

        

        inv_err_m = 1/err_m
        inv_err_m[(err_m==unobs_marker)] = 0

        m1[m1!=unobs_marker] = m1[m1!=unobs_marker]+subtract_CMB
        obs_maps.append(m1)
    
        inv_err_maps.append(inv_err_m)

        data_err_maps.append(err_m)
    else:
        m1 = np.loadtxt(load_path+fname1)
        m1[m1!=unobs_marker] = m1[m1!=unobs_marker]+subtract_CMB
        obs_maps.append(m1)
    

obs_maps=np.array(obs_maps)
inv_err_maps=np.array(inv_err_maps)
print ("dataset loaded")

#CREATE THE INVERSE NOISE MATRICES
#====================================================================#
#generate the inverse noise covariance matrix for each pixel
inv_noise_mats = np.empty(shape=(Max_m,len(freqs),len(freqs)))
for p in range(0,Max_m):
    inv_stds_for_pixel = np.zeros(len(freqs))
    inv_stds_for_pixel[~freqs_to_fit_noise] = inv_err_maps[:,p]
    #print (inv_stds_for_pixel)

    Np_inv = np.diag(inv_stds_for_pixel**2)
    #print (Np_inv)
    inv_noise_mats[p,:,:] = Np_inv

print ("max and min for the inv noise mats: ",np.max(inv_noise_mats),np.min(inv_noise_mats[(inv_noise_mats!=0)]))

print ("inverse noise matrices created")
#PLOT THE DATASET
#====================================================================#
fig = plt.figure(figsize=(12,16))
for i in range(len(freqs)):
    map_i = np.copy(obs_maps[i])
    ax = plt.subplot(5,3,int(i+1))
    map_i[(map_i==unobs_marker)]=float("NaN")
    plt.axes(ax)
    hp.mollview(map_i,title="Synthetic Data Freq="+str(freqs[i]),hold=True,notext=True,norm="log")



plt.savefig(path+"sky_maps_for_dataset_for_plt")
#plt.show()
fig = plt.figure(figsize=(12,16))
for i in range(len(freqs)):
    map_i = np.copy(inv_err_maps[i])
    ax = plt.subplot(5,3,int(i+1))
    map_i[(map_i==0)]=float("NaN")
    plt.axes(ax)
    hp.mollview(1/map_i,title="input errs freq="+str(freqs[i]),hold=True,notext=True,norm="log")



plt.savefig(path+"err_maps_for_dataset_for_plt")
freqs=np.array(freqs)
#plot the priors and the data
#====================================================================
log_mean_temps = []
mean_temps = []
for i in range(len(freqs)):
    the_map = obs_maps[i]
    mean = np.mean(the_map[(the_map!=unobs_marker)])
    log_mean_temps.append(np.log(mean))
    mean_temps.append(mean)
log_mean_temps = np.array(log_mean_temps)
mean_temps = np.array(mean_temps)

log_freqs = np.log(freqs/f0)
fun = lambda x: np.nansum((log_mean_temps - x[0]*log_freqs -x[1])**2)
res = minimize(fun,[-2.15,np.log(16)])
print (res)

fig = plt.figure(figsize=(6,6))#figsize=(12,16))
#the fitted powerlaw
targ = np.exp(res.x[1])*((freqs/f0)**res.x[0])

ax1 = plt.subplot(1,1,1)
ax1.plot(freqs,targ,c="red",label="fitted power law")
ax1.scatter(freqs,mean_temps,label="data")
ax1.set_title("map mean temps")
ax1.legend(loc="upper right")
ax1.set_xscale("log")
ax1.set_yscale("log")
plt.savefig(path+"/input_map_for_plt_mean_temps.png")


#generate a set of pre rotated EDGES beams at each of the frequencies that we want to compare the model to EDGES for
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print ("generating EDGES beams")
EDGES_beams = np.empty(shape=(obs_maps.shape[1],len(freqs_for_T_v_LST_comp),len(LSTs_for_comparison)))
j=0
for f in freqs_for_T_v_LST_comp:
    print ("gen EDGES beams for freq:",f)
    beams_at_LSTS = gen_beams_and_T_vs_LST_v2.gen_EDGES_beams_at_LSTs(f,LSTs_for_comparison,Max_Nside)
    EDGES_beams[:,j,:] = beams_at_LSTS
    #for i in range(len(LSTs_for_comparison)):
    #    beam = beams_at_LSTS[:,i]
        #print (beam)
        #print (beam.shape)
    #    hp.mollview(beam.flatten(),title="EDGES beam (galactic coords) for freq: "+str(f)+" for LST: "+str(LSTs_for_comparison[i]))
    #    plt.savefig(path+"EDGES_beam_galactic_coords_for_freq:"+str(f)+"_for_LST:"+str(LSTs_for_comparison[i])+".png")
    #    plt.close("all")
    j+=1

#Load the mock EDGES T vs LST plots (these are produced by convolving the mock sky map (before any pertubation) with the model beam at that freq)
#===============================================================================================
EDGES_temps_at_calib_LSTs_and_freqs = []
EDGES_errs = []
print ("generating the EDGES T vs LST traces")
for f in freqs_for_T_v_LST_comp:
   # if f<100:
   #     T_vs_LST, T_vs_LST_errs = gen_beams_and_T_vs_LST_v2.gen_EDGES_low_T_LST_trace(f,LSTs_for_comparison)
   #     EDGES_temps_at_calib_LSTs_and_freqs.append(T_vs_LST)
   #     EDGES_errs.append(T_vs_LST_errs)
   #     fig=plt.figure()
   #     plt.errorbar(LSTs_for_comparison,T_vs_LST,T_vs_LST_errs)
   #     plt.savefig(path+"EDGES_T_vs_LST_freq="+str(f)+".png")
   # if f>=100:
   #     T_vs_LST, T_vs_LST_errs = gen_beams_and_T_vs_LST_v2.gen_EDGES_high_T_LST_trace(f,LSTs_for_comparison)
   #     EDGES_temps_at_calib_LSTs_and_freqs.append(T_vs_LST)
   #     EDGES_errs.append(T_vs_LST_errs)
   #     fig=plt.figure()
   #     plt.errorbar(LSTs_for_comparison,T_vs_LST,T_vs_LST_errs)
   #     plt.savefig(path+"EDGES_T_vs_LST_freq="+str(f)+".png")
    T_vs_LST = np.loadtxt(load_path+"true_noisy_TvsLST_"+str(f))
    T_vs_LST_errs = np.loadtxt(load_path+"TvsLST_errs_"+str(f))
    print ("TvsLST for freq:",f)
    print (T_vs_LST)
    print (T_vs_LST_errs)
    EDGES_temps_at_calib_LSTs_and_freqs.append(T_vs_LST)
    EDGES_errs.append(T_vs_LST_errs)
    fig=plt.figure()
    plt.errorbar(LSTs_for_comparison,T_vs_LST,T_vs_LST_errs)
    plt.savefig(path+"EDGES_T_vs_LST_freq="+str(f)+".png")
plt.close("all")

EDGES_temps_at_calib_LSTs_and_freqs = np.array(EDGES_temps_at_calib_LSTs_and_freqs)
EDGES_errs = np.array(EDGES_errs)
#generate the EDGES noise covar mats for each freq assuming noise for each LST is independent of the other LSTs
EDGES_inv_noise_mats = []
for i in range(len(test_freqs)):
    inv_cov = np.diag(1/EDGES_errs[i,:]**2)
    EDGES_inv_noise_mats.append(inv_cov)
EDGES_inv_noise_mats = np.array(EDGES_inv_noise_mats)
#=====================================================================================================
#set up the likelihood function
#bayes_eval = likelihood.bayes_mod(obs_maps=obs_maps,obs_freqs=freqs,inv_noise_mats=inv_noise_mats,gaussian_prior_covar_mat=gaussian_prior_covar_matrix,gaussian_prior_mean=gaussian_prior_mean,no_of_comps=no_of_comps,f0=f0,un_obs_marker=unobs_marker)

#set up the likelihood function
bayes_eval = likelihood.bayes_mod(obs_maps=obs_maps,obs_freqs=freqs,inv_noise_mats=inv_noise_mats,EDGES_beams=EDGES_beams,EDGES_temps_at_calib_LSTs_and_freqs=EDGES_temps_at_calib_LSTs_and_freqs,EDGES_errs=EDGES_errs,EDGES_inv_noise_mats=EDGES_inv_noise_mats,freqs_for_T_v_LST_comp=freqs_for_T_v_LST_comp,LSTs_for_comparison=LSTs_for_comparison,no_of_comps=no_of_comps,save_root=root2,un_obs_marker=unobs_marker,map_prior_std=map_prior_std,map_prior_spec_index=map_prior_variance_spec_index,map_prior_f0=map_prior_variance_f0)

#test
print ("===================================")
print ("testing")
sample_map = bayes_eval.gen_comp_map_sample([150,-2.50,0,150,-0.5,-2.5,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],freqs_to_fit_noise,freqs_to_calibrate)
for c in range(no_of_comps):
    m = sample_map[:,c,0]
    hp.mollview(m)
    plt.savefig("test_c="+str(c)+".png")


no_of_params_for_spec_mod = no_of_comps + no_to_fit_curve 
print ("we arn't fitting f0: f0=",fixed_f0," no of comps with curved spectra is",no_to_fit_curve," no of params for spectral model is",no_of_params_for_spec_mod)


#-------------NESTED SAMPLING PARAMS-------------
nDims =  int(no_of_params_for_spec_mod + no_of_fitted_noise + 2*no_of_calibrated)
nrepeat = 5*nDims #the nrepeat is set to 5 times the total number of pars that we fit
print ("no of dimensions for sampling region is:",nDims)
nDerived = 0 #we don't derive any parameters 
settings = PolyChordSettings(nDims, nDerived)
settings.file_root = root
settings.nlive = nlive
settings.nrepeats = nrepeat
settings.do_clustering = True
settings.read_resume = True
settings.write_resume = True
settings.maximise = False #find the maximum of the poseterior
settings.precision_criterion = precision_criterion
#------------------------------------------------

#find the mean spectral params and their standard deviations
samples = NestedSamples(root= settings.base_dir + '/' + settings.file_root)

mean_logZ = samples.logZ()#mean
std_logZ = samples.logZ(100).std()#100 posterior samples from estimate of log Z (use these to calculate standard deviation in Z)

print ("log Z",mean_logZ,"std",std_logZ)

def abs_temp_likelihood(comp_map_sample,spec_params):
    comp_map_samp[np.isnan(comp_map_samp)]=0
    comp_maps = np.zeros(shape=(bayes_eval.no_of_pixels,bayes_eval.no_of_comps,1))
    #print (np.sum(bayes_eval.pix_to_regard))
    comp_maps= comp_map_sample#[bayes_eval.pix_to_regard] = comp_map_sample

    spectral_mix_mat_for_test = bayes_eval.gen_A(spec_params,bayes_eval.freqs_for_T_v_LST_comp)#self.freqs_for_T_v_LST_comp[:,np.newaxis] ** spec_params[np.newaxis,:]
    #print ("spectral mix mat")
    #print (spectral_mix_mat_for_test)
    sky_preds = spectral_mix_mat_for_test @ comp_maps
    sky_preds[~bayes_eval.pix_to_regard] = float("NaN")

    convolved_sky_preds = sky_preds * bayes_eval.EDGES_beams

    #compute the integrated sky temp for each freq and LST in the freqs to compare
    integrated_skys = np.nansum(convolved_sky_preds,axis=0)

            
    EDGES_likelihood_t2s = ((integrated_skys-bayes_eval.EDGES_temps)/bayes_eval.EDGES_errs)**2
    EDGES_likelihood_t2 = -1*np.sum(EDGES_likelihood_t2s)#-1 * np.sum(diff_trans @ self.EDGES_inv_noise_mats @ diff)
    EDGES_int_sky_temps_log_likelihood = EDGES_likelihood_t2 + bayes_eval.EDGES_noise_det_term#(self.no_of_defined_pix/len(self.LSTs_for_comparison))*(EDGES_likelihood_t2 + self.EDGES_noise_det_term)
            

    return EDGES_int_sky_temps_log_likelihood


#load the samples from the marginal posterior
marg_post_samps = np.loadtxt(root+"/post_samples_marginal.csv",delimiter=",")
weights = np.loadtxt(root+"/post_samples_marginal_weights.csv",delimiter=",")

print (marg_post_samps)
#loop through the marginal posterior samps
abs_temp_likelihood_values = []
count = 0
for marg_samp in marg_post_samps[how_many_post_points[0]:how_many_post_points[1]]:
    #print (marg_samp)
    for i in range(how_many_maps):
        #generate a map sample
        comp_map_samp = bayes_eval.gen_comp_map_sample(marg_samp,freqs_to_fit_noise,freqs_to_calibrate)
        #print (i)
        #print (comp_map_samp)

        #compute the EDGES likelihood term for this comp map sample
        log_likelihood_value = abs_temp_likelihood(comp_map_samp,marg_samp[:6]) 
        #print ("log likelihood:",log_likelihood_value)
        abs_temp_likelihood_values.append(log_likelihood_value)
    count+=1
    print("percent done:",100*count/abs(chunksize))

abs_temp_likelihood_values = np.array(abs_temp_likelihood_values)
print ("mean likelihood val:",np.mean(abs_temp_likelihood_values))
print ("std:",np.std(abs_temp_likelihood_values))
np.savetxt(root+"/abs_temp_likelihood_values"+str(save_no),abs_temp_likelihood_values,delimiter=",")
plt.figure(figsize=(6,6))
plt.rcParams['font.size'] = 14
plt.hist(abs_temp_likelihood_values,bins=50)
plt.xlabel(r"log(P(E|M,S))")
plt.ylabel("occurrence")
plt.savefig(root+"/histo_of_abs_temp_likelihood_vals.pdf")
plt.show()