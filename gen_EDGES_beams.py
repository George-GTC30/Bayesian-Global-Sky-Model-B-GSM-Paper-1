import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import os
import healpy as hp


def gen_EDGES_beam(freq,Max_Nside,azimuth=0,lat_FWHM=None,long_FWHM=None,beam_gain=None):

    Npix = hp.nside2npix(Max_Nside)
    pixel_thetas,pixel_phis = hp.pix2ang(Max_Nside, np.arange(Npix))

    #mask out below the EDGES obs horizon
    obs_lat = -26.7
    rot3=hp.Rotator(rot=(0,90-obs_lat))
    mask=np.zeros(Npix)
    mask[(pixel_thetas<=np.pi/2)] = 1
    mask=rot3.rotate_map_alms(mask)
    
    pixel_thetas,phis = hp.pix2ang(Max_Nside, np.arange(Npix),lonlat=True)

    #lat_FWHM = 112
    #long_FWHM = 72
    #beam_gain = 7

    if beam_gain==None:
        beam_gain = np.interp(freq,np.array([45,150]),np.array([7,5.89]))
    #lat_FWHM = np.interp(freq,np.array([45,150]),np.array([98,110]))
    #long_FWHM = np.interp(freq,np.array([45,150]),np.array([68,72]))
    if long_FWHM==None:
        long_FWHM = np.interp(freq,np.array([45,150]),np.array([98,112]))
    if lat_FWHM==None:
        lat_FWHM = np.interp(freq,np.array([45,150]),np.array([68,72]))

    #print ("for freq",freq,"gain",beam_gain,"lat FWHM",lat_FWHM,"long_FWHM",long_FWHM)

    beam = np.exp(-0.5*(phis/(lat_FWHM/2.355))**2)*np.exp(-0.5*((pixel_thetas-180)/(long_FWHM/2.355))**2)
    
     #center the peak of the beam to a longditude of 0 deg
    rot=hp.Rotator(rot=(180,0))
    beam = rot.rotate_map_alms(beam)
    #hp.mollview(beam,title="beam straight up")
    #plt.show()

    #rotate so the beams peak is at the north pole 
    rot=hp.Rotator(rot=(0,-90))
    beam = rot.rotate_map_alms(beam)
    #hp.mollview(beam,title="beam center at north pole")
    #plt.show()
    
    #rotate to the correct azimuth angle (rotate around the north pole)
    rot1=hp.Rotator(rot=(azimuth,0))
    beam = rot1.rotate_map_alms(beam)
    #hp.mollview(beam,title="rotated to azimuth="+str(azimuth))
    #plt.show()
    
    #rotate back to having the beam point straight up but at the correct azimuth angle
    rot=hp.Rotator(rot=(0,90))
    beam = rot.rotate_map_alms(beam)
    #hp.mollview(beam,title="beam straight up at azimuth angle"+str(azimuth))
    #plt.show()

    
    #rotate the beam to correct for our observation latitude 
    rot2=hp.Rotator(rot=(0,0-obs_lat))
    beam = rot2.rotate_map_alms(beam)

    beam *=beam_gain

    
    return beam, mask


def gen_EDGES_beams_at_LSTs(freq,LSTs,Nside,azimuth=0,test=False,lat_FWHM=None,long_FWHM=None,beam_gain=None):
    beam_at_LST0, horizon0 = gen_EDGES_beam(freq,Nside,azimuth=azimuth,lat_FWHM=lat_FWHM,long_FWHM=long_FWHM,beam_gain=beam_gain) #the beam for LST 0 in equatorial coords
    rot4 = hp.Rotator(coord=["C","G"])
    beams = []
    
    if test==True:
        hp.mollview(beam_at_LST0,title="beam at LST=0")
        plt.show()
        ts =[]
        m = hp.read_map(os.getcwd()+"/dataset_Haslam_LWA1_uncal_Guz_LW_Monsalve_cal_smoothed_FWHM=5_at_Nside=32/map_"+str(freq)+"MHz_FWHM=5_nside=32.fits")
        m[(m==-32768)]=float("NaN")
        for LST in LSTs:
            rotation = hp.Rotator(rot=(-15*LST,0))
            beam = rotation.rotate_map_alms(beam_at_LST0)
            horizon = rotation.rotate_map_alms(horizon0)
            #hp.mollview(beam,title="beam for LST="+str(LST))
            #plt.show()

            #rotate the beam into galactic coords to match the map, do the same to the horizon
            #rotate to galactic coords
            beam = rot4.rotate_map_alms(beam)
            horizon = rot4.rotate_map_alms(horizon)
            
            mask2use=np.zeros(len(beam))
            mask2use[(horizon<0.5)] = 0
            mask2use[(horizon>=0.5)] = 1
            #mask below the horizon
            beam *= mask2use
            beams.append(beam)
            #hp.mollview(beam,title="beam for LST="+str(LST)+" Galactic coords")
            #plt.show()

            m1=np.zeros(len(beam))
            m2=np.zeros(len(beam))

            #m1[(beam>0.495*np.max(beam))]=1
            #m2[(beam<0.505*np.max(beam))]=1
            #hp.mollview(m1*m2)
            #plt.show()
            #print (np.nansum((1/(4*np.pi))*beam*m*hp.nside2pixarea(Nside)))
            ts.append(np.nansum((1/(4*np.pi))*beam*m*hp.nside2pixarea(Nside)))
        return ts,beams
    
    else:
        beams = np.zeros((len(beam_at_LST0),len(LSTs)))
        i=0
        for LST in LSTs:
            rotation = hp.Rotator(rot=(-15*LST,0))
            beam = rotation.rotate_map_alms(beam_at_LST0)
            horizon = rotation.rotate_map_alms(horizon0)
            #hp.mollview(beam,title="beam for LST="+str(LST))
            #plt.show()

            #rotate the beam into galactic coords to match the map, do the same to the horizon
            #rotate to galactic coords
            beam = rot4.rotate_map_alms(beam)
            horizon = rot4.rotate_map_alms(horizon)
            
            mask2use=np.zeros(len(beam))
            mask2use[(horizon<0.5)] = 0
            mask2use[(horizon>=0.5)] = 1
            #mask below the horizon
            beam *= mask2use

            beams[:,i] = beam
            i+=1
        return beams
#lst = np.linspace(0,24,97)
#t,b=gen_EDGES_beams_at_LSTs(45,np.array(lst),32,azimuth=-5,test=True)#,lat_FWHM=71.6,long_FWHM=110)
#print ("simulated ant temps for 45MHz")
#print (t)
#zs,z2s=gen_EDGES_high_T_LST_trace(45,lst)



#fig = plt.figure(figsize=(10,10))
#plt.errorbar(lst,zs,z2s,label="gen from spec index")#
#plt.plot(lst,t,label="simulated obs using EDGES beam model")
#plt.legend(loc="upper left")
#plt.title("EDGES high 45 MHz")
#plt.show()

#t=np.array(t)
#zs=np.array(zs)
#func = lambda x: np.sum(((zs-x[0]*t-x[1])/np.array(z2s))**2)
#res=minimize(func,[1,0])
#print (res)

#print ("reduced chi sqr after corrections:",res.fun/(len(lst)-2))

#fig = plt.figure(figsize=(10,10))
#plt.errorbar(lst,zs,z2s,label="gen from spec index")#
#plt.plot(lst,res.x[0]*t+res.x[1],label="simulated obs using EDGES beam model")
#plt.legend(loc="upper left")
#plt.title("EDGES high 45 MHz")
#plt.show()


