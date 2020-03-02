'''
Module Fantomanalysis for analysing phantom dumpfiles in ascii format.
This module contains 4 functions:

-> read
-> flag_dust
-> update_particle_traj
-> bins

If you are on a jupyter notebook or ipython, you can type either one of these function along with "?" to have more informations.
'''
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

# constants used by the library
msol = 1.98909993E+30 # kg
au   = 1.49600002E+11 # m
yr   = 365.25 * 86400 # s

utime  = yr/(2*np.pi)
umass  = msol
udist  = au
uvel   = udist/utime
udens  = umass/(udist**3)
usigma = umass/(udist**2)

def read(file, nrows=1200001, rin=20, rout=300, clean=True):
    '''
    Reads an ascii file and cleans the data by rin (r>rin), rout (r<rout) and smoothing lentghs (h>0).
    Changes the units from code to SI, except for x, y, z (AU).
    Arguments are:
    file  - (str)  : name of the phantom dump file to read (ascii)
    nrows - (int)  : number of rows to read (= npart)
    rin   - (float): inner radius for cleaning
    rout  - (float): outer radius for cleaning
    clean - (bool) : wether or not to clean the data
    '''
    
    # info relative to file reading
    header = ['x', 'y', 'z', 'm', 'h', 'rho', 'dustfrac', 'grainsize', 'graindens', 'vrelvf', 'cs', 'rhogas', 'st', 'dv', 'vx', 'vy', 'vz', 'divv', 'dt', 'type']

    # read infile data
    data = pd.read_csv(file, delim_whitespace=True, skiprows=14, nrows=nrows, header=None, names=header)
    timeline = np.loadtxt(file, skiprows=3, max_rows=1, usecols=(1,2), comments='&')
    time = int(timeline[0] * timeline[1])
    
    # cleanup of the dataset by rin, rout and h
    r = np.sqrt(data.x*data.x + data.y*data.y)
    data.loc[:,"r"] = r
    if clean:
        data = data[(data.r>rin) & (data.r<rout) & (data.h>0)]
    
    # add new quantities
    vr = np.sqrt(data.vx*data.vx + data.vy+data.vy)
    data.loc[:,"vr"] = vr
    
    # transform concerned columns in SI units - lenghts are still in au
    data.loc[:,"m"]         = data.loc[:,"m"]*umass
    data.loc[:,"rho"]       = data.loc[:,"rho"]*udens
    data.loc[:,"grainsize"] = data.loc[:,"grainsize"]*udist
    data.loc[:,"graindens"] = data.loc[:,"graindens"]*udist
    data.loc[:,"cs"]        = data.loc[:,"cs"]*uvel
    data.loc[:,"rhogas"]    = data.loc[:,"rhogas"]*udens
    data.loc[:,"dv"]        = data.loc[:,"dv"]*uvel
    data.loc[:,"vr"]        = data.loc[:,"vr"]*uvel
        
    return data, time
    
def flag_dust(file, nrows=1200001, by_r=True, radii=None, tolr=1.e-2, by_z=False, alti=None, tolz=1.e-1, by_size=False, sizes=None, tols=1.e-2, random_choice=True, one_fluid=False):
    '''
    Returns a list of indexes conrresponding to dust particles fullfiling certain conditions
    Arguments are:
    file          - (str)           : name of the phantom file where particles need to be flagged (ascii)
    nrows         - (int)           : number of rows to read in dump file (= npart)
    by_r          - (bool)          : wether or not to flag particles by their distance to the star
    by_z          - (bool)          : wether or not to flag particles by their altitude in the disc
    by_size       - (bool)          : wether or not to flag dust particles by their size
    radii         - (seq. or number): radii where to look for dust particles
    alti          - (seq. or number): altitude where to look for dust particles
    sizes         - (seq. or number): sizes where to look for dust particles
    tolr          - (float)         : tolerance on radius for finding dust particles
    tolz          - (float)         : tolerance on altitude for finding dust particles
    tols          - (float)         : tolerance on size for finding dust particles
    random_choice - (bool)          : wether or not to return only one randomly selected particles fullfiling the conditions
    '''
    # read file, do not clean in case particle is accreted
    data, time = read(file, clean=False)
    if one_fluid:
        dust = data[data.type==1]
    else:
        dust = data[data.type==8]
    
    # every dimension of radii has its own dataframe containing chosen particles
    flagged = np.empty(len(radii), dtype=pd.DataFrame)
    
    # flag the particles by applying masks according to the user's choice
    for i in range(0, len(flagged)):
        if by_r:
            flagged[i] = dust[np.absolute(dust.r-radii[i])/dust.r < tolr]
        if by_z:
            flagged[i] = flagged[i][np.absolute((flagged[i].z-alti[i])/flagged[i].z) < tolz]
        if by_size:
            flagged[i] = flagged[i][np.absolute(flagged[i].grainsize-sizes[i])/flagged[i].grainsize < tols]
        flagged[i] = flagged[i].index
        if random_choice:
            flagged[i] = np.random.choice(flagged[i])
    
    # print out some info
    print(f"{len(radii)} particles selected:\n")
    print(f"r ---- z ---- size \n")
    for index in flagged:
        print(dust.loc[index,"r"], dust.loc[index,"z"], dust.loc[index,"grainsize"])
            
    return flagged
    
def update_particle_traj(file, part_ind, nrows=1200001, dfs=None):
    '''
    Reads data from file (ascii) and continue filling the dataframe "dfs" with particles selected with the seq. part_ind.
    If dfs is None (first time calling this function), creates dataframe and start filling it.
    Arguments are:
    file     - (str)           : filename of the phantom dumpfile (in ascii)
    part_ind - (seq. or number): index of particle(s) to track
    nrows    - (int)           : number of rows to read in dump file (= npart)
    nrows    - (int)           : number of rows to read in the dump file (= npart)
    dfs      - (DataFrame)     : array of dataframe contianing the particles data at every iterations of this function. If None, creates it.
    '''
    # read ascii file and puts the data in dataframe
    data, time = read(file=file, nrows=nrows, clean=False)
    
    # add time as a new column
    data.loc[:,"time"] = time

    # apply mask on data using the particle index seq. part_ind
    rows = data.loc[part_ind,:]
    
    # create dataframe if first time around
    if dfs is None:
        dfs = np.empty(len(part_ind), dtype=pd.DataFrame)
    
    # fill passed dataframe with selected particles
    for i in range(0, len(part_ind)):
        if dfs[i] is None:
            dfs[i] = pd.DataFrame(rows.loc[part_ind[i]]).T
        else:
            dfs[i] = dfs[i].append(rows.loc[part_ind[i]].T, ignore_index=True)

    return dfs
    
def bins(file, nrows=1200001, rin=20, rout=300, logr=True, rbins=200, vazmin=15, zbins=15, sbins=100, one_fluid=False):
    '''
    Process an ascii phantom dumpfile and returns (in that order):
    -> Time as a float
    -> Radial profiles of various quantities as a DataFrame
    -> Altitude profiles of various quantities as a DataFrame
    -> Size profiles of various quantities as a DataFrame
    -> Stokes number profiles of various quantities as a DataFrame
    Arguments are:
    file   - (str)  : filename of the phantom dumpfile (in ascii)
    nrows  - (int)  : number of rows to read in dumpfile (=npart)
    rin    - (float): inner radius inside of which particles are deleted
    rout   - (float): outer radius outsidre of which particles are deleted
    logr   - (bool) : wether or not to bin radius in logscale
    rbins  - (int)  : number of bins along the r axis
    vazmin - (float): absolute value of the minimum altitude where altitude binning starts (symmetrical)
    zbins  - (int)  : number of bins along the z axis between vazmin and -vazmin
    sbins  - (int)  : number of size & St bins (only logspaced)
    '''
    
    # read file and clean it
    data, time = read(file=file, nrows=nrows, rin=rin, rout=rout, clean=True)

    # define idust and igas depending on dust method
    if one_fluid:
        idust = 1
    else:
        idust = 8
    igas = 1
    
    # find minimum and maximum values of size and St for binning
    stmin = np.min(data[data.type == idust].st)
    stmax = np.max(data[data.type == idust].st)
    smin = np.min(data[data.type == idust].grainsize)
    smax = np.max(data[data.type == idust].grainsize)
    
    # radii binning
    if logr:
        rindex, bins_r = pd.cut(data.r, np.logspace(np.log10(rin), np.log10(rout), num=rbins), retbins=True)
    else:
        rindex, bins_r = pd.cut(data.r, rbins, retbins=True)
        
    # z, s and St indexes for binning
    zindex                = pd.cut(data.z, np.linspace(-vazmin, vazmin, num=zbins)) # linear
    gsindex, bins_gsize   = pd.cut(data.grainsize, np.logspace(np.log10(smin), np.log10(smax), num=sbins), retbins=True, duplicates="drop")
    stindex, bins_st      = pd.cut(data.st, np.logspace(np.log10(stmin), np.log10(stmax), num=sbins), retbins=True) # log
    
    # add the indexes to the DataFrame as a new column
    data.loc[:,"rindex"]  = rindex
    data.loc[:,"zindex"]  = zindex
    data.loc[:,"gsindex"] = gsindex
    data.loc[:,"stindex"] = stindex

    # gas and dust specific dataframes with masks
    dust = data[data.type==idust]
    gas  = data[data.type==igas]

    # binning r, z, size and St - count particles per bins of St and size
    bin_r       = gas.groupby("rindex").r.mean().values # always bin spatial dim. with gas - more extended than dust
    bin_z       = gas.groupby("zindex").z.mean().values # same
    bin_gsize   = dust.groupby("gsindex").grainsize.mean().values
    bin_st      = dust.groupby("stindex").st.mean().values
    count_st    = dust.groupby("stindex").st.count().values
    count_gsize = dust.groupby("gsindex").grainsize.count().values
    
    # dust and gas surface densities
    bin_dust_mass = dust.groupby("rindex").m.sum().values
    bin_gas_mass  = gas.groupby("rindex").m.sum().values
    if one_fluid:
        bin_dustfrac  = gas.groupby("rindex").dustfrac.mean().values
    bins_r        = np.pi*bins_r*bins_r
    s_annuli      = np.diff(bins_r)
    
    # adapt density evluation with one fluid
    if one_fluid:
        sigmad        = bin_dust_mass * bin_dustfrac / s_annuli / (udist**2)
        sigmag        = bin_gas_mass * (1-bin_dustfrac) / s_annuli / (udist**2)
    else:
        sigmad        = bin_dust_mass / s_annuli / (udist**2)
        sigmag        = bin_gas_mass / s_annuli / (udist**2)
    
    # grainsize, stokes number, velocities and densities - radii
    pro_size_r    = dust.groupby("rindex").grainsize.mean().values
    pro_st_r      = dust.groupby("rindex").st.mean().values
    if one_fluid:
        pro_rhod_r    = dust.groupby("rindex").rho.mean().values * bin_dustfrac
        pro_rhog_r    = gas.groupby("rindex").rho.mean().values * (1-bin_dustfrac)
    else:
        pro_rhod_r    = dust.groupby("rindex").rho.mean().values
        pro_rhog_r    = gas.groupby("rindex").rho.mean().values
    pro_rhog_d_r  = dust.groupby("rindex").rhogas.mean().values
    pro_vrelvf_r  = dust.groupby("rindex").vrelvf.mean().values
    pro_cs_r      = dust.groupby("rindex").cs.mean().values
    pro_vdr_r     = dust.groupby("rindex").vr.mean().values
    pro_vgr_r     = gas.groupby("rindex").vr.mean().values
    
    # hgas and hdust - radii
    Hgas  = gas.groupby("rindex").z.std().values
    Hdust = dust.groupby("rindex").z.std().values # doesn't work with one fluid
    if one_fluid:
        hdust = dust.groupby("rindex").h.mean().values * 1/bin_dustfrac
        hgas  = gas.groupby("rindex").h.mean().values * 1/(1-bin_dustfrac)
    else:
        hdust = dust.groupby("rindex").h.mean().values
        hgas  = gas.groupby("rindex").h.mean().values
    
    # same - altitude
    if one_fluid:
        pro_dustfrac_z = gas.groupby("zindex").dustfrac.mean().values
    pro_size_z     = dust.groupby("zindex").grainsize.mean().values
    pro_st_z       = dust.groupby("zindex").st.mean().values
    if one_fluid:
        pro_rhod_z     = dust.groupby("zindex").rho.mean().values * pro_dustfrac_z
        pro_rhog_z     = gas.groupby("zindex").rho.mean().values * (1-pro_dustfrac_z)
    else:
        pro_rhod_z     = dust.groupby("zindex").rho.mean().values
        pro_rhog_z     = gas.groupby("zindex").rho.mean().values
    pro_vrelvf_z   = dust.groupby("zindex").vrelvf.mean().values
    
    # same - size and St
    if one_fluid:
        pro_dustfrac_s  = gas.groupby("gsindex").dustfrac.mean().values
        pro_dustfrac_st = gas.groupby("stindex").dustfrac.mean().values
        pro_rhod_s      = gas.groupby("gsindex").rho.mean().values * pro_dustfrac_s
        pro_rhod_st     = gas.groupby("stindex").rho.mean().values * pro_dustfrac_st
        pro_hd_s        = gas.groupby("gsindex").h.mean().values * 1/pro_dustfrac_s
        pro_hd_st       = gas.groupby("stindex").h.mean().values * 1/pro_dustfrac_st
    else:
        pro_rhod_s      = dust.groupby("gsindex").rho.mean().values
        pro_rhod_st     = dust.groupby("stindex").rho.mean().values
        pro_hd_s        = dust.groupby("gsindex").h.mean().values
        pro_hd_st       = dust.groupby("stindex").h.mean().values
    pro_vrelvf_s   = dust.groupby("gsindex").vrelvf.mean().values
    pro_vrelvf_st  = dust.groupby("stindex").vrelvf.mean().values
    pro_Hd_st      = dust.groupby("stindex").z.std().values
    pro_Hd_s       = dust.groupby("gsindex").z.std().values
    pro_vdr_st     = dust.groupby("stindex").vr.mean().values
    pro_vdr_s      = dust.groupby("gsindex").vr.mean().values
    
    # profiles dicts
    dict_r    = {"r": bin_r, "vgr": pro_vgr_r, "vdr": pro_vdr_r, "Hg": Hgas, "Hd": Hdust, "hg": hgas, "hd": hdust, "sigmag": sigmag, "sigmad": sigmad, "rhog": pro_rhog_r, "rhog_d": pro_rhog_d_r, "rhod": pro_rhod_r, "gsize": pro_size_r, "vrelvfrag": pro_vrelvf_r, "st": pro_st_r, "cs": pro_cs_r}
    dict_z    = {"z": bin_z, "rhog": pro_rhog_z, "rhod": pro_rhod_z, "gsize": pro_size_z, "vrelvfrag": pro_vrelvf_z, "st": pro_st_z}
    dict_size = {"gsize": bin_gsize, "npart": count_gsize, "vdr": pro_vdr_s, "hd": pro_hd_s, "rhod": pro_rhod_s, "vrelvfrag": pro_vrelvf_s, "Hd": pro_Hd_s}
    dict_st   = {"st": bin_st, "npart": count_st, "vdr": pro_vdr_st, "hd": pro_hd_st, "rhod": pro_rhod_st,"vrelvfrag": pro_vrelvf_st, "Hd": pro_Hd_st}
    
    # convert to DataFrames
    profiles_r    = pd.DataFrame(data=dict_r)
    profiles_z    = pd.DataFrame(data=dict_z)
    profiles_size = pd.DataFrame(data=dict_size)
    profiles_st   = pd.DataFrame(data=dict_st)
    
    return time, profiles_r, profiles_z, profiles_size, profiles_st
