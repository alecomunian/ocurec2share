#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:This file:

    `run_deesse.py`

:Purpose:

    Create a simple workflow where:
        1. Data are read from an input file
        2. Some data are removed
        3. DeeSse reconstruction is performed
        4. Some comparison parameters are finally computed

:Usage:

    Run it as a "standard" Python script

:Parameters:

:Version:

    * 2025/10/21

:Authors:

    Alessandro Comunian

.. notes::
    - The dimension of the cells should not be a problem, (for example, the fact that along 
      *z* axis the time dimension is 1 hour and along *x* and *y* the size is about 1000  m.
    - No normalization of the input data required. All should be automatically done
      by geone.

.. warning::
    - Some trick is used to transform the sparse point grids into a structured
      grid that can be easily used as TI. However, for domains that cover
      a wider lat/lon range, some adjustment might be needed.

.. limitations::
    
.. license::
    This code is distributed under the
 
        GNU GENERAL PUBLIC LICENSE
        Version 3, 29 June 2007
        
    The complete version of the license is provided inside the same
    github repository.
                       
"""

import tomli 
import os
import numpy as np
import geone as gn
import copy
import time
import logging
import pathlib
import sys
# ADAPT HERE ACCORDING TO YOUR FOLDER SETTINGS
sys.path.append("/home/alex/workspace/dev/ocurec/src")
import ocurec

# Intial setting for the log file
logger = logging.getLogger(__name__)
# Name of this script
this_file = pathlib.Path(__file__).stem

# Define the working directory and fig and out directories
# (the working directory root is where the script is located)
wdir = os.path.dirname(__file__)
log_file = os.path.join(wdir, '{0}.log'.format(this_file))
logging.basicConfig(filename=log_file, level=logging.INFO,
                    force=True, filemode="w")
logger.info('Started')

# Create out and fig folders
out_dir = os.path.join(wdir, 'out')
if not os.path.exists(out_dir):
      # If not, create the folder.
      os.makedirs(out_dir)
fig_dir = os.path.join(wdir, 'fig')
if not os.path.exists(fig_dir):
    # If not, create the folder.
    os.makedirs(fig_dir)

# Read the input parameters
with open("par_deesse.toml", mode="rb") as fp:
    par = tomli.load(fp)

# Select day and hour of the day to be used as training
t_ini = (par["time"]["DD_ini"]-1)*24+par["time"]["HH_ini"]
t_fin = (par["time"]["DD_ref"]-1)*24+par["time"]["HH_ref"]

# Set up some frequently used parameters
nx = par["grid"]["nx"] 
ny = par["grid"]["ny"]
# Determine the 3D time component of the TI
nt = t_fin-t_ini


# Read the input file and compute the total number of points where data
# is recorded (computed).
mat, nb_points = ocurec.mat_load(par)

# Print out some statistics about the input data
ocurec.mat_stats(mat)

logger.info("Number of points where (lon,lat) are defined: {0}".format(nb_points))
logger.info("Total number of time steps considered: {0}".format(nt))
logger.info("Data start date     : {0}".format(ocurec.par2date(par,0)))
logger.info("Data stop date (ref): {0}".format(ocurec.par2date(par)))


# Create the MASK object
# (MASK is related to the domain when all measurements locations are defined)

# Compute a mask to restrict the simulation domain in a structured (DS) grid
img_mask = ocurec.mat2mask(mat, nx, ny, nt)

# Plot the mask (2D only)
ocurec.plot_mask2D(img_mask, fig_dir)

# Select day and hour of the day to be reconstructed
# (and also used as a reference)
sel_t = (par["time"]["DD_ref"]*24-1)+par["time"]["HH_ref"]

# %% Plot the reference time step


# Create and empty container for the TI containing the velocities
vxvy = np.nan*np.ones((2,nt,ny,nx))

# Create at TI object
img_ref = ocurec.mat2ti(mat, par, nx, ny, nt, name="reference")

# Compute speed from the two velocity components in a training image
speed = ocurec.img_vel2speed(img_ref)

# Compute the total number of uninformed nodes (inside the simulation grid)
# in the training data set (check only vx)
# NOTE: EXCLUDED THE REFERENCE TIME STEP
tot_nan =  np.sum(np.isnan(img_ref.val[0,:-1,:,:])) - np.sum(img_mask.val[0,:-1,:,:]==0)
# Total number of nodes that could be filled (460*number of time steps)
tot_grid = np.sum(img_mask.val[0,:-1,:,:]==1)
logger.info("# of total uninformed nodes: {0:d}".format(tot_nan))
logger.info("# of total of nodes: {0:d}".format(tot_grid))
logger.info("fraction of uninformed nodes: {0:.3f}".format(tot_nan/tot_grid))

# Plot the velocity field (only the "reference" one)
file_out = os.path.join(wdir, "fig", "velocity_ref.png")
ocurec.plot_vel(img_ref, img_mask, file_out=file_out, t=-1, par=par)

# Total number of locations where (a full dataset) has data
mask_size = np.sum(img_mask.val[0,-1, :,:])
# Number of nodes to be removed
nb_tbr = int(par["test"]["frac"]*mask_size)


# %% Create an incomplete TI
img_inc = copy.deepcopy(img_ref)
img_inc.name = "incomplete"
rem = 0
for i in range(nx):
    for j in range(ny):
        if img_mask.val[0,-1,j,i]:
            img_inc.val[0, -1, j, i] = np.nan
            img_inc.val[1, -1, j, i] = np.nan
            rem = rem + 1
            if rem >= nb_tbr:
                break
    if rem >= nb_tbr:
        break

# Store the mask of the missing values
# (this also includes the borders ouside the simulation mask)
mask_inc = np.isnan(img_inc.val[0,-1,:,:])

# Check if some numbers are OK
# nb_maskin = np.sum(img_mask.val == 1)
# nb_maskout = np.sum(img_mask.val == 0)

# nb_maskout_ti = np.sum(img_inc.val[0,:,:,:] == np.nan)
# nb_inc = np.sum(np.isnan(img_inc.val[0,:,:,:]))


# Plot the INCOMPLETE velocity field
file_out = os.path.join(wdir, "fig", "velocity_inc.png")
ocurec.plot_vel(img_inc, img_mask, file_out=file_out, t=-1, par=par, mode="inc")

nx, ny, nz = img_ref.nx, img_ref.ny, img_ref.nz
# TODO: CHECK IF SETTING THE DEFAULT VALUE TO Z CHANGES SOMETHING
sx, sy, sz = img_ref.sx, img_ref.sy, img_ref.sz
ox, oy, oz = img_ref.ox, img_ref.oy, img_ref.oz

#
# CONTROLARE SE MIGLIORA QUALCOSA, MA IN TEORIA TUTTO DOVREBBE ESSERE
# NORMALIZZATO ALL'INTERNO DEL CODICE!
#



##############################################################
# APPLY DEESSE
##############################################################
deesse_input = gn.deesseinterface.DeesseInput(
    nx=nx, ny=ny, nz=nz, # set same dimensions grid as for the incomplete image
    sx=sx, sy=sy, sz=sz, # set same cell units grid as for the incomplete image
    ox=ox, oy=oy, oz=oz, # set same origin grid as for the incomplete image
    nv=2, varname=img_inc.varname,     # number of variable(s), name of the variable(s), 
                                  #    as for the incomplete image
    TI=img_inc,                        # set the incomplete image as TI
    dataImage=img_inc,                 # set the incomplete image as hard data
    distanceType=['continuous','continuous'],
    nneighboringNode=par["DS"]["nneighboringNode"],
    distanceThreshold=par["DS"]["distanceThreshold"],
    maxScanFraction=1.0,
    npostProcessingPathMax=1,
    simType = par["DS"]["simType"],
    simPathType = par["DS"]["simPathType"],
    simPathStrength = par["DS"]["simPathStrength"],
    #simPathStrength=0.5,
    # simPathType="unilateral",
    mask = img_mask.val,
    seed=444,
    nrealization=par["DS"]["real"])

# Run deesse
t1 = time.time() # start time
deesse_output = gn.deesseinterface.deesseRun(deesse_input,
                                             verbose=par["DS"]["verbose"],
                                             nthreads=par["DS"]["nthreads"])
t2 = time.time() # end time
logger.info(f'DS simulation time: {t2-t1:.2g} sec')



# %%
sim = deesse_output['sim']

# By default, select the 1st realization for the plotting
img_out = deesse_output['sim'][0]

file_out = os.path.join(wdir, "fig", "velocity_cmp")
# Plot a comparison of the velocities
ocurec.plot_vel_cmp(img_ref, img_inc, img_out, file_out=file_out,
             t=-1, par=par)


# Print out some statistics
outfile = os.path.join(par["folders"]["out"], "statistics.csv")

# Compare and plot comparison results
ocurec.compare_many(img_ref, img_inc, sim, outfile=outfile,
                    plotfile=par["folders"]["fig"], z=-1)
logger.info('Stopped')







