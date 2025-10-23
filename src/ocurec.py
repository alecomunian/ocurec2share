#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:This file:

    `ocurec.py`

:Purpose:

    Collection of functions useful to support the reconstruction of incomplete
    HF coastal radar data.

:Usage:

    Explain here how to use it.

:Parameters:

:Version:

    Thu Jun 13 14:29:23 2024

:Authors:

    Alessandro Comunian

.. notes::
    - As a general rule, `vx` and `vy` are used in place of `u` and `v` to
      be easily found in the text.
      

.. warning::

.. limitations::
    
.. license::
    This code is distributed under the
 
        GNU GENERAL PUBLIC LICENSE
        Version 3, 29 June 2007
        
    The complete version of the license is provided inside the same
    github repository.
                       
            
"""
# Size of one velocity map
FIGSIZE_VELS = ((16,10))

# Size of the two scatter plots on the velocity
FIGSIZE_2SCAT = ((12,7))

# Speed variability
VMIN = 0.0 # [cm/s]
VMAX = 50.0 # [cm/s]
# Size of the dot to be plotted where velocity is reconstructed
VCSIZE = 20

QSCALE = 300
QWIDTH = 0.003

import matplotlib.pylab as pl
import matplotlib.patches as mp
import matplotlib as ml
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import scipy.stats as ss
import pandas as pd
import copy

import geone as gn
import scipy.io as si

import logging
import utm
import datetime
import minisom

# Turn off interactive plotting
pl.ioff() 

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
def plot_vel(img_vel, img_mask=None, file_out=None, t=-1, par=None, mode=None):
    """
    Plot a velocity field contained in a bi-dimensional TI of type
    `geone.img.Img`.
    When a `img_mask` is provided, it is considered as domain where to
    restrict the plotting.

    Parameters
    ----------
    img_vel : geone.img.Img
        Incomplete or default velocity field to be plotted.
    img_mask : geone.img.Img, optional
        If a mask is provided, then it can be used.
        Otherwise, the mask is computed automatically where there are some 
        values equal to NaN (or -9999999). The default is None.
    file_out : string, optional
        If provided, an output figure with this file name is saved.
        The default is None.
    t : int, optional
        In this case study, the *z* coordinate of the TIs is used for storing
        all the time steps. This parameter allows to select which time step
        should be plotted. The default is -1, because it is in general expected
        that the reference time steps to be reconstructed is the last one in 
        the image.
    par : dict
        A dictionary containing the parameters used to run the script,
        possibily useful for printing the image title.

    Returns
    -------
    When defined, it saves a PNG figure with the plot.

    """
  
    # Unpack the components
    vx = img_vel.val[0,t,:,:]
    vy = img_vel.val[1,t,:,:]
    # Compute the module
    speed = np.hypot(vy, vx)
    
    # Get the grid properties
    nx, ny = img_vel.nx, img_vel.ny
    ox, oy = img_vel.ox, img_vel.oy
    sx, sy = img_vel.sx, img_vel.sy
    
    # Compute the grid containing the points.    
    # Note that (ox,oy) is the lower left corner of the 1st cell.
    east4plot = np.arange(ox+0.5*sx, ox+sx*nx, sx)
    nort4plot = np.arange(oy+0.5*sy, oy+sy*ny, sy)
    # east_mg, nort_mg = np.meshgrid(east4plot, nort4plot, indexing="ij")
    east_mg, nort_mg = np.meshgrid(east4plot, nort4plot, indexing="xy")
    
    fig, ax = pl.subplots(1, 1, num="Velocity field", figsize=FIGSIZE_VELS)
    
    if img_mask is not None:
        # A mask is provided
        mask4plot = (img_mask.val[0,t,:,:] == 1) # .astype(bool)
        # These are the missing values
        mask4miss = np.logical_and(mask4plot, np.isnan(vx))
    else:
        # Extract the mask from the TI
        # TODO: Check if here setting a variable invalid value can be more
        #       appropriate.
        # mask4plot = (vx != -9999999)
        mask4plot = (~np.isnan(vx))
        # These are the missing values
        mask4miss = np.isnan(vx)
    
    # Plot an arrow where a value is defined.
    qui = ax.quiver(east_mg[mask4plot], nort_mg[mask4plot],
              vx[mask4plot],
              vy[mask4plot],
              # vx[mask4plot]/speed[mask4plot],
              # vy[mask4plot]/speed[mask4plot],
              speed[mask4plot], clim=(VMIN,VMAX), scale=QSCALE, width=QWIDTH)
    # Plot a red dot where values are missing or outiside the grid.
    ax.scatter(east_mg[mask4miss], nort_mg[mask4miss], color="red", s=VCSIZE, alpha=0.5)
        
    if mode == "ref":
        mode_str = " - reference velocity field"
    elif mode == "inc":
        mode_str = " - incomplete velocity field"
    else:
        mode_str = ""
    if par is not None:
        ax.set_title(par2date(par) + mode_str)
       
    ax.axis("equal")
    ax.set_xlabel("easting [m]")
    ax.set_ylabel("northing [m]")
    pl.colorbar(qui)
    pl.tight_layout()
    pl.savefig(file_out, dpi=400)
    pl.close(fig)
    
    
def par2date(par, t=-1):
    """
    Extract from the input parameters the date to be printed, depending on
    the value of the time step provided. If no time step is provided, then
    the refence value is printed, that is the last time step.

    Parameters
    ----------
    par : dict
        Contains information about the starting and stopping date of the data
        set. See the attached TOMLI file for more info about its structure.
    t : int, optional
        The time step that should be plotted. The default is -1, that is the
        final reference time step is printed.

    Returns
    -------
    A formatted string with the date, computed based on the provided input.

    """
   
    t_ini = datetime.datetime(par["time"]["YY_ini"],
                              par["time"]["MM_ini"],
                              par["time"]["DD_ini"],
                              par["time"]["HH_ini"])
    t_ref = datetime.datetime(par["time"]["YY_ref"],
                              par["time"]["MM_ref"],
                              par["time"]["DD_ref"],
                              par["time"]["HH_ref"])
    
    if t>=0:
        t_out = t_ini + datetime.timedelta(hours=t)
    elif t<0:
        t_out = t_ref + datetime.timedelta(hours=t+1)
            
    return t_out.strftime("%Y-%m-%d %H:%M")    
    
        
    
    
def plot_vel_cmp(img_ref, img_inc, img_out, file_out=None,
             t=-1, par=None):
    """
    Plot a velocity field contained in a bi-dimensional TI of type
    `geone.img.Img` for a given time step.
    If an image for comparison is provided, plot it ovelapping to
    the location where there are some missing values.

    Parameters
    ----------
    img_vel : geone.img.Img
        Incomplete or default velocity field to be plotted.
    img_mask : geone.img.Img, optional
        If a mask is provided, then it can be used.
        Otherwise, the mask is computed automaticall where there are some 
        values equal to NaN (or -9999999). The default is None.
    file_out : string, optional
        If provided, an output figure with this file name is saved.
        The default is None.
    im_comp : geone.img.Img, optional
        If provided, the values that are missing for `img_vel` but defined in
        this image are plotted and eventually overlapped. The default is None.
    t : int, optional
        In this case study, the *z* coordinate of the TIs is used for storing
        all the time steps. This parameter allows to select which time step
        should be plotted. The default is -1, because it is in general expected
        that the reference time steps to be reconstructed is the last one in 
        the image.

    Returns
    -------
    When defined, it saves a PNG figure with the plot.

    """
  
    # Unpack the components
    vx = img_ref.val[0,t,:,:]
    vy = img_ref.val[1,t,:,:]
    
    vx_out = img_out.val[0,t,:,:]
    vy_out = img_out.val[1,t,:,:]
    
    # Compute the module
    speed = np.hypot(vy, vx)
    speed_out = np.hypot(vy_out, vx_out)
    
    # Get the grid properties
    nx, ny = img_ref.nx, img_ref.ny
    ox, oy = img_ref.ox, img_ref.oy
    sx, sy = img_ref.sx, img_ref.sy
    
    # Compute the grid containing the points.    
    # Note that (ox,oy) is the lower left corner of the 1st cell.
    east4plot = np.arange(ox+0.5*sx, ox+sx*nx, sx)
    nort4plot = np.arange(oy+0.5*sy, oy+sy*ny, sy)
    # east_mg, nort_mg = np.meshgrid(east4plot, nort4plot, indexing="ij")
    east_mg, nort_mg = np.meshgrid(east4plot, nort4plot, indexing="xy")
    
    fig, ax = pl.subplots(1, 1, num="Velocity field - comparison", figsize=FIGSIZE_VELS)
    

    
    # Here the locations where one measurement should be available
    mask_sim = ~np.isnan(img_out.val[0,t,:,:])
    
    # Here the points where the grid was incomplete
    mask_inc = np.logical_and(np.isnan(img_inc.val[0,t,:,:]), mask_sim)
    

    # Plot an arrow where a value is defined.
    qui1 = ax.quiver(east_mg[mask_sim], nort_mg[mask_sim],
              vx[mask_sim],
              vy[mask_sim],
              # vx[mask4plot]/speed[mask4plot],
              # vy[mask4plot]/speed[mask4plot],
              speed[mask_sim], clim=(VMIN,VMAX), scale=QSCALE, width=QWIDTH)#, pivot="mid")
    
    # Plot an arrow where a value is defined.
    qui2 = ax.quiver(east_mg[mask_inc], nort_mg[mask_inc],
              vx_out[mask_inc],
              vy_out[mask_inc],
              # vx[mask4plot]/speed[mask4plot],
              # vy[mask4plot]/speed[mask4plot],
              speed_out[mask_inc], clim=(VMIN,VMAX), scale=QSCALE, width=QWIDTH)#, pivot="mid")
    
    # Plot a red dot where values are missing in the incomplete grid
    ax.scatter(east_mg[mask_inc], nort_mg[mask_inc], color="red", s=VCSIZE, alpha=0.5)
       
    ax.axis("equal")
    ax.set_xlabel("easting [m]")
    ax.set_ylabel("northing [m]")
    ax.set_title(par2date(par)+" - reference VS reconstruction")
    pl.colorbar(qui1)
    pl.tight_layout()
    pl.savefig(file_out, dpi=400)
    pl.close(fig)    

    
def test_plot_vel():
    print("")
    print("    *** START - testing `plot_vel` ***")
    # First create a "fake" velocity image
    nx, ny, nz = 5, 4, 3 # number of cells along each axis
    
    val01 = np.ones((2, nz, ny, nx))
    val01[0, -1, 2, 3] = 3
    val01[1, -1, 2, 3] = 2    
    
    im01 = gn.img.Img(nx=nx, ny=ny, nz=nz,
                # sx=1.0, sy=1.0, sz=1.0, # default values
                # ox=1.0, oy=1.0, oz=1.0, # default values
                nv=2, varname=["vx", "vy"], val=val01,
                name="Test velocity image")
    print(im01)
    
    #
    # Uniform velocity field
    #
    plot_vel(im01, file_out="test_plot_vel01.png")
    
    
    #
    # Check when an incomplete image is provided without mask
    #
    val02 = copy.copy(val01)
    val02[:,:,:2,:3] = np.nan
    
    im02 = gn.img.Img(nx=nx, ny=ny, nz=nz,
                # sx=1.0, sy=1.0, sz=1.0, # default values
                # ox=1.0, oy=1.0, oz=1.0, # default values
                nv=2, varname=["vx", "vy"], val=val02,
                name="Test velocity image")
    
    plot_vel(im02, file_out="test_plot_vel02.png")
    
    #
    # Check when an incomplete image is provided with a mask
    #
    val03 = copy.copy(val01)
    val03[:,:,:2,:3] = np.nan
    
    im03 = gn.img.Img(nx=nx, ny=ny, nz=nz,
                # sx=1.0, sy=1.0, sz=1.0, # default values
                # ox=1.0, oy=1.0, oz=1.0, # default values
                nv=2, varname=["vx", "vy"], val=val03,
                name="Test velocity image")
    
    mask03 =np.ones((1, nz, ny, nx))
    mask03[:,:,:2,:2] = 0
    
    img_mask03 = gn.img.Img(nx=nx, ny=ny, nz=nz,
                # sx=1.0, sy=1.0, sz=1.0, # default values
                # ox=1.0, oy=1.0, oz=1.0, # default values
                nv=1, varname=["mask"], val=mask03,
                name="Mask test vel 03")
    
    plot_vel(im03, img_mask=img_mask03, file_out="test_plot_vel03.png")
    
    
    
    #
    # Check when an incomplete reference is provided with a mask and simulation
    #
    val04 = np.sqrt(2)*np.ones((2, nz, ny, nx))
    val04[1,:,:,:] = 0
    # val03[:,:,:2,:2] = np.nan
    
    im04 = gn.img.Img(nx=nx, ny=ny, nz=nz,
                # sx=1.0, sy=1.0, sz=1.0, # default values
                # ox=1.0, oy=1.0, oz=1.0, # default values
                nv=2, varname=["vx", "vy"], val=val04,
                name="Test velocity image")
    
    plot_vel(im03, img_mask=img_mask03, file_out="test_plot_vel04.png",
             im_comp=im04)
    
    
    # # 
    # # Check when a mask is provided
    # #
    # val = np.ones((1, nz, ny, nx))
    # val[0,:,0,:2] = 0
    
    # mask = gn.img.Img(nx=nx, ny=ny, nz=nz,
    #             # sx=1.0, sy=1.0, sz=1.0, # default values
    #             # ox=1.0, oy=1.0, oz=1.0, # default values
    #             nv=1, varname=["mask"], val=val,
    #             name="Mask")
    
    # plot_vel(im, mask, file_out="test_plot_vel03.png")
    
    # #
    # #
    # #
    # im.val[:,-1,1,2] = np.nan
    
    # plot_vel(im, mask, file_out="test_plot_vel04.png")
    
    
    # plot_vel(img_vel, par, img_mask=None, file_out=None, im_comp=None, t=-1)
    print("    *** STOP - testing `plot_vel` ***")
    print("")

    
def compare(im_ref, im_inc, im_sim, verbose=False, plot=False, z=-1):
    """
    Compare the reference againts one simulation. Only the reconstruted part
    of the image is considered.

    Parameters
    ----------
    im_ref : TYPE
        DESCRIPTION.
    im_sim : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    Notes
    -----
    - For computing the simulation mask, NaN values of the last variable of the
      last time step of the input `im_sim` are considered.
      This should be in general OK.
    
    """
    
    # The simulation should cover the entire simulation domain, unless a simulation
    # mask is defined
    mask_sim = ~np.isnan(im_sim.val[-1,-1,:,:])
    nb_nan_sim = np.sum(~mask_sim)
    
    # The incomplete image should contain NaN where data are incomplete and also
    # NaN if there is a simulation mask
    mask_inc = ~np.isnan(im_inc.val[-1,-1,:,:])
    nb_nan_inc = np.sum(~mask_inc)
    
    # The reference image should contain NaN only where there is a simulation 
    # mask. If it also contains NaN inside the simulation mask, then raise
    # a Warning message
    mask_ref = ~np.isnan(im_ref.val[-1,-1,:,:])
    nb_nan_ref = np.sum(~mask_ref)
    
   
    logger.info("# of NaN locations in the reference field: {0:d}".format(nb_nan_ref-nb_nan_sim))
    logger.info("# of NaN locations in the incomplete field: {0:d}".format(nb_nan_inc-nb_nan_sim))
    if nb_nan_ref > nb_nan_sim:
        logger.warning("# of NaN in the reference >"
                       " NaN in simulation grid ({0:d}>{1:d})".format(nb_nan_ref, nb_nan_sim))
        # The comparison makes sense only where there are reference values
        # mask_cmp = mask_ref
    # else:
    # pl.close(fig)
    # Data should be compared only in locations where some reconstruction is 
    # made.    
    mask_cmp = np.logical_and(mask_ref, mask_ref != mask_inc)
    
    # fig, ax = pl.subplots(1,1)
    # pl.imshow(mask_inc)
    # pl.show()
    
    
    
    ref_x = im_ref.val[0,z,:,:][mask_cmp]
    ref_y = im_ref.val[1,z,:,:][mask_cmp]
    
    # print(mask)
    # print(ref_x)
    # nb_nan_ref_x = np.sum(np.isnan(ref_x))
    # nb_nan_ref_y = np.sum(np.isnan(ref_y))
    # print(nb_nan_ref_x)
    # print(nb_nan_ref_y)
    # Flag if there are nan inside the mask
    # nan_in_mask = nb_nan_ref_x>0 or nb_nan_ref_y>0
    
    # # print(nan_in_mask)
    # if nan_in_mask:
    #     logger.warning("Some nodes in the reference image"
    #                    " are NaN ({0:d},{1:d})".format(
    #                        nb_nan_ref_x, nb_nan_ref_y))
    #     mask_ref_nan = np.logical_or(np.isnan(ref_x), np.isnan(ref_y))
    #     ref_x = ref_x[np.logical_not(mask_ref_nan)]
    #     ref_y = ref_y[np.logical_not(mask_ref_nan)]
    
    ref = np.hypot(ref_x, ref_y)
    # print(ref_x)
    
    sim_x = im_sim.val[0,z,:,:][mask_cmp]
    sim_y = im_sim.val[1,z,:,:][mask_cmp]
    # if nan_in_mask:
    #     sim_x = sim_x[np.logical_not(mask_ref_nan)]
    #     sim_y = sim_y[np.logical_not(mask_ref_nan)]
    sim = np.hypot(sim_x, sim_y)

    # Pearson coefficient
    # print("ciao", ref_x.shape, sim_x.shape)
    r_vx = ss.pearsonr(ref_x, sim_x).statistic
    r_vy = ss.pearsonr(ref_y, sim_y).statistic
    r_mod = ss.pearsonr(ref, sim).statistic
    # print(ref, sim)
    
    # MAE
    mae_vx = np.sum(np.abs(ref_x-sim_x))/ref_x.size
    mae_vy = np.sum(np.abs(ref_y-sim_y))/ref_y.size
    mae_mod = np.sum(np.abs(ref-sim))/ref.size
    
    # RMSE
    rmse_vx = np.sqrt(np.sum((ref_x-sim_x)**2)/ref_x.size)
    rmse_vy = np.sqrt(np.sum((ref_y-sim_y)**2)/ref_y.size)
    rmse_mod = np.sqrt(np.sum((ref-sim)**2)/ref.size)
    
    # Circular correlation
    th_ref = np.arctan2(ref_y,ref_x)
    th_sim = np.arctan2(sim_y,sim_x)
    # n = ref_x.size
    # th_ref_ave = np.mean(th_ref)
    # th_sim_ave = np.mean(th_sim)
    # sin_ref = np.sin(th_ref-th_ref_ave)
    # sin_sim = np.sin(th_sim-th_sim_ave)
    # rcc = np.sum(sin_ref*sin_sim)/(
    #     np.sqrt(np.sum(sin_ref**2)/n)*
    #     np.sqrt(np.sum(sin_sim**2)/n)
    #     )
    # compute correlation coeffcient from p. 176
    num = np.sum(np.sin(th_ref) * np.sin(th_sim))
    den = np.sqrt(np.sum(np.sin(th_ref) ** 2) *
              np.sum(np.sin(th_sim) ** 2))
    rcc =  num / den
    
    
    # print(th_ref)
    
    # print(mae_vx, ref_x.shape)
    if verbose:
        print("    Pearson correlation (vx)    : {0:.3f}".format( r_vx))
        print("    Pearson correlation (vy)    : {0:.3f}".format( r_vy))
        print("    Pearson correlation (module): {0:.3f}".format( r_mod))
        print("    Cirular correlation         : {0:.3f}".format( rcc))
        print("    MAE (vx):   {0:3.3f}".format( mae_vx))
        print("    MAE (vy):   {0:3.3f}".format( mae_vy))
        print("    MAE (mod):  {0:3.3f}".format( mae_mod))
        print("    RMSE (vx):  {0:3.3f}".format( rmse_vx))
        print("    RMSE (vy):  {0:3.3f}".format( rmse_vy))
        print("    RMSE (mod): {0:3.3f}".format( rmse_mod))
    
    if plot:
        fig, ax = pl.subplots(1,3, sharex=False, sharey=False, figsize=(FIGSIZE_2SCAT))
        # print(sim_x)
        allx = np.concatenate((sim_x, ref_x, sim_y, ref_y))
        speedall = np.concatenate((ref, sim))
        speedmin = np.nanmin(speedall)
        speedmax = np.nanmax(speedall)
        xmin = np.nanmin(allx)
        xmax = np.nanmax(allx)
        # logging.info("Range of variability *u* and *v*: ({0:.4f},{1:.4f})".format( xmin, xmax))
        # print("    Range of variability *u* and *v*: ({0:.4f},{1:.4f})".format( xmin, xmax))
        line = np.arange(xmin, xmax, 0.1)
        # print(xmin, xmax)
        # ax.axis("equal")
        ax[0].set_title("a) $v_x$ [cm/s], $r=${0:.4f}".format(r_vx))
        ax[0].set(aspect='equal')
        ax[0].scatter(ref_x, sim_x)
        ax[0].set_xlim(([0.9*xmin, 1.1*xmax]))
        ax[0].set_ylim(([0.9*xmin, 1.1*xmax]))
        ax[0].set_xlabel("reference")
        ax[0].set_ylabel("reconstructed")
        ax[0].plot(line, line, color="red")
        
        ax[1].set_title("b) $v_y$ [cm/s], $r=${0:.4f}".format(r_vy))
        ax[1].plot(line, line, color="red")
        ax[1].set_xlim(([0.9*xmin, 1.1*xmax]))
        ax[1].set_ylim(([0.9*xmin, 1.1*xmax]))
        ax[1].set(aspect='equal')
        ax[1].scatter(ref_y, sim_y)
        # ax[1].set_xlim(([1.1*xmin, 1.1*xmax]))
        # ax[1].set_ylim(([1.1*xmin, 1.1*xmax]))
        ax[1].set_xlabel("reference")
        ax[1].set_ylabel("reconstructed")
        # ax[1].set_ylabel("reconstructed $v_y$ [cm/s]")
        
        lines = np.arange(speedmin, speedmax, 0.1)
        ax[2].set_title("c) speed [cm/s], $r=${0:.4f}".format(r_mod))
        ax[2].plot(lines, lines, color="red")
        ax[2].set_xlim(([0.9*speedmin, 1.1*speedmax]))
        ax[2].set_ylim(([0.9*speedmin, 1.1*speedmax]))
        ax[2].set(aspect='equal')
        ax[2].scatter(ref, sim)
        # ax[1].set_xlim(([1.1*xmin, 1.1*xmax]))
        # ax[1].set_ylim(([1.1*xmin, 1.1*xmax]))
        ax[2].set_xlabel("reference")
        ax[2].set_ylabel("reconstructed")
        
        
        scatter_plot = os.path.join(plot, "scatterplot_real0.png")
        pl.tight_layout()
        pl.savefig(scatter_plot, dpi=400)
        pl.close(fig)
        
        # ax.set_ylim([xmin, xmax])
        # pl.show() #QUI CONTROLLARE CHE COSA SUCCEDE SE COMMENTO...
        
        #
        # PLOTTO IN CONFRONTO FRA LE VELOCITÀ
        #
        
    
    out = {}
    out["r_vx"] = r_vx
    out["r_vy"] = r_vy
    out["r_mod"] = r_mod
    out["rcc"] = rcc
    out["mae_vx"] = mae_vx
    out["mae_vy"] = mae_vy
    out["mae_mod"] = mae_mod
    out["rmse_vx"] = rmse_vx
    out["rmse_vy"] = rmse_vy
    out["rmse_mod"] = rmse_mod
    
    return out

def compare_many(im_ref, im_inc, im_sims, plotfile=None,
                 outfile=None, z=0):
    """
    Compare many realizations

    Parameters
    ----------
    im_ref : geone Img object type
        Reference data set.
    im_inc : geone Img object type
        Incomplete data set.
    im_sims : list of many (or one) geone Img object type
        Reconstruction results.
    mask : geone Img object type
        Mask that defines the location where the dataset is incomplete.
    plotfile : string, optional
        This is the file name where the comparison results should be plotted.
        The default is None.
    outfile : string, optional
        This is the file name where the comparison results in numbers should be
        saved as a CSV file. The default is None.
    z : integer, optional
        The realization that should be used for the comparison. The default is 0.

    Returns
    -------
    None.

    """
    
    # Total number of realizations
    nb_real = len(im_sims)
    # Pearson correlation for vx, vy and speed
    r_vx = np.zeros((nb_real))
    r_vy = np.zeros((nb_real))
    r_mod = np.zeros((nb_real))
    rcc = np.zeros((nb_real))
    # MAE
    mae_vx = np.zeros((nb_real))
    mae_vy = np.zeros((nb_real))
    mae_mod = np.zeros((nb_real))
    # RMSE
    rmse_vx = np.zeros((nb_real))
    rmse_vy = np.zeros((nb_real))
    rmse_mod = np.zeros((nb_real))

    # A dictionary to be used to print out a CSV DataFrame containing all the 
    # computed statistics
    xx = {}
    xx["r_vx"] = r_vx
    xx["r_vy"] = r_vy
    xx["r_mod"] = r_mod
    xx["rcc"] = rcc
    xx["mae_vx"] = mae_vx
    xx["mae_vy"] = mae_vy
    xx["mae_mod"] = mae_mod
    xx["rmse_vx"] = rmse_vx
    xx["rmse_vy"] = rmse_vy
    xx["rmse_mod"] = rmse_mod
    
    for i in range(nb_real):
        logger.info("*** Realizaion # {0}".format(i))
        if i==0:
            out = compare(im_ref, im_inc, im_sims[i], z=z, plot=plotfile)
        else:
            out = compare(im_ref, im_inc, im_sims[i], z=z)
            
        # print(out)
        r_vx[i] = out["r_vx"]
        r_vy[i] = out["r_vy"]
        r_mod[i] = out["r_mod"]
        rcc[i] = out["rcc"]
        mae_vx[i] = out["mae_vx"]
        mae_vy[i] = out["mae_vy"]
        mae_mod[i] = out["mae_mod"]
        rmse_vx[i] = out["rmse_vx"]
        rmse_vy[i] = out["rmse_vy"]
        rmse_mod[i] = out["rmse_mod"]

    
    if plotfile is not None:
        # out_dir= os.path.dirname(plotfile)
        if not os.path.exists(plotfile):
            # If not, create the folder.
            os.makedirs(plotfile)
            print('\n    Directory "{0}" successfully created.'.format(plotfile))
        fig, ax = pl.subplots(1,1, figsize=((9,7)))
        labels = ["$u$", "$v$", "speed"]
        ax.boxplot([mae_vx, mae_vy, mae_mod], labels=labels)
        ax.set_ylabel("MAE [cm/s]")
        box_corr = os.path.join(plotfile, "boxplot_mae.png")
        pl.savefig(box_corr, dpi=400)
        # pl.show()
        pl.close(fig)
        
        fig, ax = pl.subplots(1,1, figsize=((9,7)))
        ax.set_ylabel("Pearson $r$ [-]")
        labels = ["$u$", "$v$", "speed", "circular"]
        ax.boxplot([r_vx, r_vy, r_mod, rcc], labels=labels)
    
        box_corr = os.path.join(plotfile, "boxplot_corr.png")
        pl.savefig(box_corr, dpi=400)
        # pl.show()
        pl.close(fig)
    
    
    if outfile is not None:
        # Get the name of the outpuf folder and check if the folder exists
        # out_folder = os.path.split(outfile)[0]
        # # If not, create the folder.
        # os.makedirs(out_folder)
        # Save the output into a CSV file for later processing
        df = pd.DataFrame(xx)
        df.to_csv(outfile, index=False, float_format="%.6e")
    
def time_norm(ti):
    """
    Takes an input training image with two variables :math:`u` and :math:`v` in 
    a 3D space (with the :math:`z` coordinate as time coordinate), and normalize
    it along the time dimension.
    In practice, :math:`u` and :math:`v` are transformed as
        
    .. math::
        
        u(\mathbf{x}, t) = \\frac{u(\mathbf{x}, t)-\overline{u(\mathbf{x})}}{\sigma_{u}(\mathbf{x})};
        \\
        v(\mathbf{x}, t) = \\frac{v(\mathbf{x}, t)-\overline{v(\mathbf{x})}}{\sigma_{v}(\mathbf{x})}
        
        

    Parameters
    ----------
    ti : geone.img.Img
        3D training image (time along the *z* direction) to be "normalized".

    Returns
    -------
    The training image normalized.

    """
    ti_norm = copy.deepcopy(ti)
    nb_var = ti.val.shape[0]
    
    for i in range(nb_var):
        x = ti.val[i,:,:,:]
        ti_norm.val[i,:,:,:] = (x-np.nanmean(x, axis=0))/np.nanstd(x, axis=0)
        
    return ti_norm
        

def time_backnorm(ti_norm, ti):
    """
    Takes an input training image with two variables :math:`u` and :math:`v` in 
    a 3D space (with the :math:`z` coordinate as time coordinate), and normalize
    it along the time dimension.
    In practice, :math:`u` and :math:`v` are transformed as
        
    .. math::
        
        u'(\mathbf{x}, t) = \\frac{u(\mathbf{x}, t)-\overline{u(\mathbf{x})}}{\sigma_{u}(\mathbf{x})};
        \\
        v'(\mathbf{x}, t) = \\frac{v(\mathbf{x}, t)-\overline{v(\mathbf{x})}}{\sigma_{v}(\mathbf{x})}
        
        

    Parameters
    ----------
    ti : geone.img.Img
        3D training image (time along the *z* direction) to be "normalized".

    Returns
    -------
    The training image back from normalization.

    """
    ti_back = copy.deepcopy(ti_norm)
    nb_var = ti_norm.val.shape[0]
    
    for i in range(nb_var):
        x = ti_norm.val[i,:,:,:]
        y = ti.val[i,:,:,:]
        ti_back.val[i,:,:,:] = x*np.nanstd(y, axis=0) + np.nanmean(y, axis=0)
        
    return ti_back
        
        
# class MiniSomNaN(minisom.MiniSom):
#     """
#     """
    
#     def __init__(self, x, y, input_len, sigma=1, learning_rate=0.5,
#                 decay_function='asymptotic_decay',
#                 neighborhood_function='gaussian', topology='rectangular',
#                 activation_distance='euclidean', random_seed=None,
#                 sigma_decay_function='asymptotic_decay'):
#         super().__init__(x, y, input_len, sigma, learning_rate,
#                 decay_function, neighborhood_function, topology,
#                 activation_distance, random_seed, sigma_decay_function)
        
#     def distance(self, x, w):
#         print("plof")
#         mask = ~np.logical_or(np.isnan(x), np.isnan(w))
        
#         return np.linalg.norm(np.subtract(x[mask], w[mask]), axis=-1)

    



def data2ti(data, mat, img_mask):
    """
    
    MASK HAS 22X31 SHAPE
    
    DATA IS ONLY ONE LINE OF DATA

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    ii : TYPE
        DESCRIPTION.
    jj : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nb_data = int(data.shape[0]/2)
    vx = data[:nb_data]
    vy = data[nb_data:]
    
    nx, ny = img_mask.nx, img_mask.ny
    
    ii, jj, dx, dy, ox, oy = mat2ij(mat, nx, ny)
    vx_ti = np.nan*np.ones((1, ny, nx))
    vy_ti = np.nan*np.ones((1, ny, nx))
    vxvy = np.nan*np.ones((2, 1, ny, nx))
    
    vx_ti[0, jj, ii] = vx
    vy_ti[0, jj, ii] = vy
    
    vxvy[0,:,:,:] = vx_ti
    vxvy[1,:,:,:] = vy_ti 

    # Create at TI object
    im_vxvy = gn.img.Img(nx=nx, ny=ny, nz=1, sx=dx, sy=dy, ox=ox, oy=oy, nv=2,
                          varname=['vx', 'vy'], val=vxvy)
    
    return im_vxvy
    
    
def parse_log(file_log, engine):
    """
    Parse a typical log file and return some useful information, including for
    example the SOM quantization error and the training time (in seconds).

    Parameters
    ----------
    file_log : string
        Name of the input log file.

    Returns
    -------
    train_time : float
        Number of seconds required for the SOM training.
    QE : float
        Quantization Error.

    """
    if engine == "minisom":
        qe = np.nan
        te = np.nan
        tt = np.nan
        with open(file_log) as f:
            print('    Working on file {0}'.format(file_log))
            for line in f:
                cline = line.split(":")
                # print(cline[2].rstrip())
                if cline[2] == "QE":
                    qe = float(cline[3])
                elif cline[2] == "TE":
                    te = float(cline[3])
                elif cline[2] == "SOM Training time":
                    tt = float(cline[3].split(" ")[1])
                elif cline[2].rstrip() == "Some nodes in the reference image are NaN":
                    print("WARNING: Found NaN in the reference time step.")
                    tt = np.nan
                    qe = np.nan
                    te = np.nan
                    break
                # else:
                #     print("    ERROR, parameter not found")
        out = qe, tt, te
    elif engine == "deesse":
        with open(file_log) as f:
            print('    Working on file "{0}"'.format(file_log))
            for line in f:
                cline = line.split(":")
                if cline[2] == "DS simulation time":
                    tt = float(cline[3].split(" ")[1])
        out = tt
        
        
    return out
        
def plot_mask2D(img_mask, out_dir):
    """
    Plot a 2D mask that defines where velocity fields should be
    available/provided. When this mask is provided as 3D, it is supposed that
    it has no variability along the *z* (time) coordinate. Therefore, only
    the last *z* (time) step is plotted.

    Parameters
    ----------
    img_mask : geone.img.Img type object
        The mask object to be plotted.
    out_dir : string
        Default folder where the image should be saved.

    Returns
    -------
    Saves a figure containing the mask.

    """
    
    file_out = os.path.join(out_dir, "mask.png")
    fig, ax = pl.subplots(1,1,num="raw mask", figsize=FIGSIZE_VELS)
    pl.title("Domain mask")
    # gn.imgplot.drawImage2D(im_mask, categ=True)
    ax.imshow(img_mask.val[0,-1,:,:], origin="lower")
    ax.set_xlabel("easting [m]")
    ax.set_ylabel("northing [m]")
    pl.savefig(file_out, dpi=400)
    pl.close(fig)
    
    
def mat_load(par):
    """
    Get the number of points where latitute and longitude are defined in the 
    MAT file, that is the total number of points that should be always filled.

    Parameters
    ----------
    par : dictionary
        Input parameters dictionary, containing the location of the input MAT
        file. See the attached TOMLI file for more details about its structure.

    Returns
    -------
    The values contained in the MAT file and the (integer) number
    of points where a value of velocity should be defined.

    """
    mat = si.loadmat(os.path.join(par["folders"]["input"],
                                  par["files"]["dataset"]))

    # Read longitude
    lon = mat["hfr_total_march_2012"][0,0][3][:,0]
    #    lat = mat["hfr_total_march_2012"][0,0][3][:,1]    
    # TODO: Here we could also add some debugging stuff to check if the number
    # is the same for lon and lat.

    return mat, len(lon)

def mat_stats(mat):
    """
    Print out some statistical info about the info contained in the input
    MAT file.

    Parameters
    ----------
    mat : matrix
        Matrix extracted from a MAT file, containing the dataset.

    Returns
    -------
    Prints out some statistical information.
    The information is printed on a logger output file.

    """
    
    lon = mat["hfr_total_march_2012"][0,0][3][:,0]
    lat = mat["hfr_total_march_2012"][0,0][3][:,1]
    
    vx = mat["hfr_total_march_2012"][0,0][0][:,:]
    vy = mat["hfr_total_march_2012"][0,0][1][:,:]
    
    lon_len = len(lon)
    
    logger.info("*** mat statistics - START ***")
    logger.info("    Tot.number of locations: {0}".format(lon_len))
    logger.info("    Longitude range: ({0:.6f},{1:.6f})".format(lon.min(), lon.max()))
    logger.info("    Latitude range: ({0:.6f},{1:.6f})".format(lat.min(), lat.max()))
    logger.info("    vx range and mean [cm/s]:"
          "({0:.2f},{1:.2f}) {2:.2f}".format(
              np.nanmin(vx), np.nanmax(vx), np.nanmean(vx)))
    logger.info("    vy range and mean [cm/s]:"
          "({0:.2f},{1:.2f}) {2:.2f}".format(
              np.nanmin(vy), np.nanmax(vy), np.nanmean(vy)))
    logger.info("*** mat statistics - STOP ***")

    
    
def mat2ij(mat, nx, ny):
    # Read lon. and lat.
    lon = mat["hfr_total_march_2012"][0,0][3][:,0]
    lat = mat["hfr_total_march_2012"][0,0][3][:,1]


    # logger.info("Number of points where (lon,lat) are defined: {0:d}".format(nb_points))
    # logger.info("Total number of time steps considered: {0:d}".format(nt))

    # Create array to contain the UTM transformed coordinates
    east = np.empty(lon.shape)
    nort = np.empty(lat.shape)

    # Convert the coordinates in UTM
    for i in range(lon.shape[0]):
        east[i], nort[i] = utm.from_latlon(lat[i], lon[i], force_zone_number=33, force_zone_letter="T")[:2]

    # This is a little "tricky", add a small value to the delta...   
    dx = (east.max()-east.min())/nx + 1.0
    dy = (nort.max()-nort.min())/ny + 1.0
    # logging.info("Grid spacing [m] (E,N): ({0:.3f}, {1:.3f})".format(d_east, d_nort))

    # Convert the coordinate into index
    ii = ((east - east.min())/dx).astype(int)
    # WARNING: High values of jj correspond to high values of northing. Remember
    #          this when initializing matrix and geone.img.Img objects.
    jj = ((nort - nort.min())/dy).astype(int)
    
    # Origin is in the lower left corner of the 1st cell
    ox = east.min() - 0.5*dx
    oy = nort.min() - 0.5*dy
    
    return ii, jj, dx, dy, ox, oy
    
def mat2mask(mat, nx, ny, nt):
    """
    Get from the input matlab MAT file the shape of the mask and returns it
    in a structured
    grid setting like the one for the application of DeeSse.

    Parameters
    ----------
    par : dictionary
        All the information required and contained in the parameters input file.
        See the attached TOMLI file for more information about its structure.
    nx : int
        number of cells along *x*.
    ny : int
        number of cells along *y*.
    nt : int
        number of time steps.

    Returns
    -------
    img_mask : geone.img.Img object
        Object containing the mask: 1 where the simulation should be performed,
        0 where there is no need to simulate.
    """
    

    ii, jj, dx, dy, ox, oy = mat2ij(mat, nx, ny)
    
    # The mask is by default 0 (False)
    mask = np.zeros((nt, ny, nx), dtype=bool)

    # Set the mask where there are defined coordinates
    mask[:, jj, ii] = True

    # Create the MASK Img object
    # (MASK is related to the domain when all measurements are available)
    # dx = d_east
    # dy = d_nort
  
   
    img_mask = gn.img.Img(nx=nx, ny=ny, nz=nt, sx=dx, sy=dy, ox=ox, oy=oy,
                          nv=1, varname='mask', val=(mask).astype("int"))
    
    return img_mask


def mat2som_input(mat, nb_points, nt, t_ini, t_fin):
    """
    Convert data provided in the MAT input matrix into the input suitable
    for the application of SOM.

    Parameters
    ----------
    mat : dict
        Dictionary containing the data extracted from the MAT input file.
    nb_points : int
        Total number of points in the (filled) grid.
    nt : int
        Total number of time steps.
    t_ini : int
        Initial time step.
    t_fin : int
        Final time step. This should be the same as the time step used for 
        the reference velocity field.

    Returns
    -------
    data : numpy.ndarray
        Array containing the data extracted from the MAT file.

    """

    data = np.nan*np.ones((nt, nb_points*2))
    
    for i in range(nt):
        data[i, :nb_points] = mat["hfr_total_march_2012"][0,0][0][:,i+t_ini]
        data[i, nb_points:] = mat["hfr_total_march_2012"][0,0][1][:,i+t_ini]

        
    # Check if in the final time step (the reference one) is there any nan
    ref_nan = np.sum(np.isnan(data[-1,:]))
    if ref_nan>0:
        print("    WARNING: {0} NaN values found in the reference time step".format(ref_nan))
        
    return data


def plot_som_data(som_data, fig_dir, file_name="som_data.png", title="SOM data"):
    # Plot raw data
    file_out = os.path.join(fig_dir, file_name)
    fig, ax = pl.subplots(1,1, figsize=FIGSIZE_VELS)
    pl.title(title)
    im = ax.imshow(som_data)
    ax.set_xlabel('$v_x$ and $v_y$')
    ax.set_ylabel('time steps')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    pl.colorbar(im, cax=cax)
    pl.savefig(file_out, dpi=400)
    pl.close(fig)


# ############################################
# Functions related to minisom application
# ############################################



# We need to define a distance function that allows to compute a distance
# also when some NaN are present in the dataset.
def distance_nan(x, w):
    # print("plof")
    # print(x.shape, w.shape)
    nrow, ncol, _ = w.shape
    # TODO: check if this is correct
    # diff = np.zeros((nrow, ncol))
    # When there are no data, the worst case scenario is
    diff = np.ones((nrow, ncol))
    # ATTENZIONE: RIFLETTERE BENE SU CHE VALORE METTERE QUI PER `diff`!
    # IN REALTÀ NON DOVREBBE CAMBIARE NIENTE... PIÙ CHE ALTRO DOVREI PREVEDERE
    # LA SITUAZIONE IN CUI TUTTI SONO NAN...
    for i in range(nrow):
        for j in range(ncol):
            mask = ~np.logical_or(np.isnan(x), np.isnan(w[i,j,:]))
            diff[i,j] = np.linalg.norm(np.subtract(x[mask], w[i,j,mask]))
    return diff

# # def _euclidean_distance(self, x, w):
#         return linalg.norm(subtract(x, w), axis=-1)

# def distance_nan(x, w):
#     print(x.shape, w.shape)
#     mask = ~np.isnan(x).reshape(-1)
#     return np.linalg.norm(np.subtract(x[:,mask], w[:,:,mask]))



def plot_som_dist(som, fig_dir, par):
    
    file_out = os.path.join(fig_dir, "som_distance_map.png")
    fig, ax = pl.subplots(1,1,figsize=(6, 6))
    if par["SOM"]["topology"] == "rectangular":
        pc = ax.pcolor(som.distance_map().T, cmap='gist_yarg') 
        pl.colorbar(pc)
    elif par["SOM"]["topology"] == 'hexagonal':
        xx, yy = som.get_euclidean_coordinates()
        weights = som.get_weights()
        umatrix = som.distance_map()
        # iteratively add hexagons
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                wy = yy[(i, j)] * np.sqrt(3) / 2
                hex = mp.RegularPolygon((xx[(i, j)], wy), 
                                     numVertices=6, 
                                     radius=.95 / np.sqrt(3),
                                     facecolor=ml.cm.Blues(umatrix[i, j]), 
                                     alpha=.4, 
                                     edgecolor='gray')
                ax.add_patch(hex)
        xrange = np.arange(weights.shape[0])
        yrange = np.arange(weights.shape[1])
        pl.xticks(xrange-.5, xrange)
        pl.yticks(yrange * np.sqrt(3) / 2, yrange)
        
        
        ax.set_ylim((0,weights.shape[1]* np.sqrt(3) / 2))
        
        pl.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False, colors="w") # labels along the bottom edge are off
        
        divider = make_axes_locatable(pl.gca())
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
        cb1 = ml.colorbar.ColorbarBase(ax_cb, cmap=ml.cm.Blues, 
                            orientation='vertical', alpha=.4)
        cb1.ax.get_yaxis().labelpad = 16
        cb1.ax.set_ylabel('distance from neurons in the neighbourhood',
                  rotation=270) #, fontsize=16)
        pl.gcf().add_axes(ax_cb)
        # pl.tight_layout()
    else:
        logger.warning("SOM distance plotting option not available")
    # show the color bar
  
    ax.set_aspect('equal', 'box')
   
    pl.savefig(file_out, dpi=400)
    pl.close(fig)
    
    
    

def mat2ti(mat, par, nx, ny, nt, name=""):
    """
    Gets the output of the function `mat_load` and returns is as a geone
    training image object.

    Parameters
    ----------
    mat : output of the function `mat_load`
        Data set extracted from the MAT file, to be used to fill the training
        image in the `geone.img.Img` format.
    par : dict
        A dictionary that describes the input dataset. See the attached TOMLI
        file for more information about its structure.
    nx : int
        Number of cells along *x* coordinate.
    ny : int
        Number of cells along *x* coordinate.
    nt : int
        Number of cells along *z* coordinate (time steps).
    name : string, optional
        If needed, a name can be given to the training image object.
        The default is "".

    Returns
    -------
    im_vxvy : geone.img.Img
        The training image object created from the data extracted from the
        MAT file.

    """
    
    
    ii, jj, dx, dy, ox, oy = mat2ij(mat, nx, ny)
    
    # Select day and hour of the day to be used as training
    t_ini = (par["time"]["DD_ini"]*24-1)+par["time"]["HH_ini"]
    t_fin = (par["time"]["DD_ref"]*24-1)+par["time"]["HH_ref"]
    
    # Create the values to be fit in the TI
    # vx_ti = par["DS"]["missing"]*np.ones((nt, ny, nx))
    # vy_ti = par["DS"]["missing"]*np.ones((nt, ny, nx))    
    
    vx_ti = np.nan*np.ones((nt, ny, nx))
    vy_ti = np.nan*np.ones((nt, ny, nx))    
    
    # print(t_ini, t_fin)
    
    for t_step in range(t_ini, t_fin):
        # print(t_step)
        # Read velocity for the selected time step reference
        vx = mat["hfr_total_march_2012"][0,0][0][:,t_step]
        vy = mat["hfr_total_march_2012"][0,0][1][:,t_step]
    
            
        # Set the numerical values in the matrix
        # (nan where there are no measurements, -9999999 outside the mask)
        vx_ti[t_step-t_ini, jj, ii] = vx
        vy_ti[t_step-t_ini, jj, ii] = vy
    
    vx_nan = np.sum(np.isnan(vx_ti))
    vy_nan = np.sum(np.isnan(vy_ti))
    logger.info("Number of (total) NaN, TI (vx,vy): ({0},{1})".format(vx_nan, vy_nan))
    
    # Create and empty container for the TI containing the velocities
    vxvy = np.nan*np.ones((2,nt,ny,nx))
    
    # Set the values read from the input file
    vxvy[0,:,:,:] = vx_ti
    vxvy[1,:,:,:] = vy_ti 
    
    # Create at TI object
    im_vxvy = gn.img.Img(nx=nx, ny=ny, nz=nt, sx=dx, sy=dy, ox=ox, oy=oy, nv=2,
                          varname=['vx', 'vy'], val=vxvy, name=name)    
    
    return im_vxvy



def img_vel2speed(img_vel):
    """
    Reads as input a training image (`geone.img.Img`) with the two velocity 
    components and returns a 3D matrix
    containing the computed speed.

    Parameters
    ----------
    img_vel : `geone.img.Img`
        Input data with the two components *u* and  *v* of the velocity.

    Returns
    -------
    speed : numpy.ndarray
        A 3D array containing the computed speed.

    """
    
    speed = np.zeros(img_vel.val[0,:,:,:].shape)
    
    vx = img_vel.val[0,:,:,:]
    vy = img_vel.val[1,:,:,:]
    nt = img_vel.nz                            
    for i in range(nt):
        speed[i,:,:] = np.hypot(vx[i,:,:], vy[i,:,:])
        
    return speed
        
        
        
def img_data_rm(img_vel, img_mask, per=0.25):
    
    # Compute the number of points where there are data
    mask = img_mask.val[-1,-1,:,:]
    nb_masked = int(np.sum(mask))
    
    # Compute the number of points to be removed
    nb_tbr = int(per*nb_masked)
    
    logger.info("Number of points to be removed from the reference time step: {0:d}".format(nb_tbr))
    
    # Create the incomplete TI
    img_vel_inc = copy.deepcopy(img_vel)
    
    nx, ny = img_vel.nx, img_vel.ny
    
    rem = 0
    for i in range(nx):
        for j in range(ny):
            if mask[j,i]:
                # TODO: Check here if the incomplete values should be set to
                # np.nan or -9999999
                img_vel_inc.val[0, -1, j, i] = np.nan
                img_vel_inc.val[1, -1, j, i] = np.nan
                rem = rem + 1
                if rem >= nb_tbr:
                    break
        if rem >= nb_tbr:
            break
        
    logger.info("Number of points inside the simulation grid: {0:d}".format( nb_masked))
    logger.info("Number of points outside the simulation grid: {0} ({1})".format(
        mask.size-nb_masked, mask.size))
    logger.info("Number of points to be completed: {0}".format(nb_tbr))
    
    return img_vel_inc
    

        
        
        
        
def som_data_rm(som_vel, per=0.25):
    
    # TODO: This should work if data are removed from the left of the domain.
    # (CHECK IF IT WORKS PROPERLY!)
    # Other options should be made available, like random removal, right, top
    # or bottom.
    
    # Number of available data per time step
    nb_data = int(som_vel.shape[1]/2)
    # nt = som_vel.shape[0]
    
    # Compute the number of points to be removed
    nb_tbr = int(per*nb_data)
    
    logger.info("Number of points (per variable) to be removed: {0:d}".format(nb_tbr))
    
    # Compute the number of points where there are data
    som_vel_inc = copy.deepcopy(som_vel)
    
    som_vel_inc[-1, :nb_tbr] = np.nan
    nb_tbr2 = int(nb_tbr+nb_data)
    som_vel_inc[-1, nb_data:nb_tbr2] = np.nan
    
    return som_vel_inc
        
        
def plot_som_cmp(som_vel_ref, som_vel_rec, per, plot, sharexy=True, alldata=False):
    
        
    # Number of available data per time step
    nb_data = int(som_vel_ref.shape[0]/2)
    # nt = som_vel.shape[0]
    
    # Compute the number of points to be removed
    nb_tbr = int(per*nb_data)
    nb_tbr2 = int(nb_tbr+nb_data)
    
    if alldata:
        vx_ref = som_vel_ref[:]
        vx_rec = som_vel_rec[:]
        vy_ref = som_vel_ref[:]
        vy_rec = som_vel_rec[:]
    else:
        vx_ref = som_vel_ref[:nb_tbr]
        vx_rec = som_vel_rec[:nb_tbr]
        vy_ref = som_vel_ref[nb_data:nb_tbr2]
        vy_rec = som_vel_rec[nb_data:nb_tbr2]
        
    
    r_vx = ss.pearsonr(vx_ref, vx_rec).statistic
    r_vy = ss.pearsonr(vy_ref, vy_rec).statistic
    
    vx_rmse = np.sqrt(np.sum((vx_ref-vx_rec)**2)/vx_ref.size)
    vy_rmse = np.sqrt(np.sum((vy_ref-vy_rec)**2)/vy_ref.size)
    
    vx_mae = np.sum(np.abs(vx_ref-vx_rec)/vx_ref.size)
    vy_mae = np.sum(np.abs(vy_ref-vy_rec)/vy_ref.size)
    
   
   
    
    
    
    if sharexy:
        
        xmin = ymin = som_vel_ref.min()
        xmax = ymax = som_vel_ref.max()
        
        fig, ax = pl.subplots(1,2, sharex=True, sharey=True, figsize=((12,10)))

        liney = linex = np.arange(xmin, xmax, 0.1)

        ax[0].set_title("$r=${0:.2f}".format(r_vx))
        ax[0].set(aspect='equal')
        ax[0].scatter(vx_ref, vx_rec)

    else:
        xmin = vx_ref.min()
        xmax = vx_ref.max()
        
        ymin = vy_ref.min()
        ymax = vy_ref.max()
        
        linex = np.arange(xmin, xmax, 0.1)
        liney = np.arange(ymin, ymax, 0.1)
        
        fig, ax = pl.subplots(1,2, figsize=((12,10)))

        linex = np.arange(xmin, xmax, 0.1)
        liney = np.arange(ymin, ymax, 0.1)

    # Plot vx
    # ax[0].set_title("$r=${0:.2f}, RMSE={1:.2f} [cm/s], MAE={1:.2f} [cm/s]".format(r_vx, vx_rmse, vx_mae))
    ax[0].set_title("$r=${0:.2f}".format(r_vx))
    ax[0].set(aspect='equal')
    ax[0].scatter(vx_ref, vx_rec, c="C0")
    ax[0].plot(linex, linex, color="red")
    ax[0].set_xlabel("reference $v_x$ [cm/s]")
    ax[0].set_ylabel("reconstructed $v_x$ [cm/s]")
    ax[0].set_xlim(([1.1*xmin, 1.1*xmax]))
    ax[0].set_ylim(([1.1*xmin, 1.1*xmax]))

    
    # Plot vy
    ax[1].plot(liney, liney, color="red")
    ax[1].set(aspect='equal')
    ax[1].scatter(vy_ref, vy_rec, c="C0")
    ax[1].plot(liney, liney, color="red")
    ax[1].set_xlabel("reference $v_y$ [cm/s]")
    ax[1].set_ylabel("reconstructed $v_y$ [cm/s]")
    ax[1].set_xlim(([1.1*ymin, 1.1*ymax]))
    ax[1].set_ylim(([1.1*ymin, 1.1*ymax]))    
    
    scatter_plot = os.path.join(plot, "scatterplot_real0.png")
    # pl.show()
    pl.savefig(scatter_plot, dpi=400)
    pl.close(fig)
          
          
          
    
    
    
def som_input2img(som_input, img_mask):
    """
    Get from the input matlab MAT file the shape of the mask in a structured
    grid setting like the one for the application of DS.

    Parameters
    ----------
    par : dictionary
        All the information required and contained in the parameters input file.
    nx : int
        number of cells along *x*.
    ny : int
        number of cells along *y*.
    nt : int
        number of time steps.

    Returns
    -------
    img_mask : geone.img.Img object
        Object containing the mask.

    """

    nt_som = som_input.shape[0]
    nval_som = int(som_input.shape[1]/2)
    

    nx, ny, nt = img_mask.nx, img_mask.ny, img_mask.nz
    ox, oy = img_mask.ox, img_mask.oy
    sx, sy = img_mask.sx, img_mask.sy
    
    if nt_som == 1:
        print("    WARNING: Small SOM out.")
        nt = 1
    
    
    val = np.nan*np.ones((2, nt, ny, nx), dtype=bool)
    

    if nt != nt_som:
        print("    WARNING: time size of the SOM data different from provided"
              "*nt* value.")
        
    # iv, it, iy, ix = np.where(img_mask.val==1)

    # val[0,it, iy, ix] = som_input[:nval_som]
    # val[1,it, iy, ix] = som_input[nval_som:]
    
    
    for k in range(nt):
        count = 0
        for j in range(ny-1,-1,-1):
            for i in range(nx):                
                if img_mask.val[0, k, j, i] ==1:
                    val[0,k, j, i] = som_input[k, count]
                    count_vy = nval_som+count
                    # print(i, j, k, nval_som, count_vy)
                    val[1,k, j, i] = som_input[k, count_vy]
                    count = count + 1 
                    
                

    # Create the MASK Img object
    # (MASK is related to the domain when all measurements are available)
    # dx = d_east
    # dy = d_nort
    print(val.shape)
    img_out = gn.img.Img(nx=nx, ny=ny, nz=nt, sx=sx, sy=sy, ox=ox, oy=oy,
                         nv=2, varname=["vx", "vy"], val=val)
    print(img_out)
    # img_out = gn.img.Img(nx=nx, ny=ny, nz=st, sx=sx, sy=sy, ox=ox, oy=oy,
    #                       nv=2, varname=["vx", "vy"])
    
    
    return img_out

def som_input2imgBIS(som_input, img_mask, mat):
    """
    Get from the input matlab MAT file the shape of the mask in a structured
    grid setting like the one for the application of DS.

    Parameters
    ----------
    par : dictionary
        All the information required and contained in the parameters input file.
    nx : int
        number of cells along *x*.
    ny : int
        number of cells along *y*.
    nt : int
        number of time steps.

    Returns
    -------
    img_mask : geone.img.Img object
        Object containing the mask.

    """

    
    nt_som = som_input.shape[0]
    nval_som = int(som_input.shape[1]/2)
    

    nx, ny, nt = img_mask.nx, img_mask.ny, img_mask.nz
    ox, oy = img_mask.ox, img_mask.oy
    sx, sy = img_mask.sx, img_mask.sy
    
    if nt_som == 1:
        print("    WARNING: Small SOM out.")
        nt = 1
    
    
    val = np.nan*np.ones((2, nt, ny, nx), dtype=bool)
    

    if nt != nt_som:
        print("    WARNING: time size of the SOM data different from provided"
              "*nt* value.")
        
    # iv, it, iy, ix = np.where(img_mask.val==1)

    # val[0,it, iy, ix] = som_input[:nval_som]
    # val[1,it, iy, ix] = som_input[nval_som:]
    
    ii, jj, _, _, _, _ = mat2ij(mat, nx, ny)
    
    for k in range(nt):
        val[0,k, jj, ii] = som_input[k, :nval_som]
        val[1,k, jj, ii] = som_input[k, nval_som:]

                    
                

    # Create the MASK Img object
    # (MASK is related to the domain when all measurements are available)
    # dx = d_east
    # dy = d_nort
    print(val.shape)
    img_out = gn.img.Img(nx=nx, ny=ny, nz=nt, sx=sx, sy=sy, ox=ox, oy=oy,
                         nv=2, varname=["vx", "vy"], val=val)
    print(img_out)
    # img_out = gn.img.Img(nx=nx, ny=ny, nz=st, sx=sx, sy=sy, ox=ox, oy=oy,
    #                       nv=2, varname=["vx", "vy"])
    
    
    return img_out

def test_som_input2img():
    print("")
    print("    *** START - testing `som_input2img` ***")
    
    nt = 11
    nx = 4
    ny = 3
    som_in = np.ones((nt,2*nx*ny))
    
    val = np.ones((nt, ny, nx))
    val[:,:1,:2] = 0
    img_mask = gn.img.Img(nx=nx, ny=ny, nz=nt,
                         nv=1, varname=["mask"], val=val)
    
    im = som_input2img(som_in, img_mask)
    plot_vel(im, file_out="test_som_input2img01.png")
    
    #
    #
    #
    som_in[:,:2] = 3
    im = som_input2img(som_in, img_mask)
    plot_vel(im, file_out="test_som_input2img02.png")
    
    print("    *** STOP - testing `som_input2img` ***")
    print("")
    pass

class MiniSomWithNaN(minisom.MiniSom):
    
    
    pass


#
# chatgpt
#
# Monkey-patch MiniSom to handle NaN
# class MiniSomWithNaN(minisom.MiniSom):
#     def _nan_euclidean_distance(self, x, w):
#         """Euclidean distance ignoring NaNs."""
#         mask = ~np.isnan(x)
#         if not np.any(mask):  # All values are NaN
#             return np.inf
#         return np.linalg.norm(x[mask] - w[mask])

#     def winner(self, x):
#         """Override winner to use nan-safe distance."""
#         distances = np.zeros((self._weights.shape[0], self._weights.shape[1]))
#         for i in range(self._weights.shape[0]):
#             for j in range(self._weights.shape[1]):
#                 distances[i, j] = self._nan_euclidean_distance(x, self._weights[i, j])
#         return np.unravel_index(distances.argmin(), distances.shape)

#     def _update(self, x, win, t):
#         """Override update to skip NaNs in x."""
#         eta = self._learning_rate * np.exp(-t / self._tau2)
#         sig = self._sigma * np.exp(-t / self._tau1)
#         g = self._gaussian(win, sig)[:, :, np.newaxis]

#         # Update only non-NaN features
#         mask = ~np.isnan(x)
#         delta = eta * g * (x - self._weights)
#         delta[:, :, ~mask] = 0  # Ignore NaNs in update
#         self._weights += delta
        
#     def quantization_error_nan(self, data):
#         """Quantization error that ignores NaNs."""
#         error = 0
#         count = 0
#         for x in data:
#             if np.all(np.isnan(x)):
#                 continue  # skip samples with all NaNs
#             bmu = self._weights[self.winner(x)]
#             mask = ~np.isnan(x)
#             error += np.linalg.norm(x[mask] - bmu[mask])
#             count += 1
#         return error / count if count > 0 else np.nan  

#     # def quantization_error_nan(self, data):
#     #     """Returns the quantization error computed as the average
#     #     distance between each input sample and its best matching unit."""
#     #     error = 0.0
#     #     count = 0
#     #     for x in data:
#     #         mask = np.isnan(x)
#     #         error += np.linalg.norm()
#     #     return norm(data-self.quantization(data), axis=1).mean()
      
        
#     # def _distance_from_weights(self, data):
#     #     """Returns a matrix d where d[i,j] is the euclidean distance between
#     #     data[i] and the j-th weight.
#     #     """
#     #     # print("ciao")
#     #     input_data = np.array(data)
#     #     weights_flat = self._weights.reshape(-1, self._weights.shape[2])
#     #     input_data_sq = np.power(input_data, 2).sum(axis=1, keepdims=True)
#     #     weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
#     #     cross_term = np.dot(input_data, weights_flat.T)
#     #     return np.sqrt(-2 * cross_term + input_data_sq + weights_flat_sq.T)


def img2som(img_in, img_mask, t=-1):
    
    img_in_x = img_in.val[0,t,:,:]
    img_in_y = img_in.val[1,t,:,:]
    mask = img_mask.val[0,t,:,:].astype(bool)
    data_in_x = np.flipud(img_in_x).T[np.flipud(mask).T]
    data_in_y = np.flipud(img_in_y).T[np.flipud(mask).T]
    data_out = np.concatenate((data_in_x, data_in_y))
    
    return data_out
    

def img2som_data(img_in, mat, t=-1):
    """
    

    Parameters
    ----------
    img_in : TYPE
        DESCRIPTION.
    mat : TYPE
        DESCRIPTION.
    t : TYPE, optional
        DESCRIPTION. The default is -1.

    Returns
    -------
    None.

    """
    vx = img_in.val[0,t,:,:]
    vy = img_in.val[1,t,:,:]
    ny, nx = vx.shape
    
    ii, jj, dx, dy, ox, oy = mat2ij(mat, nx, ny)
    
    vx4som = np.ones(len(ii))
    vy4som = np.ones(len(ii))
    
    for k in range(len(ii)):
        vx4som[k] = vx[jj[k], ii[k]]
        vy4som[k] = vy[jj[k], ii[k]]
        
    return np.concatenate((vx4som, vy4som))
    
    
    
    
    
def test_compare():
    
    nt = 11
    nx = 4
    ny = 3
    
    # Create and save the reference image    
    val_ref = np.ones((2, nt, ny, nx))
    val_ref[:,:,:1, :2] = np.nan
    val_ref[:,:,-1, -1] = np.nan
    
    img_ref = gn.img.Img(nx=nx, ny=ny, nz=nt,
                         nv=2, varname=["vx", "vy"], val=val_ref)
    plot_vel(img_ref, file_out="test_compare01_img_ref.png")
    
    # Create and save the incomplete image
    val_inc = np.ones((2, nt, ny, nx))
    val_inc[:,:,:1, :2] = np.nan
    val_inc[:,:,1,1:3] = np.nan
    img_inc = gn.img.Img(nx=nx, ny=ny, nz=nt,
                         nv=2, varname=["vx", "vy"], val=val_inc)
    plot_vel(img_inc, file_out="test_compare01_img_inc.png")
 
    # Create and save the simulated image
    val_sim = np.ones((2, nt, ny, nx))
    val_sim[:,:,:1, :2] = np.nan
    val_sim[:,:,1,1:3] = 1.5
    img_sim = gn.img.Img(nx=nx, ny=ny, nz=nt,
                         nv=2, varname=["vx", "vy"], val=val_sim)
    plot_vel(img_sim, file_out="test_compare01_img_sim.png")
 
    # Get the mask
    # Create and save the reference image    
    mask = np.ones((ny, nx), dtype=bool)
    mask[:1, :2] = 0
    # mask[-1, -1] = 0
    # val_ref[:,:,-1, -1] = 1
    # mask = val_ref[0,0,:,:].astype(bool)
    
    compare(img_ref, img_inc, img_sim, verbose=True,
            plot=".", z=-1)

        
if __name__ == "__main__":
    
    test_plot_vel()
    test_som_input2img()
    
    test_compare()
        
    
    print("Check the output images in the module folder to check if all is OK!")