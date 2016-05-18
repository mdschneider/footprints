#!/usr/bin/env python
# encoding: utf-8
"""
sheller_great3.py

Created by Michael Schneider on 2015-10-16
"""

import argparse
import sys
import os.path
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt

from astropy.io import fits
from footprints import Footprints

import logging

# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/sheller_great3.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


### Number of pixels per galaxy postage stamp, per dimension
k_g3_ngrid = {"ground": 48, "space": 96}
### Pixel scales in arcseconds
k_g3_pixel_scales = {"ground": 0.2, "space_single_epoch": 0.05,
                     "space_multi_epoch": 0.1}
### Guess what values were used to simulate optics PSFs
k_g3_primary_diameters = {"ground": 8.2, "space": 2.4}
### Guess that GREAT3 used LSST 'r' band to render images
k_filter_name = 'r'
k_filter_central_wavelengths = {'r':620.}


def get_background_and_noise_var(data, clip_n_sigma=3, clip_converg_tol=0.1,
    verbose=False):
    """
    Determine the image background level.

    clip_n_sigma = Number of standard deviations used to define outliers to
        the assumed Gaussian random noise background.
    convergence_tol = the fractional tolerance that must be met before
        iterative sigma clipping proceedure is terminated.

    This is currently largely based on the SExtractor method, which is in
    turn based on the Da Costa (1992) method. Currently the entire image is
    used in the background estimation proceedure but you could imaging a
    gridded version of the following which could account for background
    variation across the image.
    TODO: Create a background image instead of just a background value.
    """
    # Inatilize some of the iterative parameters for the while statement.
    sigma_frac_change = clip_converg_tol + 1
    i = 0
    #
    x = np.copy(data.ravel())
    # Calculate the median and standard deviation of the initial image
    x_median_old = np.median(x)
    x_std_old = np.std(x)
    # Iteratively sigma clip the pixel distribution.
    while sigma_frac_change > clip_converg_tol:
        # Mask pixel values
        mask_outliers = np.logical_and(x >= x_median_old -
                                       clip_n_sigma*x_std_old,
                                       x <= x_median_old +
                                       clip_n_sigma*x_std_old)
        # Clip the data.
        x = x[mask_outliers]
        x_std_new = np.std(x)
        # Check percent difference between latest and previous standard
        # deviation values.
        sigma_frac_change = np.abs(x_std_new-x_std_old)/((x_std_new+x_std_old)/2.)
        if verbose:
            print 'Masked {0} outlier values from this iteration.'.format(np.sum(~mask_outliers))
            print 'Current fractional sigma change between clipping iterations = {0:0.2f}'.format(sigma_frac_change)
        # Replace old values with estimates from this iteration.
        x_std_old = x_std_new.copy()
        x_median_old = np.median(x)
        # Make sure that we don't have a run away while statement.
        i += 1
        if i > 100:
            print 'Background variance failed to converge after 100 sigma clipping iterations, exiting.'
            sys.exit()
    # Calculate the clipped image median.
    x_clip_median = np.median(x)
    # Calculate the clipped image mean.
    x_clip_mean = np.mean(x)
    # Estimate the clipped image mode (SExtractor's version of Da Costa 1992).
    # This is the estimate of the image background level.
    background = float(2.5 * x_clip_median - 1.5 * x_clip_mean)
    # Calculate the standard deviation of the pixel distribution
    noise_var = float(np.var(x))
    return background, noise_var


class Epoch(object):
    """
    Load an image of stars and galaxies from a FITS file

    - Parse the WCS in the FITS header
    - Store catalog information to find and cutout select sources
    """
    ### Target star and galaxy centroids from SDSS catalog
    ### RA, DEC in J2000 degrees decimal
    # star_pos = [[9.924491, 0.964729],
    #             [9.922445, 0.977756]]
    # gal_pos = [[9.928746, 0.973612],
    #            [9.930459, 0.960453]]
    def __init__(self, infile, gal_pos=None, star_pos=None):
        self.gal_pos = gal_pos
        self.star_pos = star_pos

        infile = os.path.join('data', infile)
        h = fits.getheader(infile)
        self.w = wcs.WCS(h)
        f = fits.open(infile)
        self.im = f[0].data
        f.close()
        return None
    
    def star_pos_pix(self, istar=0):
        return np.array(self.w.wcs_world2pix(self.star_pos[istar][0],
                                             self.star_pos[istar][1], 1)).astype(int)

    def gal_pos_pix(self, igal=0):
        return np.array(self.w.wcs_world2pix(self.gal_pos[igal][0],
                                             self.gal_pos[igal][1], 1)).astype(int)

    def get_rect_footprint(self, nx=16, ny=16):
        """
        Get a rectangular image footprint around a given catalog object

        @param nx   The number of pixels to cut out along the x-axis
        @param ny   The number of pixels to cut out along the y-axis
        """
        return np.asarray(self.im[(y-ny):(y+ny), (x-nx):(x+nx)])


def create_footprints_from_catalog(infiles, seg_filename, 
                                   gal_catalog=None, star_catalog=None,
                                   verbose=False):
    """
    Create a footprints file for all stars and galaxies in the given catalogs

    @param gal_catalog  A dict with fields: 
                        'ra', 'dec', 'star_ra', 'star_dec', 'filter_name',
                        'telescope_name'
    """
    if verbose:
        print "input files:", infiles
        print "star input files:", starfiles

    ### Specify the output filename for the Segments
    segdir = os.path.join(indir, "footprints")
    if not os.path.exists(segdir):
        os.makedirs(segdir)
    seg_filename = os.path.join(segdir, seg_filename)
    if verbose:
        print "seg_filename:", seg_filename

    ### Set some common metadata required by the Segment file structure
    dummy_mask = 1.0

    ### Create and fill the elements of the segment file for all galaxies
    ### in the current sub-field. Different sub-fields go in different segment
    ### files (no particular reason for this - just seems convenient).
    seg = Footprints(seg_filename)

    ### Save all images to the footprints file, but with distinct 'segment_index'
    ### values.
    images = []
    psfs = []
    noise_vars = []
    backgrounds = []
    for ifile, infile in enumerate(infiles): # Iterate over epochs
        e = Epoch(infile)

        images.append(e.get_rect_footprint())
        bkgrnd, noise_var = get_background_and_noise_var(e.im)
        noise_vars.append(noise_var)
        backgrounds.append(bkgrnd)
        # print "empirical nosie variance: {:5.4g}".format(np.var(f[0].data))

        psfs.append(e.get_rect_footprint())

    print "noise_vars:", noise_vars
    seg.save_images(images, noise_vars, [dummy_mask], backgrounds,
        segment_index=igal, telescope=telescope_name)
    seg.save_psf_images(psfs, segment_index=igal, telescope=telescope_name,
        filter_name=filter_name, model_names=None)
    seg.save_source_catalog(np.reshape(gal_cat[igal], (1,)),
        segment_index=igal)

    # ### It's not strictly necessary to instantiate a GalSimGalaxyModel object
    # ### here, but it's useful to do the parsing of existing bandpass files to
    # ### copy filter curves into the segment file.
    # gg_obj = gg.GalSimGalaxyModel(telescope_name=telescope_name,
    #     filter_names=list(filter_name), filter_wavelength_scale=1.0)
    # gg.save_bandpasses_to_segment(seg, gg_obj, k_filter_name, telescope_name,
    #     scale={"LSST": 1.0, "WFIRST": 1.e3}[telescope_name])

    # seg.save_tel_metadata(telescope=telescope_name,
    #                       primary_diam=k_g3_primary_diameters[observation_type],
    #                       pixel_scale_arcsec=k_g3_pixel_scales[observation_type],
    #                       atmosphere=(observation_type == "ground"))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Parse multi-epoch image FITS files and save as a footprint HDF5 file.')

    parser.add_argument('--infiles', type=str, nargs='+',
                        help="Input image FITS files")

    parser.add_argument('--outfile', type=str, default='footprint.h5',
                        help="Name of the output HDF5 file (Default: footprint.h5)")

    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose messaging")

    args = parser.parse_args()
    logging.debug('Creating footprint file for {:d} epochs'.format(len(args.infiles)))


    create_footprints_from_catalog(args.infiles, args.outfile, verbose=args.verbose)

    logging.debug('Finished creating footprint file')
    return 0


if __name__ == "__main__":
    sys.exit(main())
