#!/usr/bin/env python
# encoding: utf-8
"""
convert_to_footprint.py

Parse multi-epoch image FITS files and save as a footprint HDF5 file.
"""

import argparse
import sys
import os.path
import numpy as np
# import json
import ConfigParser
import pandas as pd
#import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import wcs
from footprints import Footprints

import logging

# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/sheller_great3.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


### How to convert source_type labels to integers for output catalogs
source_type_index = {'star': 0, 'galaxy': 1}



class ConfigFileParser(object):
    """
    Parse a configuration file for this script 
    """
    def __init__(self, config_file_name):
        self.config_file = config_file_name

        config = ConfigParser.RawConfigParser()
        config.read(config_file_name)

        infiles = config.items("infiles")
        print("infiles:")
        for key, infile in infiles:
            print(infile)
        self.infiles = [infile for key, infile in infiles]

        self.data_dir = os.path.expanduser(config.get('metadata', 'data_dir'))
        self.outfile = config.get('metadata', 'outfile')

        self.telescope_name = config.get('metadata', 'telescope_name')
        self.primary_diam = float(config.get('metadata', 'primary_diameter'))
        self.pixel_scale = float(config.get('metadata', 'pixel_scale'))
        self.filter_name = config.get('metadata', 'filter_name')

        self.catalog_file = config.get('metadata', 'catalog_file')

        self.nx_stamp = int(config.get('stamps', 'nx'))
        self.ny_stamp = int(config.get('stamps', 'ny'))
        return None    


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
            print('Masked {0} outlier values from this iteration.'.format(np.sum(~mask_outliers)))
            print('Current fractional sigma change between clipping iterations = {0:0.2f}'.format(sigma_frac_change))
        # Replace old values with estimates from this iteration.
        x_std_old = x_std_new.copy()
        x_median_old = np.median(x)
        # Make sure that we don't have a run away while statement.
        i += 1
        if i > 100:
            print('Background variance failed to converge after 100 sigma clipping iterations, exiting.')
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

        infile = os.path.join(infile)
        h = fits.getheader(infile)
        self.w = wcs.WCS(h)
        f = fits.open(infile)
        self.im = f[0].data
        f.close()
        return None
    
    def pos_in_pixels(self, ra, dec):
        return np.array(self.w.wcs_world2pix(ra, dec, 1)).astype(int)

    def get_rect_footprint(self, ra, dec, nx=16, ny=16):
        """
        Get a rectangular image footprint around a given catalog object

        @param nx   The number of pixels to cut out along the x-axis
        @param ny   The number of pixels to cut out along the y-axis
        """
        pos = self.pos_in_pixels(ra, dec)
        x = pos[0]
        y = pos[1]
        return np.asarray(self.im[(y-ny):(y+ny), (x-nx):(x+nx)])


def _set_outfile_with_path(indir, footprint_filename):
    """
    Specify the output file name for the Footprints including the correct path
    """
    segdir = os.path.join(indir, 'footprints')
    if not os.path.exists(segdir):
        os.makedirs(segdir)
    return os.path.join(segdir, footprint_filename)

def create_footprints_from_catalog(params, verbose=False):
    """
    Create a footprints file for all stars and galaxies in the given catalogs

    @param gal_catalog  A dict with fields: 
                        'ra', 'dec', 'star_ra', 'star_dec', 'filter_name',
                        'telescope_name'
    """
    ### Read the catalog file
    ### The number of rows defines the number of unique sources (or 
    ### 'footprints') in the output file. The catalog RA, DEC values define 
    ### where to look for sources in the input images.
    ### *** We're assuming throughout that no sources are blended ***
    catalog = pd.read_csv(os.path.join(params.data_dir, params.catalog_file))
    catalog['source_type'] = catalog['source_type'].apply(source_type_index.get).astype(float)
    n_sources = len(catalog.index)
    print("Catalog:\n", catalog)

    footprint_filename = _set_outfile_with_path(params.data_dir, params.outfile)
    # if verbose:
    print("Output file name:", footprint_filename)

    ### Set some common metadata required by the Segment file structure
    dummy_mask = 1.0
    dummy_psf = 1.0

    ### Create and fill the elements of the segment file for all galaxies
    ### in the current sub-field. Different sub-fields go in different segment
    ### files (no particular reason for this - just seems convenient).
    seg = Footprints(footprint_filename)

    ### Save all images to the footprints file, but with distinct 'segment_index'
    ### values for distinct sources (galaxies or stars).
    ### Assuming no blending.
    for isrc in xrange(n_sources):
        print("--- Saving segment {:d} ---".format(isrc))
        images = []
        psfs = []
        psf_model_names = []
        noise_vars = []
        backgrounds = []
        masks = []
        for ifile, infile in enumerate(params.infiles): # Iterate over epochs
            e = Epoch(os.path.join(params.data_dir,infile))

            images.append(e.get_rect_footprint(catalog['RA'][isrc],
                                               catalog['DEC'][isrc],
                                               nx=params.nx_stamp,
                                               ny=params.ny_stamp))
            bkgrnd, noise_var = get_background_and_noise_var(e.im)
            noise_vars.append(noise_var)
            backgrounds.append(bkgrnd)
            masks.append(dummy_mask)
            # print "empirical nosie variance: {:5.4g}".format(np.var(f[0].data))

            ### Assuming we don't have images of galaxy PSFs, put a dummy value for the PSF
            ### and specify a PSF modeling class for JIF to parse and create a PSF model
            psfs.append(dummy_psf)
            psf_model_names.append('PSFModel class')

        print("noise_vars:", noise_vars)
        seg.save_images(images,
                        noise_vars, 
                        masks,
                        backgrounds,
                        segment_index=isrc, 
                        telescope=params.telescope_name,
                        filter_name=params.filter_name)
        seg.save_psf_images(psfs,
                            segment_index=isrc,
                            telescope=params.telescope_name,
                            filter_name=params.filter_name,
                            model_names=psf_model_names)
        cat_out = catalog.iloc[isrc][['RA', 'DEC', 'source_type']]

        print("Saving catalog:", cat_out)
        seg.save_source_catalog(np.array([cat_out.values]),
                                segment_index=isrc,
                                column_names=list(cat_out.index.values))

    # ### It's not strictly necessary to instantiate a GalSimGalaxyModel object
    # ### here, but it's useful to do the parsing of existing bandpass files to
    # ### copy filter curves into the segment file.
    # gg_obj = gg.GalSimGalaxyModel(telescope_name=telescope_name,
    #     filter_names=list(filter_name), filter_wavelength_scale=1.0)
    # gg.save_bandpasses_to_segment(seg, gg_obj, k_filter_name, telescope_name,
    #     scale={"LSST": 1.0, "WFIRST": 1.e3}[telescope_name])

    seg.save_tel_metadata(telescope=params.telescope_name,
                          primary_diam=params.primary_diam,
                          pixel_scale_arcsec=params.pixel_scale,
                          atmosphere=True)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Parse multi-epoch image FITS files and save as a footprint HDF5 file.')

    parser.add_argument('--config_file', type=str, default=None,
                        help="Name of a configuration file listing inputs." +
                             "If specified, ignore other command line flags." +
                             "(Default: None)")

    parser.add_argument('--infiles', type=str, nargs='+',
                        help="Input image FITS files")

    parser.add_argument('--outfile', type=str, default='footprints.h5',
                        help="Name of the output HDF5 file (Default: footprints.h5)")

    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose messaging")

    args = parser.parse_args()
    verbose = args.verbose

    ###
    ### Get the parameters for input/output from configuration file or argument list
    ###
    if isinstance(args.config_file, str):
        logging.info('Reading from configuration file {}'.format(args.config_file))
        args = ConfigFileParser(args.config_file)

    elif not isinstance(args.infiles, list):
        raise ValueError("Must specify either 'config_file' or 'infiles' argument")

    logging.debug('Creating footprint file for {:d} epochs'.format(len(args.infiles)))

    create_footprints_from_catalog(args, verbose=verbose)

    logging.debug('Finished creating footprint file')    
    return 0


if __name__ == "__main__":
    sys.exit(main())
