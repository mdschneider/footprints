"""
This is a simple script to create a plot of the segment pixel data. Can be
run from the command line like:
> python showfootprint.py somefile.hdf5 footprint#

where somefile.hdf5 is the file name (perhaps including path) to an hdf5 file
created with sheller.py, and segment# is a integer referecing one of the
segments in that hdf5 file.

Alternatively it can be called by importing showseg and using:
showseg.plot(filename,segment)
"""
import sys
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# plt.style.use('ggplot')

def plot_single_epoch(file_name, segment, telescope, filter_name, epoch_num, output_name=None):
    """
    Plot the image from single epoch and telescope

    Input:
    file_name = [string] the name perhaps including path of an hdf5 file
       created by sheller.py
    segment = [string] the id of the segment to plot
    output_name = [string] name of the image to save
    """
    # open the hdf5 file as read only, techincally this is a fast operation
    # as there is only an i/o hit when specific datasets are accessed
    f = h5py.File(file_name,"r")
    g_img = f["Footprints/seg{:d}/{}/{}/epoch_{:d}/image".format(segment,
        telescope.lower(), filter_name, epoch_num)]
    # create the figure
    # adjust y size of figure
    x_size = 8
    y_size = 8
    # note that some of the following will have to be changed when we start
    # considering objects on differnt pixels scales and orientations
    fig, ax0 = plt.subplots(1, 1, sharey=True, figsize=(x_size,y_size))
    # plot the data
    im = ax0.imshow(g_img, origin='lower', interpolation='nearest', cmap=plt.cm.pink)
    fig.colorbar(im)
    
    ax0.set_title('Image')
    ax0.set_ylabel('y [pixels]')
    ax0.set_xlabel('x [pixels]')
    
    # correct imshow sizes
    ax0.set_adjustable('box-forced')
    fig.subplots_adjust(wspace=0)
    # save the image to file if output name 
    if output_name != None:
        plt.savefig(output_name)
    
    plt.show()

    f.close()
    return None


def plot_all_epochs(file_name, segment, telescope, filter_name, output_name=None):
    f = h5py.File(file_name,"r")

    h = 'Footprints/seg{:d}/{}/{}'.format(segment, telescope.lower(), filter_name)
    nepochs = len(f[h])
    # Try to make plot aspect ratio near golden
    ncols = int(np.ceil(np.sqrt(nepochs*1.618)))
    nrows = int(np.ceil(1.0*nepochs/ncols))

    ### Set color scale limits from the first epoch (limits to be applied to all epochs)
    img = f[h + '/epoch_0/image']
    vmin = np.min(img) * 0.99
    vmax = np.max(img) * 1.01

    fig = plt.figure(figsize = (3.0*ncols,3.0*nrows))

    for iepoch in xrange(nepochs):
        img = f[h + '/epoch_{:d}/image'.format(iepoch)]
        ax = fig.add_subplot(nrows, ncols, iepoch+1)
        im = ax.imshow(img, origin='lower', interpolation='nearest', cmap=plt.cm.pink,
                       vmin=vmin, vmax=vmax)
        ax.set_title("epoch {:d}".format(iepoch))
    ### Make a common colorbar for all subplots
    cax = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    fig.colorbar(im, cax=cax)
    fig.suptitle("Footprint {:d}".format(segment), fontsize=14)

    if output_name != None:
        plt.savefig(output_name)
    
    plt.show()
    f.close()
    return None


def plot_image_and_model(file_name, segment, telescope, filter_name, epoch_num,
                         galaxy_model="star", active_params=['psf_e'], parameters=[0.0],
                         psf_model=None, achromatic=True, output_name=None):
    """
    Plot a single epoch image and an associated model

    Requires JIF for the image model
    """
    import galsim
    import jif

    f = h5py.File(file_name,"r")
    
    ### Get telescope metadata
    tel_group = f['telescopes/{}'.format(telescope)]
    pixel_scale = tel_group.attrs['pixel_scale_arcsec']
    primary_diam = tel_group.attrs['primary_diam']
    atmosphere = tel_group.attrs['atmosphere']
    gain = tel_group.attrs['gain']
    
    ### Get the image data
    h = 'Footprints/seg{:d}/{}/{}'.format(segment, telescope.lower(), filter_name)
    img = f[h + "/epoch_{:d}/image".format(epoch_num)]
    nx, ny = img.shape

    gg = jif.GalSimGalaxyModel(telescope_name=telescope,
                               pixel_scale_arcsec=pixel_scale,
                               galaxy_model=galaxy_model,
                               active_parameters=active_params,
                               primary_diam_meters=primary_diam,
                               filter_names=[filter_name],
                               atmosphere=atmosphere,
                               psf_model=psf_model,
                               achromatic_galaxy=achromatic)
    gg.set_params(parameters)
    out_image = galsim.ImageF(nx, ny, scale=pixel_scale)
    model = gg.get_image(out_image=out_image, add_noise=False, filter_name=filter_name, gain=gain)


    fig = plt.figure(figsize=(15, 6/1.618))

    vmin = np.min(img)
    vmax = np.max(img)

    ### pixel data
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Image")
    cax = ax.imshow(img, interpolation='nearest',
                    cmap=plt.cm.pink, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cax)

    ### model
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Model")
    cax = ax.imshow(model.array, interpolation='nearest',
                    cmap=plt.cm.pink)#, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(cax)

    resid = img - model.array

    ### residual (chi)
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Residual")
    cax = ax.imshow(resid, interpolation='nearest',
                    cmap=plt.cm.BrBG)
    cbar = fig.colorbar(cax)

    plt.tight_layout()
    # fig.suptitle("Fooptrint {:d}".format(segment), fontsize=14)
    if output_name != None:
        plt.savefig(output_name)

    plt.show()
    f.close()
    return None


def main():
    parser = argparse.ArgumentParser(description="Create a plot of segment pixel data.")

    parser.add_argument("hdf5filename", type=str, help="Name of the input HDF5 file.")

    parser.add_argument("segment_num", type=int, help="ID number of the segment to plot.")

    parser.add_argument("telescope", type=str, help="Name of the originating telescope for the image")

    parser.add_argument("filter", type=str, help="Name of the filter for the image")

    parser.add_argument("--epoch_num", type=int, default=0, help="Epoch index")

    args = parser.parse_args()

    # plot_single_epoch(args.hdf5filename, args.segment_num, args.telescope, args.filter, args.epoch_num)

    plot_all_epochs(args.hdf5filename, args.segment_num, args.telescope, args.filter)

    # model_paramnames = ['psf_fwhm', 'psf_e', 'psf_beta', 'psf_mag']
    # model_params = [1.3, 0.0, 0.0, 28.6]
    # plot_image_and_model(args.hdf5filename, args.segment_num, args.telescope, args.filter, args.epoch_num,
    #                      active_params=model_paramnames, parameters=model_params)
    return 0    

if __name__ == "__main__":
    sys.exit(main())