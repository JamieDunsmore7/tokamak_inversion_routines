# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:24:19 2024

@author: jrh
"""

import matplotlib.pyplot as plt
from numpy import exp, linspace, min, max, sqrt, meshgrid, reshape, abs, shape, mean, arange, zeros, polyfit, poly1d, trapz
from scipy.interpolate import interp1d
from scipy import optimize
import cv2
from os import path, listdir
import xmltodict
import pandas as pd

from scipy.special import ndtr as norm_cdf

def ncdf(x, c, w):
    z = (x - c) / w
    return norm_cdf(z)

def gaussian(x, c, w):
    z = (x - c) / w
    return exp(-0.5*z**2)

def hsv_cal_1d(z, sigma, shape):
    g = gaussian(z, 0.0, sigma)
    f = ncdf(z, 1, shape)
    return g * (1 - f)

def hsv_cal_2d(x, y, params):
    x0, y0, wx, wy, sigma, shape, amp, offset = params
    z = sqrt(((x - x0) / wx) ** 2 + ((y - y0) / wy) ** 2)
    return amp * hsv_cal_1d(z, sigma, shape) + offset

def hsv_cal_2d_fit(xy, x0, y0, wx, wy, sigma, shape, amp, offset):
    x, y = xy
    z = sqrt(((x - x0) / wx) ** 2 + ((y - y0) / wy) ** 2)
    
    output = amp * hsv_cal_1d(z, sigma, shape) + offset
    
    return output.ravel()

def fit_image(img):
    
    '''
    
    '''
    # img_filename = 'N:/CCFE/MAST-U/Operations/Diagnostics/Planning/09-10 Visible Cameras/2024-03 Calibration/SAX2 B_C001H001S0002/SAX2 B_C001H001S0002000001.bmp'
    #
    
    #fit_data = img[500,:]
    
    img_shape = shape(img)
    
    x = linspace(0, img_shape[0]-1, img_shape[0])
    y = linspace(0, img_shape[1]-1, img_shape[1])
    
    xgrid, ygrid = meshgrid(x, y)
    
    p0 = [0.5*img_shape[0], 0.5*img_shape[1], 150, 150, 300, 0.2, max(img), min(img)]
    
    opt, _ = optimize.curve_fit(hsv_cal_2d_fit, (xgrid, ygrid), img.ravel(), p0)
    
    imgfit = reshape(hsv_cal_2d_fit((xgrid, ygrid), *opt), shape(img))
    
    residual = mean(abs(imgfit - img))
    
    return (opt, residual, imgfit)
    
    
    
    # Generate a flat-field correction to the data, which should just be the reciprocal
    # of the fitted image
    
def generate_flat_field(opt, img_size, thresh = 5.0):
    
    '''
    Generates a flat-field correction matrix based on a 2D fit performed of an image
    of a uniform illumination light source, such as an integrating sphere.  The
    size of the flat-fied correction matrix is taken from the dimensions of
    the input image, img.  In cases where there is severe vignetting, flat-field
    scale factors greater than thresh are set to zero, to mask off the part of
    the image where the flat-field correction is very high, due to low signal
    in the calibration image
    
    Inputs:
        - opt: flat-field calibration values
        - img_size: a 2 element touple containing the dimensions of the flat-field image
        
    Outputs:
        - An image of size (img_size) that can be multipled to a camera image
          to perform flat-field correction
    '''
    
    x = linspace(0, img_size[0]-1, img_size[0])
    y = linspace(0, img_size[1]-1, img_size[1])
    
    xgrid, ygrid = meshgrid(x, y)
    
    flat_field = reshape(1.0 / hsv_cal_2d_fit((xgrid, ygrid), opt[0],opt[1],opt[2],opt[3],opt[4],opt[5],1.0,0.0), shape(img_size))
    flat_field[flat_field > thresh] = 0
    
    return flat_field

def get_cih(filename):
    name, ext = path.splitext(filename)
    if ext == '.cih':
        cih = dict()
        # read the cif header
        with open(filename, 'r') as f:
            for line in f:
                if line == '\n': #end of cif header
                    break
                line_sp = line.replace('\n', '').split(' : ')
                if len(line_sp) == 2:
                    key, value = line_sp
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                        cih[key] = value
                    except:
                        cih[key] = value

    elif ext == '.cihx':
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            first_last_line = [ i for i in range(len(lines)) if '<cih>' in lines[i] or '</cih>' in lines[i] ]
            xml = ''.join(lines[first_last_line[0]:first_last_line[-1]+1])

        raw_cih_dict = xmltodict.parse(xml)
        cih = {
            'Date': raw_cih_dict['cih']['fileInfo']['date'], 
            'Camera Type': raw_cih_dict['cih']['deviceInfo']['deviceName'],
            'Record Rate(fps)': float(raw_cih_dict['cih']['recordInfo']['recordRate']),
            'Shutter Speed(s)': float(raw_cih_dict['cih']['recordInfo']['shutterSpeed']),
            'Total Frame': int(raw_cih_dict['cih']['frameInfo']['totalFrame']),
            'Original Total Frame': int(raw_cih_dict['cih']['frameInfo']['recordedFrame']),
            'Image Width': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['width']),
            'Image Height': int(raw_cih_dict['cih']['imageDataInfo']['resolution']['height']),
            'File Format': raw_cih_dict['cih']['imageFileInfo']['fileFormat'],
            'EffectiveBit Depth': int(raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['depth']),
            'EffectiveBit Side': raw_cih_dict['cih']['imageDataInfo']['effectiveBit']['side'],
            'Color Bit': int(raw_cih_dict['cih']['imageDataInfo']['colorInfo']['bit']),
            'Comment Text': raw_cih_dict['cih']['basicInfo'].get('comment', ''),
        }
        
    return cih

def integrating_sphere_radiance(luminance, min_wl, max_wl):
    
    full_dat = pd.read_excel('integrating_sphere_data.xlsx','100% 50mm port',header=None)
    full_luminance = 14.15
    
    twenty_dat = pd.read_excel('integrating_sphere_data.xlsx','20% 50mm port',header=None)
    twenty_luminance = 2.868
    
    scaled_dat = full_dat.copy()
    scaled_dat[1] = scaled_dat[1] * luminance / full_luminance
    
    calib_interp = interp1d(scaled_dat[0], scaled_dat[1])
    
    filter_wl = linspace(min_wl - 1, max_wl + 1, max_wl - min_wl + 3)
    filter_transmission = filter_wl.copy() * 0.0
    filter_transmission[1:-1] = 1
    
    calib_radiance = calib_interp(filter_wl)
    
    return trapz(calib_radiance * filter_transmission, filter_wl)

if __name__ == '__main__':

    # Read the calibration image
    img_dir = './images/'
    
    img_files = ['SAX2 B_C001H001S0019000001.png',
                 'SAX2 B_C001H001S0020000001.png',
                 'SAX2 B_C001H001S0021000001.png',
                 'SAX2 B_C001H001S0022000001.png',
                 'SAX2 B_C001H001S0023000001.png',
                 'SAX2 B_C001H001S0024000001.png']
    
    header_files = ['SAX2 B_C001H001S0019.cihx',
                    'SAX2 B_C001H001S0020.cihx',
                    'SAX2 B_C001H001S0021.cihx',
                    'SAX2 B_C001H001S0022.cihx',
                    'SAX2 B_C001H001S0023.cihx',
                    'SAX2 B_C001H001S0024.cihx']
    
    # Integrating sphere brightness, in kcd/m
    sphere_brightness = [6.445]*len(img_files)
    
    phot_flux = zeros(len(img_files))
    
    fit_results = zeros((len(img_files),8))
    exposure_times = zeros(len(img_files))
    peak_signal = []
    background = []
    
    for i in arange(len(img_files)):
    
        img = cv2.imread(img_dir + img_files[i],cv2.IMREAD_UNCHANGED)
        
        # Read the header file
        header = get_cih(img_dir + header_files[i])
        
        exposure_time = 1.0 / header['Shutter Speed(s)']
        
        exposure_times[i] = exposure_time
        
        fit_result = fit_image(img)
        
        fit_results[i,:] = fit_result[0]
        
        # calculate the photon energy
        wl = 656.0E-9
        phot_e = (6.6E-34) * (3E8) / wl
        
        # Calculate the radiance from the integrating sphere, in
        # units of W / m^-2 / sr
        srad = integrating_sphere_radiance(sphere_brightness[i], 656 - 5, 656 + 5) * 1.0E-3
        
        # Convert to units of photons / s / m^-2 / sr
        srad_phot = srad / phot_e
        
        # Store the photon flux to the sensor for each image
        phot_flux[i] = srad_phot * exposure_time
        
    plt.figure()
    
    # Plot the peak signal recorded by the camera as a function of exposure
    # time.  To do this, take the peak signal values from the fit results and
    # subtract the background signal
    peak_signals = fit_results[:,6] - fit_results[:,7]
    
    plt.plot(exposure_times, peak_signals, 'o')
    plt.xlabel('Exposure time (s)')
    plt.ylabel('Peak signal (counts)')
    plt.grid()
    
    # Plot a linear fit through the data
    calib_fit = polyfit(exposure_times, peak_signals, 1)
    
    tmp = poly1d(calib_fit)
    plt.plot(exposure_times, tmp(exposure_times), '--k')
    
    # Plot the calibration data in terms of photon flux
    plt.figure()
    plt.plot(phot_flux, peak_signals, 'o')
    plt.xlabel(r'$\phi_{ph}$ (ph m$^{-2}$ sr$^{-1}$)')
    plt.ylabel('Peak signal (counts)')
    plt.grid()
    
    calib_fit_phot = polyfit(phot_flux, peak_signals, 1)
    
    tmp_phot = poly1d(calib_fit_phot)
    plt.plot(phot_flux, tmp_phot(phot_flux), '--k')
    
    calib_factor_phot = 1.0 / calib_fit_phot[0]
    