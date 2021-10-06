#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Wed Sep 29 17:01:56 2021
"""
import numpy as np

__all__ = [
           "filt2", "gaussfilter"
          ]

def filt2(Z, res, wl, filtertype, truncate=2.6) :
    filtertype = filtertype.lower()

    if len(Z.shape) != 2:
        raise TypeError('Z should b a 2D array')

    if type(res) not in [int, float]:
        raise TypeError('res must be a scalar value.')

    if filtertype not in ['lp', 'hp', 'bp', 'bs']:
        raise TypeError('filtertype must be either lp (low pass), \
hp (high pass), bp (band pass) or bs (band stop)')

    if wl <= 2 * res:
        print('WARNING:: Nyquist says the wavelength should exceed two times \
              the resolution of the dataset, which is an unmet condition based on these inputs')

    if  filtertype in ['bp','bs']:
        if hasattr(wl, '__len__'):
            if len(wl) != 2 or type(wl) == str:
                raise TypeError('Wavelength lambda must be a two-element array for a bandpass filter.')


            if type(wl) != np.array:
                wl = np.array(list(wl))

        else:
            raise TypeError('Wavelength lambda must be a two-element array for a bandpass filter.')
    else: #so must be either hp or lp
        if hasattr(wl, '__len__'):
            raise TypeError('Wavelength lambda must be a scalar for lowpass or highpass filters.')


    sigma = (wl / res) /(2 * np.pi)

    if filtertype == 'lp':
        return gaussfilter(Z, sigma, truncate) # ndnanfilter is Carlos Adrian Vargas Aguilera's excellent function, which is included as a subfunction below.

    elif filtertype == 'hp':
        return Z - gaussfilter(Z, sigma, truncate)

    elif filtertype == 'bp' :
        return  filt2(filt2(Z, res, np.max(wl), 'hp'), res, np.min(wl), 'lp')

    else: #Leaves the case of 'bs'
        return filt2(Z, res, np.max(wl), 'lp') - filt2(Z, res, np.min(wl),'hp')


def gaussfilter(Z, sigma, truncate):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(Z, sigma, truncate=truncate, mode='nearest')
