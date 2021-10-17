#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.8
Created on Wed Sep 29 17:01:56 2021
"""
import numpy as np

__all__ = ["filt2", "gaussfilter"]


def filt2(Z, res, wl, filtertype, truncate=2.6):
    filtertype = filtertype.lower()

    if len(Z.shape) != 2:
        raise TypeError("Z should b a 2D array")

    if type(res) not in [int, float]:
        raise TypeError("res must be a scalar value.")

    if filtertype not in ["lowpass", "highpass", "bandpass", "bandstop"]:
        raise TypeError(
            "filtertype must be either lp (low pass), \
hp (high pass), bp (band pass) or bs (band stop)"
        )

    if hasattr(wl, "__len__") and type(wl) != np.ndarray:
        wl = np.array(wl)

    if np.any(wl <= 2 * res):
        print(
            "WARNING:: Nyquist says the wavelength should exceed two times \
              the resolution of the dataset, which is an unmet condition based on these inputs"
        )

    if filtertype in ["bandpass", "bandstop"]:
        if hasattr(wl, "__len__"):
            if len(wl) != 2 or type(wl) == str:
                raise TypeError(
                    "Wavelength lambda must be a two-element array for a bandpass filter."
                )

            if type(wl) != np.array:
                wl = np.array(list(wl))

        else:
            raise TypeError(
                "Wavelength lambda must be a two-element array for a bandpass filter."
            )
    else:  # so must be either hp or lp
        if hasattr(wl, "__len__"):
            raise TypeError(
                "Wavelength lambda must be a scalar for lowpass or highpass filters."
            )

    sigma = (wl / res) / (2 * np.pi)

    if filtertype == "lowpass":
        return gaussfilter(
            Z, sigma, truncate
        )  # ndnanfilter is Carlos Adrian Vargas Aguilera's excellent function, which is included as a subfunction below.

    elif filtertype == "highpass":
        return Z - gaussfilter(Z, sigma, truncate)

    elif filtertype == "bandpass":
        return filt2(filt2(Z, res, np.max(wl), "highpass"), res, np.min(wl), "lowpass")

    else:  # Leaves the case of 'bs'
        return filt2(Z, res, np.max(wl), "lowpass") - filt2(
            Z, res, np.min(wl), "highpass"
        )


def gaussfilter(Z, sigma, truncate):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(Z, sigma, truncate=truncate, mode="nearest")


if __name__ == "__main__":
    res = 0.2  # 200 m resolution

    x = np.arange(0, 100 + res, res)  # eastings from 0 to 100 km
    y = np.arange(0, 100 + res, res)  # northings from 0 to 100 km
    X, Y = np.meshgrid(x, y)

    # Z contains 25 km features, ~5 km diagonal features, and noise:
    Z = (
        np.cos(2 * np.pi * X / 25)
        + np.cos(2 * np.pi * (X + Y) / 7)
        + np.random.randn(X.shape[0], X.shape[1])
    )

    import matplotlib.pyplot as pt

    pt.matshow(Z, aspect="auto", extent=[0, 100, 0, 100])
    pt.xlabel("eastings (km)")
    pt.ylabel("northings (km)")
    pt.title("Original with Noise")
    pt.show()

    Zlow = filt2(Z, res, 15, "lowpass")
    pt.matshow(Zlow, aspect="auto", extent=[0, 100, 0, 100])
    pt.xlabel("eastings (km)")
    pt.ylabel("northings (km)")
    pt.title("15 km lowpass filtered data")
    pt.show()

    Zhi = filt2(Z, res, 15, "highpass")
    pt.matshow(Zhi, aspect="auto", extent=[0, 100, 0, 100])
    pt.xlabel("eastings (km)")
    pt.ylabel("northings (km)")
    pt.title("15 km highpass filtered data")
    pt.show()

    Zbp = filt2(Z, res, [4, 7], "bandpass")
    pt.matshow(Zbp, aspect="auto", extent=[0, 100, 0, 100])
    pt.xlabel("eastings (km)")
    pt.ylabel("northings (km)")
    pt.title("4 to 7 km bandpass filtered data")
    pt.show()

    Zbs = filt2(Z, res, [3, 12], "bandstop")
    pt.matshow(Zbs, aspect="auto", extent=[0, 100, 0, 100])
    pt.xlabel("eastings (km)")
    pt.ylabel("northings (km)")
    pt.title("3 to 12 km bandstop filtered data")
