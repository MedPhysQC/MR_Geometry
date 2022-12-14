#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# PyWAD is open-source software and consists of a set of modules written in python for the WAD-Software medical physics quality control software.
# The WAD Software can be found on https://github.com/wadqc
#
# The pywad package includes modules for the automated analysis of QC images for various imaging modalities.
# PyWAD has been originaly initiated by Dennis Dickerscheid (AZN), Arnold Schilham (UMCU), Rob van Rooij (UMCU) and Tim de Wit (AMC)


"""

Dependencies:

    PYTHON
    - numpy
    - pydicom / dicom
    - datetime
    - statistics
    - ast


Version control:
    20170303: first version (WAD QC 1.0)
    20190128: second version (WAD QC 2.0)


The initial version was created by Erik van der Bijl.
The initial version was adapted by Stijn van de Schoot at the Amsterdam UMC, location AMC in order to use the code for the WAD QC 2.0 software

"""

__version__ = '20190429'
__author__ = 'Stijn van de Schoot (s.vandeschoot@amc.uva.nl)'
__location__ = 'Amsterdam UMC, location AMC'


import sys
import os
import numpy as np
import datetime

# Import WAD QC modules
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

# Import required module(s)
try:
    import GeomAcc_lib as GeomAcc
except ImportError:
    print('{0} Could not import GeomAcc_lib module.'.format(logTag()))

# Import DICOM
try:
    import pydicom as dicom
except ImportError:
    import dicom




def logTag():
    return "[GeomAcc_analysis.py] ::"

def logTagInitialisation():
    return "[****]"

def startLogging():
    print('===========================================================================================')
    print('{0} This WAD QC module is written by {1} @ {2}'.format(logTagInitialisation(), __author__, __location__))
    print('{0} The current version used is: {1}'.format(logTagInitialisation(), __version__))
    print('\n')
    print('{0} This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;'.format(logTagInitialisation()))
    print('{0} without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.'.format(logTagInitialisation()))
    print('{0} See the GNU General Public License for more details.'.format(logTagInitialisation()))
    print('\n')
    print('{0} Included module(s):'.format(logTagInitialisation()))
    print('{0}      - {1} (version: {2}; author: {3})'.format(logTagInitialisation(), GeomAcc.__name__, GeomAcc.__version__, GeomAcc.__author__))
    print('===========================================================================================')
    print('\n')



def getAcquistionDateTime(data, results):

    """
    Read acquisition data and time from dicomheaders and write to IQC database

    Workflow:
        1. Read only header and get date and time
    """

    ## 1. read only header and get date and time

    # Add info to logfile
    print('{0} getAcquisitionDateTime -- INFO: Reading DICOM header and get date and time of acquisition.'.format(logTag()))

    # Get DICOM info
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    # Add data to results
    print('{0} getAcquisitionDateTime -- INFO: The acquisition date and time is: {1}.'.format(logTag(), dt))
    print('{0} getAcquisitionDateTime -- INFO: Adding date and time of acquisition to results.'.format(logTag()))
    results.addDateTime('Acquisition date and time', dt)


def geomAccAnalysis(data, results, action):

    """"
    Start analysis

    Workflow:
        1. Initialisation of analysis module
        2. Load images and detect cluster locations
        3. Correct detected cluster positions for setup
        4. Save results to WAD database
    """

    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. Initialisation of analysis module

    # Add info to logfile
    print('{0} geomAccAnalysis -- INFO: Initialisation of analysis module.'.format(logTag()))

    # Get phantom type from config file
    try:
        phantomType = params['Phantom Type']
        print('{0} geomAccAnalysis -- INFO: The phantom type from the config file is: {1}.'.format(logTag(), phantomType))
    except KeyError:
        print('{0} geomAccAnalysis -- ERROR: Unable to read phantom specifications from config file.'.format(logTag()))


    # Initiate module
    geomAccuracy = GeomAcc.GeomAcc(phantomType)
    print('\n')


    ## 2. Load images and detect cluster locations

    # Add info to logfile
    print('{0} geomAccAnalysis -- INFO: Loading images and detecting cluster locations.'.format(logTag()))

    # load images and detect cluster locations
    geomAccuracy.loadImagesAndDetectClusterLocations(data)
    print('\n')


    ## 3. Correct detected cluster positions for setup

    # Add info to logfile
    print('{0} geomAccAnalysis -- INFO: Correcting detected cluster positions for setup.'.format(logTag()))

    #  correct detected cluster positions for current setup
    geomAccuracy.correctDetectedClusterPositionsForSetup()
    print('\n')


    ## 4. Save results to WAD database

    # Add info to logfile
    print('{0} geomAccAnalysis -- INFO: Saving results to WAD database.'.format(logTag()))

    # Save numeric
    geomAccuracy.save_numeric_to_wad(results)

    # Save images
    geomAccuracy.save_images_to_wad(results)

    # Save detected clusters
    geomAccuracy.save_detected_clusters_to_wad(results)




def dataVerification(data, config):

    # Initialisation
    print('{0} dataVerification -- INFO: Start verifying the incoming DICOM data.'.format(logTag()))
    result = True

    # Read the header of the first DICOM file
    dcmFile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    # Read parameters from config
    action = config['Data Validation']
    parameters = action['params']

    # Check if correct Patient Name is used
    if dcmFile.PatientName != parameters['Patients Name']:
        print('{0} dataVerification -- ERROR: Incorrect Patient Name.'.format(logTag()))
        result = False

    # Check if the expected number of series are present
    if len(data.getAllSeries()) < int(parameters['Number of data series']):
        print('{0} dataVerification -- ERROR: Incorrect number of data series (expected = {1}; detected = {2}).'.format(logTag(), int(parameters['Number of data series']), len(data.getAllSeries())))
        result = False

    # Check if data type of pixel data is correct
    dicomSeriesList = data.getAllSeries()
    for series in dicomSeriesList:
        instance = series[0]
        seriesNumber = str(instance.SeriesNumber)
        if seriesNumber.endswith('1'):
            try:
                series[10].pixel_array.astype('float32')
            except:
                result = False
                print('{0} dataVerification -- ERROR: Type of pixel data is not supported.'.format(logTag()))

    return result





if __name__=="__main__":

    # start logging
    startLogging()

    # initialisation of WAD module
    data, results, config = pyWADinput()

    # read runtime parameters for WAD module in specific order
    if dataVerification(data, config['actions']):

        for name, action in config['actions'].items():
            if name == 'Acquisition DateTime':
                getAcquistionDateTime(data, results)

        for name, action in config['actions'].items():
            if name == 'Phantom Analysis':
                geomAccAnalysis(data, results, action)
                results.addString('Status analysis', 'Complete')

    else:
        for name, action in config['actions'].items():
            if name == 'Acquisition DateTime':
                getAcquistionDateTime(data, results)
                results.addString('Status analysis', 'Incomplete')


        print('{0} Main -- ERROR: data does not fulfill all requirements! An incomplete analysis is performed.'.format(logTag()))

    # add version number to results
    results.addString('Version', __version__)

    # add results to database
    results.write()