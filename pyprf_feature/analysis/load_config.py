# -*- coding: utf-8 -*-
"""Load py_pRF_mapping config file."""

import os
import csv
import ast

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def load_config(strCsvCnfg, lgcTest=False, lgcPrint=True):
    """
    Load py_pRF_mapping config file.

    Parameters
    ----------
    strCsvCnfg : string
        Absolute file path of config file.
    lgcTest : Boolean
        Whether this is a test (pytest). If yes, absolute path of this function
        will be prepended to config file paths.
    lgcPrint : Boolean
        Print config parameters?

    Returns
    -------
    dicCnfg : dict
        Dictionary containing parameter names (as keys) and parameter values
        (as values). For example, `dicCnfg['varTr']` contains a float, such as
        `2.94`.

    """

    # Dictionary with config information:
    dicCnfg = {}

    # Open file with parameter configuration:
    # fleConfig = open(strCsvCnfg, 'r')
    with open(strCsvCnfg, 'r') as fleConfig:

        # Read file  with ROI information:
        csvIn = csv.reader(fleConfig,
                           delimiter='\n',
                           skipinitialspace=True)

        # Loop through csv object to fill list with csv data:
        for lstTmp in csvIn:

            # Skip comments (i.e. lines starting with '#') and empty lines.
            # Note: Indexing the list (i.e. lstTmp[0][0]) does not work for
            # empty lines. However, if the first condition is no fullfilled
            # (i.e. line is empty and 'if lstTmp' evaluates to false), the
            # second logical test (after the 'and') is not actually carried
            # out.
            if lstTmp and not (lstTmp[0][0] == '#'):

                # Name of current parameter (e.g. 'varTr'):
                strParamKey = lstTmp[0].split(' = ')[0]
                # print(strParamKey)

                # Current parameter value (e.g. '2.94'):
                strParamVlu = lstTmp[0].split(' = ')[1]
                # print(strParamVlu)

                # Put paramter name (key) and value (item) into dictionary:
                dicCnfg[strParamKey] = strParamVlu

    # Are model parameters in cartesian or polar coordinates?
    # set either pol (polar) or crt (cartesian)
    dicCnfg['strKwCrd'] = ast.literal_eval(dicCnfg['strKwCrd'])
    if lgcPrint:
        print('---Model coordinates are in: ' + str(dicCnfg['strKwCrd']))

    # Number of x- or radial positions to model:
    dicCnfg['varNum1'] = int(dicCnfg['varNum1'])
    # Number of y- or angular positions to model:
    dicCnfg['varNum2'] = int(dicCnfg['varNum2'])

    if lgcPrint:
        if dicCnfg['strKwCrd'] == 'crt':
            print('---Number of x-positions to model: ' +
                  str(dicCnfg['varNum1']))
            print('---Number of y-positions to model: ' +
                  str(dicCnfg['varNum2']))

        elif dicCnfg['strKwCrd'] == 'pol':
            print('---Number of radial positions to model: ' +
                  str(dicCnfg['varNum1']))
            print('---Number of angular positions to model: ' +
                  str(dicCnfg['varNum2']))

    # Number of pRF sizes to model:
    dicCnfg['varNumPrfSizes'] = int(dicCnfg['varNumPrfSizes'])
    if lgcPrint:
        print('---Number of pRF sizes to model: '
              + str(dicCnfg['varNumPrfSizes']))

    # Extent of visual space from centre of the screen in negative x-direction
    # (i.e. from the fixation point to the left end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtXmin'] = float(dicCnfg['varExtXmin'])
    if lgcPrint:
        print('---Extent of visual space in negative x-direction: '
              + str(dicCnfg['varExtXmin']))

    # Extent of visual space from centre of the screen in positive x-direction
    # (i.e. from the fixation point to the right end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtXmax'] = float(dicCnfg['varExtXmax'])
    if lgcPrint:
        print('---Extent of visual space in positive x-direction: '
              + str(dicCnfg['varExtXmax']))

    # Extent of visual space from centre of the screen in negative y-direction
    # (i.e. from the fixation point to the lower end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtYmin'] = float(dicCnfg['varExtYmin'])
    if lgcPrint:
        print('---Extent of visual space in negative y-direction: '
              + str(dicCnfg['varExtYmin']))

    # Extent of visual space from centre of the screen in positive y-direction
    # (i.e. from the fixation point to the upper end of the screen) in degrees
    # of visual angle.
    dicCnfg['varExtYmax'] = float(dicCnfg['varExtYmax'])
    if lgcPrint:
        print('---Extent of visual space in positive y-direction: '
              + str(dicCnfg['varExtYmax']))

    # Minimum pRF model size (standard deviation of 2D Gaussian) [degrees of
    # visual angle]:
    dicCnfg['varPrfStdMin'] = float(dicCnfg['varPrfStdMin'])
    if lgcPrint:
        print('---Minimum pRF model size: ' + str(dicCnfg['varPrfStdMin']))

    # Maximum pRF model size (standard deviation of 2D Gaussian) [degrees of
    # visual angle]:
    dicCnfg['varPrfStdMax'] = float(dicCnfg['varPrfStdMax'])
    if lgcPrint:
        print('---Maximum pRF model size: ' + str(dicCnfg['varPrfStdMax']))

    # Volume TR of input data [s]:
    dicCnfg['varTr'] = float(dicCnfg['varTr'])
    if lgcPrint:
        print('---Volume TR of input data [s]: ' + str(dicCnfg['varTr']))

    # Voxel resolution of fMRI data [mm]:
    dicCnfg['varVoxRes'] = float(dicCnfg['varVoxRes'])
    if lgcPrint:
        print('---Voxel resolution of fMRI data [mm]: '
              + str(dicCnfg['varVoxRes']))

    # Number of fMRI volumes and png files to load:
    dicCnfg['varNumVol'] = int(dicCnfg['varNumVol'])
    if lgcPrint:
        print('---Total number of fMRI volumes and png files: '
              + str(dicCnfg['varNumVol']))

    # Extent of temporal smoothing for fMRI data and pRF time course models
    # [standard deviation of the Gaussian kernel, in seconds]:
    # same temporal smoothing will be applied to pRF model time courses
    dicCnfg['varSdSmthTmp'] = float(dicCnfg['varSdSmthTmp'])
    if lgcPrint:
        print('---Extent of temporal smoothing (Gaussian SD in [s]): '
              + str(dicCnfg['varSdSmthTmp']))

    # Number of processes to run in parallel:
    dicCnfg['varPar'] = int(dicCnfg['varPar'])
    if lgcPrint:
        print('---Number of processes to run in parallel: '
              + str(dicCnfg['varPar']))

    # Size of space model in which the pRF models are
    # created (x- and y-dimension).
    dicCnfg['tplVslSpcSze'] = tuple([int(dicCnfg['varVslSpcSzeX']),
                                     int(dicCnfg['varVslSpcSzeY'])])
    if lgcPrint:
        print('---Size of visual space model (x & y): '
              + str(dicCnfg['tplVslSpcSze']))

    # Path(s) of functional data:
    dicCnfg['lstPathNiiFunc'] = ast.literal_eval(dicCnfg['lstPathNiiFunc'])
    if lgcPrint:
        print('---Path(s) of functional data:')
        for strTmp in dicCnfg['lstPathNiiFunc']:
            print('   ' + str(strTmp))

    # Path of mask (to restrict pRF model finding):
    dicCnfg['strPathNiiMask'] = ast.literal_eval(dicCnfg['strPathNiiMask'])
    if lgcPrint:
        print('---Path of mask (to restrict pRF model finding):')
        print('   ' + str(dicCnfg['strPathNiiMask']))

    # Output basename:
    dicCnfg['strPathOut'] = ast.literal_eval(dicCnfg['strPathOut'])
    if lgcPrint:
        print('---Output basename:')
        print('   ' + str(dicCnfg['strPathOut']))

    # Which version to use for pRF finding. 'numpy' or 'cython' for pRF finding
    # on CPU, 'gpu' for using GPU.
    dicCnfg['strVersion'] = ast.literal_eval(dicCnfg['strVersion'])
    if lgcPrint:
        print('---Version (numpy, cython, or gpu): '
              + str(dicCnfg['strVersion']))

    # Create pRF time course models?
    dicCnfg['lgcCrteMdl'] = (dicCnfg['lgcCrteMdl'] == 'True')
    if lgcPrint:
        print('---Create pRF time course models: '
              + str(dicCnfg['lgcCrteMdl']))

    # Path to npy file with pRF time course models (to save or laod). Without
    # file extension.
    dicCnfg['strPathMdl'] = ast.literal_eval(dicCnfg['strPathMdl'])
    if lgcPrint:
        print('---Path to npy file with pRF time course models (to save '
              + 'or load):')
        print('   ' + str(dicCnfg['strPathMdl']))

    # switch to determine which hrf functions should be used
    # 1: canonical, 2: can and temp derivative, 3: can, temp and spat deriv
    dicCnfg['switchHrfSet'] = ast.literal_eval(dicCnfg['switchHrfSet'])
    if lgcPrint:
        print('---Switch to determine which hrf functions should be used: '
              + str(dicCnfg['switchHrfSet']))

    # should model fitting be based on k-fold cross-validation?
    # if not, set to 1
    dicCnfg['varNumXval'] = ast.literal_eval(dicCnfg['varNumXval'])
    if lgcPrint:
        print('---Model fitting will have this number of folds for xval: '
              + str(dicCnfg['varNumXval']))

    # If we create new pRF time course models, the following parameters have to
    # be provided:
    if dicCnfg['lgcCrteMdl']:

        # Name of the npy that holds spatial info about conditions
        dicCnfg['strSptExpInf'] = ast.literal_eval(dicCnfg['strSptExpInf'])
        if lgcPrint:
            print('---Path to npy file with spatial condition info: ')
            print('   ' + str(dicCnfg['strSptExpInf']))

        # Name of the npy that holds temporal info about conditions
        dicCnfg['strTmpExpInf'] = ast.literal_eval(dicCnfg['strTmpExpInf'])
        if lgcPrint:
            print('---Path to npy file with temporal condition info: ')
            print('   ' + str(dicCnfg['strTmpExpInf']))

        # Factor by which time courses and HRF will be upsampled for the
        # convolutions
        dicCnfg['varTmpOvsmpl'] = ast.literal_eval(dicCnfg['varTmpOvsmpl'])
        if lgcPrint:
            print('---Factor by which time courses and HRF will be upsampled: '
                  + str(dicCnfg['varTmpOvsmpl']))

    # Is this a test?
    if lgcTest:

        # Prepend absolute path of this file to config file paths:
        dicCnfg['strPathNiiMask'] = (strDir + dicCnfg['strPathNiiMask'])
        dicCnfg['strPathOut'] = (strDir + dicCnfg['strPathOut'])
        dicCnfg['strPathMdl'] = (strDir + dicCnfg['strPathMdl'])
        dicCnfg['strSptExpInf'] = (strDir + dicCnfg['strSptExpInf'])
        dicCnfg['strTmpExpInf'] = (strDir + dicCnfg['strTmpExpInf'])

        # Loop through functional runs:
        varNumRun = len(dicCnfg['lstPathNiiFunc'])
        for idxRun in range(varNumRun):
            dicCnfg['lstPathNiiFunc'][idxRun] = (
                strDir
                + dicCnfg['lstPathNiiFunc'][idxRun]
                )

    return dicCnfg
