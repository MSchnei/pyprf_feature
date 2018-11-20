"""
Entry point.

References
----------
https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/

Notes
-----
Use config.py to set analysis parameters.
"""

import os
import argparse
import numpy as np
from pyprf_feature.analysis.load_config import load_config
from pyprf_feature.analysis.utils_general import cls_set_config, cmp_res_R2
from pyprf_feature.analysis.pyprf_main import pyprf
from pyprf_feature.analysis.save_fit_tc_nii import save_tc_to_nii
from pyprf_feature import __version__


# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


###### DEBUGGING ###############
#class Object(object):
#    pass
#objNspc = Object()
#objNspc.config = "/home/marian/Documents/Testing/pyprf_feature_devel/control/S02_config_motDepPrf_flck_smooth_inw.csv"
#objNspc.strPathHrf = None
#objNspc.supsur = [1.5, 1.8, 2.1]
#objNspc.save_tc = False
#objNspc.mdl_rsp = False
################################


def main():
    """pyprf_feature entry point."""

    # %% Print Welcome message

    strWelcome = 'pyprf_feature ' + __version__
    strDec = '=' * len(strWelcome)
    print(strDec + '\n' + strWelcome + '\n' + strDec)

    # %% Get list of input arguments

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace - config file path:
    objParser.add_argument('-config',
                           metavar='config.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis. Ignored if in \
                                 testing mode.'
                           )

    # Add argument to namespace -mdl_rsp flag:
    objParser.add_argument('-strPathHrf', default=None, required=False,
                           metavar='/path/to/custom_hrf_parameter.npy',
                           help='Path to npy file with custom hrf parameters. \
                           Ignored if in testing mode.')

    objParser.add_argument('-supsur', nargs='+',
                           help='List of floats that represent the ratio of \
                                 size neg surround to size pos center.',
                           type=float, default=None)

    # Add argument to namespace -save_tc flag:
    objParser.add_argument('-save_tc', dest='save_tc',
                           action='store_true', default=False,
                           help='Save fitted and empirical time courses to \
                                 nifti file. Ignored if in testing mode.')

    # Add argument to namespace -mdl_rsp flag:
    objParser.add_argument('-mdl_rsp', dest='lgcMdlRsp',
                           action='store_true', default=False,
                           help='When saving fitted and empirical time \
                                 courses, should fitted aperture responses be \
                                 saved as well? Ignored if in testing mode.')

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    # Get path of config file from argument parser:
    strCsvCnfg = objNspc.config

    # %% Decide which action to perform

    # If no config argument is provided, print info to user.
    if strCsvCnfg is None:
        print('Please provide the file path to a config file, e.g.:')
        print('   pyprf_feature -config /path/to/my_config_file.csv')

    # If config file is provided, either perform fitting or recreate fitted
    # and empirical time courses depending on whether save_tc is True or False
    else:

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False

        # If save_tc true, save fitted and empirical time courses to nifti file
        # This assumes that fitting has already been run and will throw an
        # error if the resulting nii files of the fitting cannot be found.
        if objNspc.save_tc:

            print('***Mode: Save fitted and empirical time courses***')
            if objNspc.lgcMdlRsp:
                print('    ***Also save fitted aperture responses***')

            # Call to function
            save_tc_to_nii(strCsvCnfg, lgcTest=lgcTest, lstRat=objNspc.supsur,
                           lgcMdlRsp=objNspc.lgcMdlRsp,
                           strPathHrf=objNspc.strPathHrf)

        # If save_tc false, perform pRF fitting, either with or without
        # suppressive surround
        else:

            # Perform pRF fitting without suppressive surround
            if objNspc.supsur is None:

                print('***Mode: Fit pRF models, no suppressive surround***')
                # Call to main function, to invoke pRF fitting:
                pyprf(strCsvCnfg, lgcTest, varRat=None,
                      strPathHrf=objNspc.strPathHrf)

            # Perform pRF fitting with suppressive surround
            else:
                print('***Mode: Fit pRF models, suppressive surround***')

                # Load config parameters from csv file into dictionary:
                dicCnfg = load_config(strCsvCnfg, lgcTest=lgcTest,
                                      lgcPrint=False)
                # Load config parameters from dictionary into namespace.
                # We do this on every loop so we have a fresh start in case
                # variables are redefined during the prf analysis
                cfg = cls_set_config(dicCnfg)
                # Make sure that lgcCrteMdl is set to True since we will need
                # to loop iteratively over pyprf_feature with different ratios
                # for size surround to size center. On every loop models,
                # reflecting the new ratio, need to be created from scratch
                errorMsg = 'lgcCrteMdl needs to be set to True for -supsur.'
                assert cfg.lgcCrteMdl, errorMsg

                # Make sure that switchHrf is set to 1. It would not make sense
                # to find the negative surround for the hrf deriavtive function
                errorMsg = 'switchHrfSet needs to be set to 1 for -supsur.'
                assert cfg.switchHrfSet == 1, errorMsg

                # Get list with size ratios
                lstRat = objNspc.supsur

                # Make sure that all ratios are larger than 1.0
                errorMsg = 'All provided ratios need to be larger than 1.0'
                assert np.all(np.greater(np.array(lstRat), 1.0)), errorMsg

                # Append None as the first entry, so fitting without surround
                # is performed once as well
                lstRat.insert(0, None)

                # Loop over ratios and find best pRF
                for varRat in lstRat:

                    # Print to command line, so the user knows which exponent
                    # is used
                    print('---Ratio surround to center: ' + str(varRat))
                    # Call to main function, to invoke pRF analysis:
                    pyprf(strCsvCnfg, lgcTest=lgcTest, varRat=varRat,
                          strPathHrf=objNspc.strPathHrf)

                # List with name suffices of output images:
                lstNiiNames = ['_x_pos',
                               '_y_pos',
                               '_SD',
                               '_R2',
                               '_polar_angle',
                               '_eccentricity',
                               '_Betas']

                # Compare results for the different ratios, export nii files
                # based on the results of the comparison and delete in-between
                # results

                # Replace first entry (None) with 0, so it can be saved to nii
                lstRat[0] = 0.0
                # Append 'hrf' to cfg.strPathOut, if fitting was done with
                # custom hrf
                if objNspc.strPathHrf is not None:
                    cfg.strPathOut = cfg.strPathOut + '_hrf'

                cmp_res_R2(lstRat, lstNiiNames, cfg.strPathOut, cfg.strPathMdl,
                           lgcDel=True)


if __name__ == "__main__":
    main()
