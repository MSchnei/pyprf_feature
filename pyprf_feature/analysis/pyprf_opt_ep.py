#!/usr/bin/env python
"""Entry point for pyprf_opt_brute."""

import os
import argparse
import numpy as np
from pyprf_feature.analysis.load_config import load_config
from pyprf_feature.analysis.utils_general import cls_set_config, cmp_res_R2
from pyprf_feature.analysis.pyprf_opt_brute import pyprf_opt_brute
from pyprf_feature import __version__

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def get_arg_parse():
    """Parses the Command Line Arguments using argparse."""
    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace -config file path:
    objParser.add_argument('-config', required=True,
                           metavar='/path/to/config.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis. Ignored if in \
                                 testing mode.'
                           )

    # Add argument to namespace -prior results file path:
    objParser.add_argument('-strPthPrior', required=True,
                           metavar='/path/to/my_prior_res',
                           help='Absolute file path of prior pRF results. \
                                 Ignored if in testing mode.'
                           )

    # Add argument to namespace -varNumOpt1 flag:
    objParser.add_argument('-varNumOpt1', required=True, type=int,
                           metavar='N1',
                           help='Number of radial positions.'
                           )

    # Add argument to namespace -varNumOpt2 flag:
    objParser.add_argument('-varNumOpt2', required=True, type=int,
                           metavar='N2',
                           help='Number of angular positions.'
                           )

    # Add argument to namespace -varNumOpt3 flag:
    objParser.add_argument('-varNumOpt3', default=None, metavar='N3',
                           help='Max displacement in radial direction.'
                           )

    # Add argument to namespace -lgcRstrCentre flag:
    objParser.add_argument('-lgcRstrCentre', dest='lgcRstrCentre',
                           action='store_true', default=False,
                           help='Restrict fitted models to stimulated area.')

    objParser.add_argument('-strPathHrf', default=None, required=False,
                           metavar='/path/to/custom_hrf_parameter.npy',
                           help='Path to npy file with custom hrf parameters. \
                           Ignored if in testing mode.')

    objParser.add_argument('-supsur', nargs='+',
                           help='List of floats that represent the ratio of \
                                 size neg surround to size pos center.',
                           type=float, default=None)

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    return objNspc


def main():
    """pyprf_opt_brute entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]
    strWelcome = 'pyprf_opt_brute ' + __version__
    strDec = '=' * len(strWelcome)
    print(strDec + '\n' + strWelcome + '\n' + strDec)

    objNspc = get_arg_parse()

    # Print info if no config argument is provided.
    if any(item is None for item in [objNspc.config, objNspc.strPthPrior,
                                     objNspc.varNumOpt1, objNspc.varNumOpt2]):
        print('Please provide the necessary input arguments, i.e.:')
        print('-strCsvCnfg -strPthPrior -varNumOpt1 and -varNumOpt2')

    else:

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False

        # Perform pRF fitting without suppressive surround
        if objNspc.supsur is None:
            print('***Mode: Fit pRF models, no suppressive surround***')

            # Call to main function, to invoke pRF analysis:
            pyprf_opt_brute(objNspc.config, objNspc, lgcTest=lgcTest,
                            strPathHrf=objNspc.strPathHrf, varRat=None)

        # Perform pRF fitting with suppressive surround
        else:
            print('***Mode: Fit pRF models, suppressive surround***')

            # Load config parameters from csv file into dictionary:
            dicCnfg = load_config(objNspc.config, lgcTest=lgcTest,
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
                pyprf_opt_brute(objNspc.config, objNspc, lgcTest=lgcTest,
                                strPathHrf=objNspc.strPathHrf, varRat=varRat)

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

            # Replace first entry (None) with 1, so it can be saved to nii
            lstRat[0] = 1.0

            # Append 'hrf' to cfg.strPathOut, if fitting was done with
            # custom hrf
            if objNspc.strPathHrf is not None:
                cfg.strPathOut = cfg.strPathOut + '_hrf'

            cmp_res_R2(lstRat, lstNiiNames, cfg.strPathOut, cfg.strPathMdl,
                       lgcDel=True, lgcSveMdlTc=False, strNmeExt='_brute')


if __name__ == "__main__":
    main()
