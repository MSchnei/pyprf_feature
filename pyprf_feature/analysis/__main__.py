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
from pyprf_feature.analysis.pyprf_main import pyprf
from pyprf_feature.analysis.save_fit_tc_nii import save_tc_to_nii
from pyprf_feature import __version__


# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def main():
    """pyprf_feature entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]
    strWelcome = 'pyprf_feature ' + __version__
    strDec = '=' * len(strWelcome)
    print(strDec + '\n' + strWelcome + '\n' + strDec)

    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace - config file path:
    objParser.add_argument('-config',
                           metavar='config.csv',
                           help='Absolute file path of config file with \
                                 parameters for pRF analysis. Ignored if in \
                                 testing mode.'
                           )

    # Add argument to namespace -save_tc flag:
    objParser.add_argument('-save_tc', dest='save_tc',
                           action='store_true', default=False,
                           help='Save fitted and empirical time courses to \
                                 nifti file.')

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    # Get path of config file from argument parser:
    strCsvCnfg = objNspc.config

    # Print info if no config argument is provided.
    if strCsvCnfg is None:
        print('Please provide the file path to a config file, e.g.:')
        print('   pyprf_feature -config /path/to/my_config_file.csv')
    # If config file is provided, either perform fitting or recreate fitted
    # and empirical time courses depending on whether save_tc is True or False
    else:

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False

        if objNspc.save_tc:
            # Save fitted and empirical time courses to nifti file.
            # This assumes that fitting has already been run and will throw an
            # error if the resulting nii files of the fitting cannot be found.
            save_tc_to_nii(strCsvCnfg, lgcTest)

        else:
            # Call to main function, to invoke pRF fitting:
            pyprf(strCsvCnfg, lgcTest)


if __name__ == "__main__":
    main()
