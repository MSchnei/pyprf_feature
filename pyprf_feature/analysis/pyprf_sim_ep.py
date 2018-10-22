#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Entry point for pyprf_sim_ep."""

import os
import argparse
from pyprf_feature.analysis.pyprf_sim import pyprf_sim
from pyprf_feature import __version__

# Get path of this file:
strDir = os.path.dirname(os.path.abspath(__file__))


def get_arg_parse():
    """Parses the Command Line Arguments using argparse."""
    # Create parser object:
    objParser = argparse.ArgumentParser()

    # Add argument to namespace -strCsvPrf results file path:
    objParser.add_argument('-strCsvPrf', required=True,
                           metavar='/path/to/my_prior_res',
                           help='Absolute file path of prior pRF results. \
                                 Ignored if in testing mode.'
                           )

    # Add argument to namespace -strStmApr results file path:
    objParser.add_argument('-strStmApr', required=True,
                           metavar='/path/to/my_prior_res',
                           help='Absolute file path to npy file with \
                                 stimulus apertures. Ignored if in testing \
                                 mode.'
                           )

    # Add argument to namespace -lgcNoise flag:
    objParser.add_argument('-lgcNoise', dest='lgcNoise',
                           action='store_true', default=False,
                           help='Should noise be added to the simulated pRF\
                                 time course?')

    # Add argument to namespace -lgcRtnNrl flag:
    objParser.add_argument('-lgcRtnNrl', dest='lgcRtnNrl',
                           action='store_true', default=False,
                           help='Should neural time course, unconvolved with \
                                 hrf, be returned as well?')

    objParser.add_argument('-supsur', nargs='+',
                           help='List of floats that represent the ratio of \
                                 size neg surround to size pos center.',
                           type=float, default=None)

    # Namespace object containign arguments and values:
    objNspc = objParser.parse_args()

    return objNspc


def main():
    """pyprf_sim entry point."""
    # Get list of input arguments (without first one, which is the path to the
    # function that is called):  --NOTE: This is another way of accessing
    # input arguments, but since we use 'argparse' it is redundant.
    # lstArgs = sys.argv[1:]
    strWelcome = 'pyprf_sim ' + __version__
    strDec = '=' * len(strWelcome)
    print(strDec + '\n' + strWelcome + '\n' + strDec)

    objNspc = get_arg_parse()

    # Print info if no config argument is provided.
    if any(item is None for item in [objNspc.strCsvPrf, objNspc.strStmApr]):
        print('Please provide necessary file paths, e.g.:')
        print('   pyprf_sim -strCsvPrf /path/to/my_config_file.csv')
        print('             -strStmApr /path/to/my_stim_apertures.npy')

    else:

        # Signal non-test mode to lower functions (needed for pytest):
        lgcTest = False

        # Call to main function, to invoke pRF analysis:
        pyprf_sim(objNspc.strCsvPrf, objNspc.strStmApr, lgcTest=lgcTest,
                  lgcNoise=objNspc.lgcNoise, lgcRtnNrl=objNspc.lgcRtnNrl,
                  lstRat=objNspc.supsur)


if __name__ == "__main__":
    main()
