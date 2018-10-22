#!/usr/bin/env python
"""Entry point for pyprf_opt_brute."""

import os
import argparse
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
    objParser.add_argument('-varNumOpt3', required=True, type=float,
                           metavar='N3',
                           help='Max displacement in radial direction.'
                           )

    # Add argument to namespace -lgcRstrCentre flag:
    objParser.add_argument('-lgcRstrCentre', dest='lgcRstrCentre',
                           action='store_true', default=False,
                           help='Restrict fitted models to stimulated area.')

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

        # Call to main function, to invoke pRF analysis:
        pyprf_opt_brute(objNspc.config, objNspc, lgcTest=lgcTest)


if __name__ == "__main__":
    main()
