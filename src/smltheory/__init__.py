# read version from installed package
from importlib.metadata import version

__version__ = version("smltheory")

#__all__ = ['bayes_decision', 'est_error', 'generate_data', 'gradient_descent',
#           'least_squares', 'opt_error', 'overfitting']
#
#from smltheory import *
