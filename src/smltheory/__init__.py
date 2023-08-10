# read version from installed package
from importlib.metadata import version

__version__ = version("smltheory")

from smltheory.generate_data import *
from smltheory.overfitting import *
from smltheory.opt_error import *
from smltheory.est_error import *

from smltheory.least_squares import *
from smltheory.gradient_descent import *
from smltheory.trun_mvnt import *
from smltheory.bayes_decision import *

from smltheory.datasets import *

