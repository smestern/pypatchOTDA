#this script
# should be run as root I gues as biocmanger refuses to install as non root

# import rpy2's package module
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
#utils.chooseCRANmirror(ind=1) # select the first mirror in the list
# R package names
packnames = ('ggplot2', 'hexbin', 'fasno.franceschini')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# Import biocmanager and install
biocmanager = rpackages.importr('BiocManager')
biocmanager.install(lib='/home/smestern/R/x86_64-pc-linux-gnu-library/4.2')
biocmanager.install('GSAR', lib='/home/smestern/R/x86_64-pc-linux-gnu-library/4.2')

