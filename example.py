from massinference.angle import Angle
from massinference.catalog import SourceCatalogFactory, FastSampleHaloCatalogFactory, \
    MutableHaloMassCatalog
from massinference.distribution import MassPrior
from massinference.lenser import MapLenser
from massinference.lightcone import LightconeManager
from massinference.map import KappaMap, ShearMap
from massinference.plot import Limits

import cProfile

#TODO: remove this in future

# Define a toy inference experiment:

# Field center (RA, Dec, in degrees):

# Field size (arcmins):
limits = Limits(Angle.from_radian(-0.0261799387799), Angle.from_radian(-0.0244346095279),
                Angle.from_radian(-0.0261799387799), Angle.from_radian(-0.0244346095279))

# Source density (per sq arcmin):
source_density = 10.0

# Number of prior samples to "test":
simple_mc_samples = 4

sigma_e = 0.2
random_seed = 1

# Make a mock WL catalog, of observed, lensed, galaxy ellipticities:
source_factory = SourceCatalogFactory(limits, source_density, sigma_e)
source_catalog = source_factory.generate()
e1, e2 = MapLenser(KappaMap.default(), ShearMap.default()).lens(source_catalog)

# Set up a high-dimensional halo model, in the form of a catalog whose masses can be updated:
halo_catalog_factory = FastSampleHaloCatalogFactory(MutableHaloMassCatalog.default(limits),
                                                    MassPrior(), random_seed)

# Figure out which halos in the model are in front of which sources in the background:
lightcone_manager = LightconeManager(source_catalog, halo_catalog_factory)

# Compute the model-predicted shear at each source position, for each iteration of the halo maseses:

# result = lightcone_manager.run(simple_mc_samples)

# log_likelihood(g1, g2, e1, e2, sigma_e).to_csv('output.csv')

cProfile.run('lightcone_manager.run(simple_mc_samples)')


# from massinference.plot import Limits
# from massinference.map import KappaMap, ShearMap
# import matplotlib.pyplot as plt
#
# limits = Limits(1.8, 1.65, -2.0, -1.9)
# plot_config = KappaMap.default().plot(limits=limits)
# ShearMap.default().plot(plot_config=plot_config)
# plt.show()
