'''
Simple example of 4 2D multivariate normals.
'''
import sys
import os 



# sys.path.append("/home/kyana/Documents/Austerweil/dpmm_understand/pgsm/pgsm")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# sys.path.append("../../")
from sklearn.metrics import homogeneity_completeness_v_measure

import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import seaborn as sb

from pgsm.distributions.mvn import MultivariateNormalDistribution
from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler
from pgsm.mcmc.concentration import GammaPriorConcentrationSampler
from pgsm.mcmc.particle_gibbs_split_merge import ParticleGibbsSplitMergeSampler
from pgsm.partition_priors import DirichletProcessPartitionPrior
from pgsm.mcmc.split_merge_setup import UniformSplitMergeSetupKernel

