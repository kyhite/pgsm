'''
Simple example of 4 2D multivariate normals.
'''
import sys, os
# import inspect
import termcolor
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from sklearn.metrics import homogeneity_completeness_v_measure

import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import seaborn as sb
import numba
# numba.config.DISABLE_JIT = True
from pgsm.distributions.mvn import MultivariateNormalDistribution
from pgsm.mcmc.collapsed_gibbs import CollapsedGibbsSampler
from pgsm.mcmc.concentration import GammaPriorConcentrationSampler
from pgsm.mcmc.particle_gibbs_split_merge import ParticleGibbsSplitMergeSampler
from pgsm.partition_priors import DirichletProcessPartitionPrior
from pgsm.mcmc.split_merge_setup import UniformSplitMergeSetupKernel
# numba.config.DISABLE_JIT = False

def plot_clustering(clustering, data, title):
    plot_df = pd.DataFrame(data, columns=['0', '1'])
    plot_df['cluster'] = clustering
    g = sb.lmplot(x='0', y='1', data=plot_df, hue='cluster', fit_reg=False)
    g.ax.set_title(title)
    pp.draw()

# local trace function which returns itself
def my_tracer(frame, event, arg = None):
    # extracts frame code
    code = frame.f_code
    # if
    # extracts calling function name
    func_name = code.co_name
    filename = code.co_filename
    relpath = f"{filename}"

    # extracts the line number
    line_no = frame.f_lineno
    if "/anaconda3/" not in relpath and "<" not in relpath:
        if  "update" in func_name:
            print(f"{func_name}  -  {termcolor.colored(relpath, 'green')}:{line_no}")
  
    return my_tracer



def print_info(pred_clustering, true_clustering, iteration):
    print( 'Iteration: {0}'.format(i))
    print( 'Number of cluster: {}'.format(len(np.unique(pred_clustering))))
    print( 'Homogeneity: {0}, Completeness: {1}, V-measure: {2}'.format(
        *homogeneity_completeness_v_measure(pred_clustering, true_clustering))
    )


def simulate_data(nun_data_points_per_cluster=100):
    mu = [[10, 10], [10, -10], [-10, 10], [-10, -10]]
    cov = np.eye(2)
    X = []
    Z = []
    for z, m in enumerate(mu):
        X.append(np.random.multivariate_normal(m, cov, size=nun_data_points_per_cluster))
        Z.append(z * np.ones(nun_data_points_per_cluster))
    X = np.vstack(X)
    Z = np.concatenate(Z).astype(int)
    return X, Z

np.random.seed(0)
datad = {        "stick_0": 0.92,
        "stick_1": 0.1,
        "stick_2": 0.5066666667,
        "stick_3": 0.3,
        "stick_4": 0.36,
        "stick_5": 0.9,
        "stick_6": 0.5466666667,
        "stick_7": 0.4666666667}


dist = MultivariateNormalDistribution(2)

partition_prior = DirichletProcessPartitionPrior(47)

gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)
data = np.array(list(datad.values()), dtype=np.ndarray).astype(np.float64)
data = np.atleast_2d(data).transpose()


setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)
pgsm_sampler = ParticleGibbsSplitMergeSampler.create_from_dist(dist, partition_prior, setup_kernel, num_anchors=2)

conc_sampler = GammaPriorConcentrationSampler(1, 1)
num_data_points = data.shape[0]
pred_clustering = np.zeros(num_data_points)


# print(vars(gibbs_sampler))
# print(gibbs_sampler.__dict__())
# print(partition_prior.__dict__())
# # print(gibbs_sampler.__dict__())
# print(setup_kernel.__dict__())
# print(setup_kernel)
# print(pgsm_sampler.__dict__())
# print(conc_sampler.__dict__)
def get_preds():
    global pred_clustering
    global data
    pred_clustering = pgsm_sampler.sample(pred_clustering, data)
    print("Pred?", pred_clustering)
    pred_clustering = gibbs_sampler.sample(pred_clustering, data)
    
    num_clusters = len(np.unique(pred_clustering))

    # partition_prior.alpha = conc_sampler.sample(partition_prior.alpha, num_clusters, num_data_points)
    print(partition_prior.alpha)

    print(pred_clustering)
    
    return pred_clustering


@numba.jit(numba.int64[::1]())
def get_one():
    # with numba.objmode(numba.int64[::1]):
    result = get_preds()
    result1 = np.ascontiguousarray(result).astype(np.int64)
    return result1


pred = get_preds()
# sys.settrace(my_tracer)
# # pred_clustering.
# for i in range(3):
#     pred = get_preds()
#     # get_one()
    

    # print(numba.typeof(pred))
    # pred_clustering = pgsm_sampler.sample(pred_clustering, data)
    # pred_clustering = gibbs_sampler.sample(pred_clustering, data)
    # num_clusters = len(np.unique(pred_clustering))
    # print(pred_clustering)
    # print("????")
    # print(partition_prior.alpha)
    # partition_prior.alpha = float(input("?"))
    # # if i % 10 == 0:
        # print("pred_clustering", pred_clustering)
        # print_info(pred_clustering, i)
    # pred_clustering = pgsm_sampler.sample(pred_clustering, data)
    # print("pred1", pred_clustering) # np.array
    # print(pred_clustering)
    # pred_clustering = gibbs_sampler.sample(pred_clustering, data)
    # # print("pred2", pred_clustering)
    # num_clusters = len(np.unique(pred_clustering))
    # partition_prior.alpha = conc_sampler.sample(partition_prior.alpha, num_clusters, num_data_points)
    # print(partition_prior.alpha)
# print(pred_clustering.shape)
# print(data.shape)
# plot_clustering(pred_clustering, data, 'Predicted clustering')
# plot_clustering(true_clustering, data, 'True clustering')
# pp.show()




# def _create_assignment_process(data, pid):
#     global collabsed_gibs_samplers
#     global pred_data
#     global pred_clusterings
#     global pgsm_samplers
#     global pred_data
#     global concentration_samplers
    

#     pred_data[pid] = data
#     num_data_points = data.shape[0]

#     dist = MultivariateNormalDistribution(2)
#     partition_prior = DirichletProcessPartitionPrior(100)
#     gibbs_sampler = CollapsedGibbsSampler(dist, partition_prior)
#     # data = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.ndarray).astype(np.float64)
#     # data = np.atleast_2d(data).transpose()

#     setup_kernel = UniformSplitMergeSetupKernel(data, dist, partition_prior)
#     pgsm_sampler = ParticleGibbsSplitMergeSampler.create_from_dist(dist, partition_prior, setup_kernel, num_anchors=2)

#     conc_sampler = GammaPriorConcentrationSampler(1, 1)
#     collabsed_gibs_samplers[pid] = gibbs_sampler
#     pgsm_samplers[pid] = pgsm_sampler
#     collabsed_gibs_samplers[pid] = conc_sampler
#     num_data_points = data.shape[0]
#     pred_clusterings[pid] = np.zeros(num_data_points)

