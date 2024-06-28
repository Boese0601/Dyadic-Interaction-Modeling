import numpy as np
from scipy import linalg
from sklearn.cluster import KMeans
from scipy.stats import entropy

def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_variance(activations):
    return np.sum(np.var(activations, axis=0))

def calcuate_sid(gt, pred, type='exp'):
    # gt: list of [seq_len, dim]
    # pred: list of [seq_len, dim]
    if type == 'exp':
        k = 40
    else:
        k = 20
    merge_gt = np.concatenate(gt, axis=0)
    if type == 'exp':
        merge_gt = merge_gt[:, 6:]
    else:
        merge_gt = merge_gt[:, :6]
    # run kmeans on gt
    kmeans_gt = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(merge_gt)
    # run kmeans on pred
    merge_pred = np.concatenate(pred, axis=0)
    if type == 'exp':
        merge_pred = merge_pred[:, 6:]
    else:
        merge_pred = merge_pred[:, :6]
    kmeans_pred = kmeans_gt.predict(merge_pred)
    # compute histogram
    hist_cnt = [0] * k
    for i in range(len(kmeans_pred)):
        hist_cnt[kmeans_pred[i]] += 1
    hist_cnt = np.array(hist_cnt)
    hist_cnt = hist_cnt / np.sum(hist_cnt)
    # compute entropy
    entropy = 0
    eps = 1e-6
    for i in range(k):
        entropy += hist_cnt[i] * np.log2(hist_cnt[i]+eps)
    return -entropy
        
def sts(x, y, timestep=0.1):
    ans = 0
    total_sample, dim = x.shape
    for di  in range(dim):
        for i in range(1, total_sample):
            ans += ((x[i][di] - x[i-1][di]) - (y[i][di] - y[i-1][di]))**2 / timestep
    return np.sqrt(ans)
