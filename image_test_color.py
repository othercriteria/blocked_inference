#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Testing application to image denoising.
# Adapting the idea to color images.
# Daniel Klein, 7/1/2011

import subprocess
import os

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import numpy.linalg as la

from distributions import MultivariateNormal
from em import em, kmeans, pi_maximize


# Parameters
image_file = 'applied_math.jpg'
image_rescale = 1
noise_var = 3340
noise_cov = -990
block_splits = 4
count_restart = 0.0
num_comps = 8
pi_max = False
show_summary = False
do_colormap = False
do_variance_viz = True


def main():
    # Load image
    im = Image.open(image_file).convert('RGB')
    width, height = im.size

    # Convenience function to build image band-by-band from array data
    def image_from_array(dat):
        bands = [Image.new('L', (width, height)) for n in range(3)]
        for i in range(3):
            bands[i].putdata(dat[:,i])
        return Image.merge('RGB', bands)

    # Resize image
    width, height = int(width / image_rescale), int(height / image_rescale)
    im = im.resize((width, height))

    # Summary image
    summary = Image.new('RGB', (width * 2 + 40, height * 2 + 60),
                        (255, 255, 255))
    draw = ImageDraw.Draw(summary)
    draw.text((5, height + 10), 'Original', fill = (0, 0, 0))
    draw.text((width + 25, height + 10),
              'Noise V = %.2f, C = %.2f' % (noise_var, noise_cov),
              fill = (0, 0, 0))
    draw.text((5, 2 * height + 40), 'Blocked Gamma', fill = (0, 0, 0))
    draw.text((width + 25, 2 * height + 40), 'Dists', fill = (0, 0, 0))
    del draw
    summary.paste(im, (10, 10))

    # Flatten to emissions
    real_emissions = list(im.getdata())
    num_data = len(real_emissions)
    real_emissions = np.array(real_emissions)

    # Block emissions
    width_blocks = np.array_split(np.arange(width), block_splits)
    height_blocks = np.array_split(np.arange(height), block_splits)
    idx = np.arange(num_data)
    idx.resize((height, width))
    blocks = []
    for hb in height_blocks:
        for wb in width_blocks:
            block = [idx[h, w] for h in hb for w in wb]
            blocks.append(np.array(block))

    # Generate noise
    v, c = noise_var, noise_cov
    cov = [[v, c, c], [c, v, c], [c, c, v]]
    noise = np.random.multivariate_normal([0, 0, 0], cov, width * height)
    noisy_emissions = real_emissions + noise

    # Generate noisy image
    noisy = image_from_array(noisy_emissions)
    summary.paste(noisy, (30 + width, 10))

    # Use K-means to initialize components
    results = kmeans(noisy_emissions, num_comps)
    init_gamma = results['best']
    means = results['means']

    # Analyze color space
    if do_colormap:
        col = { 'R': 0, 'G': 1, 'B': 2 }
        plt.figure()
        for i, (d, c1, c2) in enumerate([(real_emissions, 'R', 'G'),
                                         (real_emissions, 'R', 'B'),
                                         (real_emissions, 'G', 'B'),
                                         (noisy_emissions, 'R', 'G'),
                                         (noisy_emissions, 'R', 'B'),
                                         (noisy_emissions, 'G', 'B')]):
            plt.subplot(2, 3, i+1)
            plt.hexbin(d[:,col[c1]], d[:,col[c2]], gridsize=30,
                       extent = (0, 255, 0, 255))
            plt.plot(means[:,col[c1]], means[:,col[c2]], '.k')
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.axis([-20, 275, -20, 275])
        plt.savefig('image_test_color_colormap.png')
        plt.show()

    # Do EM
    results = em(noisy_emissions,
                 [MultivariateNormal() for n in range(num_comps)],
                 count_restart = count_restart,
                 blocks = blocks,
                 max_reps = 100,
                 init_gamma = init_gamma,
                 trace = True,
                 pi_max = pi_max)
    dists = results['dists']
    dists_trace = results['dists_trace']
    pi = results['pi']
    print 'Iterations: %(reps)d' % results

    gamma = np.transpose(results['gamma'])
    means = np.array([d.mean() for d in dists])
    covs = np.array([d.cov() for d in dists])

    # Reconstruct with blocked gamma
    rec_blocked_gamma = np.array([np.average(means, weights=g, axis=0)
                                  for g in gamma])
    im_blocked_gamma = image_from_array(rec_blocked_gamma)
    summary.paste(im_blocked_gamma, (10, 40 + height))

    # Reconstruct from distributions alone
    pi_opt = pi_maximize(noisy_emissions, dists)
    phi = np.empty((num_data, num_comps))
    for c in range(num_comps):
        phi[:,c] = dists[c].density(noisy_emissions)
    phi = np.matrix(phi)
    for i, pi in enumerate(pi_opt):
        phi[:,i] *= pi
    gamma_dists = phi / np.sum(phi, axis = 1)
    rec_dists = np.array(np.dot(gamma_dists, means))
    im_dists = image_from_array(rec_dists)
    summary.paste(im_dists, (30 + width, 40 + height))

    # Show summary image
    if show_summary:
        summary.show()
    summary.save('image_test_color_reconstruction.png')

    # Compare RMSE between reconstructions
    def rmse(x):
        return np.sqrt(np.mean((x - real_emissions) ** 2))
    print 'Raw MSE: %.1f' % rmse(noisy_emissions)
    print 'Blocked Gamma MSE: %.1f' % rmse(rec_blocked_gamma)
    print 'Dists MSE: %.1f' % rmse(rec_dists)

    # Visualize variance components
    if do_variance_viz:
        temp_files = []
        col = { 'R': 0, 'G': 1, 'B': 2 }
        fig = plt.figure()
        for i, (d, c1, c2) in enumerate([(real_emissions, 'R', 'G'),
                                         (real_emissions, 'R', 'B'),
                                         (real_emissions, 'G', 'B'),
                                         (noisy_emissions, 'R', 'G'),
                                         (noisy_emissions, 'R', 'B'),
                                         (noisy_emissions, 'G', 'B')]):
            ax = fig.add_subplot(2, 3, i+1)
            plt.hexbin(d[:,col[c1]], d[:,col[c2]], gridsize=30,
                       extent = (0, 255, 0, 255))
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.axis([-20, 275, -20, 275])
        for idx, dists in enumerate(dists_trace):
            ells = []
            for i, (d, c1, c2) in enumerate([(real_emissions, 'R', 'G'),
                                             (real_emissions, 'R', 'B'),
                                             (real_emissions, 'G', 'B'),
                                             (noisy_emissions, 'R', 'G'),
                                             (noisy_emissions, 'R', 'B'),
                                             (noisy_emissions, 'G', 'B')]):
                for dist in dists:
                    m, c = dist.mean(), dist.cov()
                    cm = (c[[col[c1], col[c2]]])[:,[col[c1], col[c2]]]
                    e, v = la.eigh(cm)
                    ell = Ellipse(xy = [m[col[c1]], m[col[c2]]],
                                  width = np.sqrt(e[0]),
                                  height = np.sqrt(e[1]),
                                  angle = (180.0 / np.pi) * np.arccos(v[0,0]))
                    ells.append(ell)
                    ax = fig.add_subplot(2, 3, i+1)
                    ax.add_artist(ell)
                    ell.set_clip_box(ax.bbox)
                    ell.set_alpha(0.9)
                    ell.set_facecolor(np.fmax(np.fmin(m / 255, 1), 0))
            file_name = 'tmp_%03d.png' % idx
            temp_files.append(file_name)
            plt.savefig(file_name, dpi = 100)
            for ell in ells:
                ell.remove()
        command = ('mencoder',
                   'mf://tmp_*.png',
                   '-mf',
                   'type=png:w=800:h=600:fps=5',
                   '-ovc',
                   'lavc',
                   '-lavcopts',
                   'vcodec=mpeg4',
                   '-oac',
                   'copy',
                   '-o',
                   'image_test_color_components.avi')
        os.spawnvp(os.P_WAIT, 'mencoder', command)
        for temp_file in temp_files:
            os.unlink(temp_file)

    # Find common variance components
    print 'True noise:'
    print cov
    chols = [la.cholesky(c) for c in covs]
    chol_recon = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if j > i: continue
            chol_recon[i,j] = np.Inf
            for chol in chols:
                if abs(chol[i,j]) < abs(chol_recon[i,j]):
                    chol_recon[i,j] = chol[i,j]
    cov_recon = np.dot(chol_recon, np.transpose(chol_recon))
    print 'Reconstructed noise:'
    print cov_recon

if __name__ == '__main__':
    main()
