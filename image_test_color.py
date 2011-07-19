#!/usr/bin/env python

# Experiments with EM estimation on HMM data.
# Testing application to image denoising.
# Adapting the idea to color images.
# Daniel Klein, 7/1/2011

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from distributions import MultivariateNormal
from em import em, kmeans


# Parameters
image_file = 'starry_night.jpg'
image_rescale = 5
noise_var = 2340
noise_cov = -790
block_splits = 1
count_restart = 0.0
num_comps = 20
do_colormap = False

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
    draw.text((5, 2 * height + 40), 'Argmax', fill = (0, 0, 0))
    draw.text((width + 25, 2 * height + 40), 'Average', fill = (0, 0, 0))
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
        plt.show()

    # Do EM
    results = em(noisy_emissions,
                 [MultivariateNormal() for n in range(num_comps)],
                 count_restart = count_restart,
                 blocks = blocks,
                 max_reps = 20,
                 init_gamma = init_gamma)
    dists = results['dists']
    pi = results['pi']
    print 'Iterations: %(reps)d' % results

    gamma = np.transpose(results['gamma'])
    means = np.array([d.mean() for d in dists])
    covs = np.array([d.cov() for d in dists])

    # Reconstruct with argmax
    rec_argmax = means[np.argmax(gamma, axis=1)]
    im_argmax = image_from_array(rec_argmax)
    summary.paste(im_argmax, (10, 40 + height))

    # Reconstruct with weighted average
    rec_avg = np.array([np.average(means, weights=g, axis=0) for g in gamma])
    im_avg = image_from_array(rec_avg)
    summary.paste(im_avg, (30 + width, 40 + height))

    # Show summary image
    summary.show()
    summary.save('image_test_color.png')

    # Compare RMSE between reconstructions
    def rmse(x):
        return np.sqrt(np.mean((x - real_emissions) ** 2))
    print 'Raw MSE: %.1f' % rmse(noisy_emissions)
    print 'ArgMax MSE: %.1f' % rmse(rec_argmax)
    print 'Avg MSE: %.1f' % rmse(rec_avg)

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
