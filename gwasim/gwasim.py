# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import pairwise_distances

class GWASim(object):
    """ Generates Synthetic GWAS Data

    ## USAGE EXAMPLE

    We will generate a GWAS dataset with two phenotype elements:

        - Population structure
        - Some case/control status

    ```
    import numpy as np
    import matplotlib.pyplot as plt

    g = GWASim(f=[falpine1, falpine1],          # Evolve both phenotypes on the Alpine1 function
               bounds=[[-1, 1],[-2, 2]],        # Initialization bounds for population between -1 and 1, and case/control between -2, 2
               nsamples=1000,                   # Generate 1000 subjects
               ndim=[1000, 2],                  # Population structure is an evolution on 1000dimensions of genome. Case control evolved on 2 dims (i.e. 998/1000 of signal will be population structure)
               pclass=[[1.], [1/2, 1/2]],       # Population structure is evolved from a common origin (pclass=1), Case/Control evolved from 2 points, with 50% of sample in each
               sigmas=[0.01, 0.01],             # Step sizes for each evolution
               ngenerations=100,                # Evolve over 100 generations
               metric='euclidean',              # Use euclidean distance metric
               thresholds=[0.5, 0.75])          # Thresholds for making genotypes 0, 1, 2

    X, y = g.generate() # Create the genotypes

    plt.imshow(g.X) # Show genotype matrix

    u,s,v = np.linalg.svd(g.X) # PCA population
    plt.scatter(u[:,0], u[:,1], c=g.y[1])
    ```

    """
    def __init__(self,
                 f,
                 bounds,
                 nsamples=None,
                 ndim=None,
                 pclass=None,
                 sigmas=None,
                 ngenerations=None,
                 metric='euclidean',
                 thresholds=[0.5, 0.75],
                 rng=np.random.RandomState()):
        """
        Arguments:

            f: `list` of length `nphenotypes`. Objective functions to use for each phenotype
            bounds: `list` of `nphenotypes` lists of length `2`. Lower and upper bounds for initialization on objective functions.
            nsamples: `int`. Number of subjects being simulated (if `None`, will be 50).
            ndim: `list` of length `nphenotypes`. Number of dimensions over which each phenotype's underlying genotype was evolved (if `None`, will be 100 for each phenotype).
            pclass: `list` of length `nphenotypes` consisting of sublists of probabilities (each must sum to 1). Class proportions
            sigmas: `list` of length `nphenotypes` or `float`. Step sizes for genotype evolution (if set to `None`, will be 0.1 for all evolution series).
            ngenerations: `list` of length `nphenotypes`. Number of evolution steps (if `None`, will be 50 for each phenotype)
            metric: `str`. Distance metric to use during evolution
            thresholds: `str`. Thresholds for setting genotype values to 0, 1, 2.

        """
        self.nphenotypes = len(f)
        self.f = f
        self.bounds = bounds
        self.nsamples=nsamples,
        self.ndim = ndim
        self.pclass = pclass
        self.nperclass = []
        self.sigmas = sigmas
        self.ngenerations = ngenerations
        self.metric = metric
        self.thresholds = thresholds
        self.rng = rng

        if nsamples is None:
            self.nsamples = 50

        if ndim is None:
            self.ndim = (np.ones(self.nphenotypes)*100).astype(np.int)

        if ngenerations is None:
            self.ngenerations = (np.ones(self.nphenotypes)*50).astype(np.int)
        elif (type(ngenerations) is float) or (type(ngenerations) is int):
            self.ngenerations = (np.ones(self.nphenotypes)*np.abs(ngenerations)).astype(np.int)

        if pclass is None:
            self.pclass = []
            for i in range(self.nphenotypes):
                self.pclass.append([0.5, 0.5])
        else:
            for i in range(self.nphenotypes):
                if np.sum(self.pclass[i]) != 1:
                    raise ValueError('Class proportions must all sum to 1.')

        # Change pclass into number of subjects per class
        for i in range(self.nphenotypes):
            if len(self.pclass[i]) == 1:
                self.nperclass.append(np.array(self.nsamples))
            else:
                ni = np.floor(np.asarray(self.pclass[i])*self.nsamples)
                ni[-1] = self.nsamples - np.sum(ni[:-1])
                self.nperclass.append(ni.astype(np.int))

        if sigmas is None:
            self.sigmas = np.ones(self.nphenotypes)*0.1
        elif (type(sigmas) is float) or (type(sigmas) is int):
            self.sigmas = np.ones(self.nphenotypes)*np.abs(sigmas)

    def scale_threshold(self, X):
        n, m = X.shape

        # Scale to 0-1
        xmin = np.tile(np.min(X, 0), [n, 1])
        X = X-xmin
        xmax = np.tile(np.max(X, 0), [n, 1])
        X = X/xmax
        X = 2*(X**2)

        # Threshold
        X[np.less(X, 2*self.thresholds[0])] = 0
        X[np.logical_and(np.greater(X, 2*self.thresholds[0]), np.less(X, 2*self.thresholds[1]))] = 1
        X[np.greater(X, 2*self.thresholds[1])] = 2

        return X

    def generate(self, returnxy=True):
        """ Creates the overall genotype and phenotype matrices

        Arguments:

            returnxy: `bool`. Whether to return the X and y at function call

        Returns:

            X: `ndarray((nsamples, SUM(ndim[i] for i=1:nphenotypes)))`. Genotype matrix
            y: `list` of length `nphenotypes` containing the class ids for each phenotype
        """
        self.X = []
        self.y = []
        for i in range(self.nphenotypes):
            lb = self.bounds[i][0]
            ub = self.bounds[i][1]
            nclasses = len(self.nperclass[i])
            Xp = []
            self.y.append(np.hstack(np.ones(self.nperclass[i][j])*j for j in range(nclasses)))
            for j in range(nclasses):
                xinit = self.rng.uniform(lb, ub, size=(self.nperclass[i][j], self.ndim[i]))
                Xpj = self.evolve_genotype(f=self.f[i],
                                           ndim=self.ndim[i],
                                           mu=self.nperclass[i][j],
                                           x=xinit,
                                           sigma=self.sigmas[i],
                                           ngenerations=self.ngenerations[i],
                                           metric=self.metric)
                Xp.append(Xpj)
            Xp = np.vstack(Xp)
            Xp = self.scale_threshold(Xp)
            self.X.append(Xp)
        self.X = np.hstack(self.X)

        if returnxy:
            return self.X, self.y

    def evolve_genotype(self,
                         f,
                         ndim,
                         mu=1000,
                         x=None,
                         sigma=1,
                         ngenerations=50,
                         metric='euclidean'):
        """ Evolves genotypes """
        if x is None:
            x = self.rng.normal(0, sigma, size=(int(mu), ndim))
        else:
            if np.ndim(x) == 1:
                x = np.tile(np.expand_dims(x, 0), [int(mu), 1])

        X = x + sigma*self.rng.normal(0, 1, size=(int(mu),ndim))
        y = np.ones(X.shape[0])
        i = np.arange(X.shape[0])
        done = False; niter=0;
        while not done:
            niter += 1
            D = pairwise_distances(X, X, metric=metric)
            j = np.argmin(D + 100*np.max(D)*np.eye(D.shape[0]), axis=1)
            w = y[i] + y[j]
            w = np.tile((y[i]/w).reshape(-1, 1), [1, X.shape[1]])
            X_ = (w*X + (1-w)*X[j]) + sigma*self.rng.normal(0, 1, size=X.shape)
            X = np.vstack((X, X_))
            y = f(X)
            i = np.argsort(y)[:mu]
            X = X[i]
            if niter >= ngenerations:
                done = True

        return X
