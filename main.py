import math

import numpy as np
import numpy.random
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform


def dummy(gaussian_mean, gaussian_var, gaussian_size, min_outlier=0, max_outlier=0):
    # Init vars
    errors = []
    estimateMean1 = 0
    gaussianVector = []

    # Create outliers
    outliers = range(min_outlier, max_outlier + 1)

    if min_outlier == 0 and max_outlier == 0:
        # not outliers
        outliers = []
        gaussianVector = np.random.normal(gaussian_mean, gaussian_var, gaussian_size)
        estimateMean1 = gaussianVector.mean()
        estimateVar1 = gaussianVector.var()
        errors = np.array([gaussian_mean - estimateMean1, gaussian_var - estimateVar1])

    for outlier in outliers:
        # Create random gaussian vector
        gaussianVector = np.random.normal(gaussian_mean, gaussian_var, gaussian_size)

        # Append outlier
        gaussianVectorWithOutlier = np.append(gaussianVector, outlier)

        # Compute the mean (via equation and via the direct mean)
        estimateMean = gaussianVectorWithOutlier.mean()

        errors = np.append(errors, gaussian_mean - estimateMean)

    return {'errors': errors, 'outliers': outliers}


def dummyUniform(a_parameter, b_parameter, size):
    # Create samples
    vectorUniformReal = np.random.uniform(a_parameter, b_parameter, size)

    # Compute parameters supossing gaussian
    meanGaussian = vectorUniformReal.mean()
    varGaussian = vectorUniformReal.var()

    # Create Gaussian distribution
    xGaussian = np.arange(a_parameter * 50, b_parameter * 50, 0.1)
    vectorGaussian = norm.pdf(xGaussian, loc=meanGaussian, scale=varGaussian)

    # Create Real Uniform distribution
    xUniform = np.arange(a_parameter * 1.2, b_parameter * 1.2, 0.1)
    vectorUniform = uniform.pdf(xUniform, loc=a_parameter, scale=b_parameter - a_parameter)

    return {'vectorUniformReal': vectorUniformReal, 'vectorUniform': vectorUniform, 'vectorGaussian': vectorGaussian, 'xUniform': xUniform, 'xGaussian': xGaussian}


if __name__ == '__main__':
    figNum = 0

    # First experiment, explaining the effect of outliers
    errors = np.zeros(201)
    outliers = dummy(1, 2, 100, -100, 100)['outliers']
    N = 100

    for i in range(N):
        errors = np.add(errors, dummy(1, 2, 100, -100, 100)['errors'])
    errors /= N

    plt.figure(figNum)
    figNum += 1
    plt.title("Error of the mean with outliers")
    plt.plot(outliers, errors)

    # Second experiment, explaining the convergence of the MLE estimator for the mean
    errors = []
    N = 2000
    for i in range(1, N + 1):
        errors = np.append(errors, dummy(1, 2, i)['errors'])

    errors = np.reshape(errors, (N, 2))

    plt.figure(figNum)
    figNum += 1
    plt.title("Mean evolution")
    plt.plot(errors[:, 0])

    plt.figure(figNum)
    figNum += 1
    plt.title("Var evolution")
    plt.plot(errors[:, 1])

    # Behavior as a normal distribution of the estimator
    errors = []
    N = 2000
    for i in range(1, N + 1):
        errors = np.append(errors, dummy(1, 2, 500)['errors'])

    errors = np.reshape(errors, (N, 2))

    fisherInfo = 500/2**2
    varEstimator = math.sqrt(1/fisherInfo)
    x = np.arange(-1, 1, 0.01)
    gaussianValuesEstimator = norm.pdf(x, loc=0, scale=varEstimator)
    print(gaussianValuesEstimator)

    plt.figure(figNum)
    figNum += 1
    plt.title("Normal distribution of the estimator")
    plt.plot(x, gaussianValuesEstimator)
    plt.hist(errors[:, 0], density=True, bins=30)

    # Third experiment, explaining what happens when wrong assumption of distribution is made
    distributions = []
    N = 2000
    distributions = dummyUniform(-10, 10, N)

    plt.figure(figNum)
    figNum += 1
    plt.title("Real histogram distribution")
    plt.hist(distributions['vectorUniformReal'], bins="auto")

    plt.figure(figNum)
    figNum += 1
    plt.title("Gaussian estimated distribution")
    plt.plot(distributions['xGaussian'], distributions['vectorGaussian'], 'b.')

    plt.figure(figNum)
    figNum += 1
    plt.title("Real distribution")
    plt.plot(distributions['xUniform'], distributions['vectorUniform'], 'b.')

    plt.show()
