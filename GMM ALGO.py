# part 1 _20points
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
#the seed value of 42 ensures that the random numbers generated for the data points' coordinates and other random operations (if any)
# will be the same each time you run the code.
np.random.seed(42)

# Parameters
num_points_group1 = 700
num_points_group2 = 300
num_features = 2

# Group 1
mean_group1 = [-1, -1]
covariance_group1 = [[0.8, 0], [0, 0.8]]
data_group1 = np.random.multivariate_normal(mean_group1, covariance_group1, num_points_group1)

# Group 2
mean_group2 = [1, 1]
covariance_group2 = [[0.75, -0.2], [-0.2, 0.6]]
data_group2 = np.random.multivariate_normal(mean_group2, covariance_group2, num_points_group2)

# Plotting
plt.scatter(data_group1[:, 0], data_group1[:, 1], label='Group 1',c='red')
plt.scatter(data_group2[:, 0], data_group2[:, 1], label='Group 2',c='blue')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data')
plt.legend()
plt.show()

# part 2/ 80 points
def gaussian(x, mean, cov):
    # Compute the Gaussian probability density function for a given data point
    d = x.shape[0]
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    coeff = 1 / ((2 * np.pi) ** (d/2) * np.sqrt(np.linalg.det(cov)))
    return coeff * np.exp(exponent)

def gmm(data, n_clusters, max_iterations=100, epsilon=1e-6):
    n_samples, n_features = data.shape

    # Initialize the cluster means, covariances, and mixing coefficients
    means = np.random.randn(n_clusters, n_features)
    covariances = [np.eye(n_features)] * n_clusters
    mixing_coeffs = np.ones(n_clusters) / n_clusters

    for iteration in range(max_iterations):
        # Expectation-step: Compute the responsibilities
        responsibilities = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            for i in range(n_samples):
                responsibilities[i, k] = mixing_coeffs[k] * gaussian(data[i], means[k], covariances[k])

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        # Maximization-step: Update the parameters
        prev_means = np.copy(means)
        prev_covariances = covariances.copy()
        prev_mixing_coeffs = mixing_coeffs.copy()

        for k in range(n_clusters):
            # Update cluster mean
            means[k] = np.sum(responsibilities[:, k].reshape(-1, 1) * data, axis=0) / np.sum(responsibilities[:, k])

            # Update cluster covariance
            diff = data - means[k]
            covariances[k] = np.dot((responsibilities[:, k].reshape(-1, 1) * diff).T, diff) / np.sum(responsibilities[:, k])

            # Update mixing coefficient
            mixing_coeffs[k] = np.mean(responsibilities[:, k])

        # Check for convergence
        if np.allclose(prev_means, means, atol=epsilon) and \
           np.allclose(prev_covariances, covariances, atol=epsilon) and \
           np.allclose(prev_mixing_coeffs, mixing_coeffs, atol=epsilon):
            print("Algorithm converged.")
            break

    predicted_labels = np.argmax(responsibilities, axis=1)
    return means, covariances, mixing_coeffs, predicted_labels

data = np.concatenate((data_group1, data_group2), axis=0)

n_clusters = 2
means, covariances, mixing_coeffs, predicted_labels = gmm(data, n_clusters)

# Print the cluster means, covariances, and mixing coefficients
for k in range(n_clusters):
    print(f"Cluster {k+1}:")
    print("Mean:", means[k])
    print("Covariance:")
    print(covariances[k])
    print("Mixing Coefficient:", mixing_coeffs[k])

print("Predicted Labels:", predicted_labels)