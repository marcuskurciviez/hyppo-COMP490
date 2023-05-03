# import numpy as np
# from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput
#
#
# def kernel_conditional_independence_test(x, y, z, alpha=0.05):
#     # Compute kernel matrices
#     Kx = np.exp(-np.square(np.linalg.norm(x[:, None, :] - x, axis=-1)) / (2 * np.median(np.abs(x - np.median(x)))))
#     Ky = np.exp(-np.square(np.linalg.norm(y[:, None, :] - y, axis=-1)) / (2 * np.median(np.abs(y - np.median(y)))))
#     Kz = np.exp(-np.square(np.linalg.norm(z[:, None, :] - z, axis=-1)) / (2 * np.median(np.abs(z - np.median(z)))))
#
#     # Compute test statistic
#     kcit = ("KCIT")
#     p_value = kcit.test(Kx, Ky, Kz).pvalue
#
#     # Determine statistical significance
#     if p_value < alpha:
#         return "Dependent"
#     else:
#         return "Independent"
#
#
# # Set random seed for reproducibility
# np.random.seed(123)
#
# # Generate three variables x, y, and z
# n = 100  # Sample size
# d = 2  # Number of variables
# x = np.random.normal(size=(n, d))
# y = np.random.normal(size=(n, d))
# z = np.random.normal(size=(n, d))
#
# # Create a linear dependence between x and y given z
# x[:, 0] = x[:, 0] + z[:, 0]
# y[:, 1] = y[:, 1] + z[:, 1]
#
# # Test for conditional independence between x and y given z
# result = kernel_conditional_independence_test(x, y, z)
# print(result)  # Should output "Dependent"
