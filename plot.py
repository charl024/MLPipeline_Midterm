import matplotlib.pyplot as plt
import numpy as np

kernels = ["Linear", "RBF", "Poly"]
pca_levels = ["PCA 50", "PCA 100", "PCA 200"]

import matplotlib.pyplot as plt
import numpy as np

kernels = ["Linear", "RBF", "Poly"]
pca_levels = ["PCA 50", "PCA 100", "PCA 200"]


# MNIST: TODO: redo this shit
svc_time_mnist = np.array([
    [5.05, 15.77, 5.41],
    [11.39, 35.27, 13.38],
    [21.86, 62.58, 27.19]
])

svc_error_mnist = np.array([
    [0.4534, 0.1889, 0.3633],
    [0.3768, 0.1649, 0.3060],
    [0.2929, 0.1459, 0.2716]
])

bag_time_mnist = np.array([
    [48.77, 128.05, 40.88],
    [97.26, 276.75, 113.45],
    [183.10, 484.71, 250.64]
])

bag_error_mnist = np.array([
    [0.1686, 0.0389, 0.0331],
    [0.1401, 0.0430, 0.0288],
    [0.0982, 0.0476, 0.0301]
])


# Fashion MNIST 
svc_time_fashion = np.array([
    [5.05, 15.77, 6.52],
    [11.36, 34.89, 26.89],
    [21.91, 62.65, 31.69]
])

svc_error_fashion = np.array([
    [0.4534, 0.1889, 0.3633],
    [0.3768, 0.1649, 0.2838],
    [0.2929, 0.1459, 0.2360]
])

bag_time_fashion = np.array([
    [43.22, 124.78, 43.34],
    [88.14, 274.14, 93.35],
    [169.56, 488.13, 191.27]
])

bag_error_fashion = np.array([
    [0.3659, 0.1589, 0.2906],
    [0.3331, 0.1521, 0.2736],
    [0.2598, 0.1456, 0.2274]
])

def plot3(title, data, ylabel, download=True):
    x = np.arange(len(pca_levels))
    width = 0.25

    plt.figure(figsize=(8, 5))

    for i, kernel in enumerate(kernels):
        plt.bar(x + (i - 1) * width, data[:, i], width, label=kernel)

    plt.xticks(x, pca_levels)
    plt.xlabel("PCA Components")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    if download == True:
        plt.savefig(f"figures/{title.replace(" ", "_")}.jpg")

    plt.show()

def plot4(title, svc_data, bag_data, ylabel, download=True): #TODO
    x = np.arange(len(pca_levels))
    width = 0.25

    for i, kernel in enumerate(kernels):
        plt.figure()

        plt.bar(x - width/2, svc_data[:, i], width, label="SVC")
        plt.bar(x + width/2, bag_data[:, i], width, label="Bagging")

        plt.xticks(x, pca_levels)
        plt.xlabel("PCA Components")
        plt.ylabel(ylabel)
        plt.title(f"{title} ({kernel} Kernel)")
        plt.legend()

        plt.tight_layout()

        if download == True:
            save_title = title + kernel
            save_title = save_title.replace(" ", "_")
            plt.savefig(f"figures/{save_title}.jpg")

        plt.show()
        


if __name__ == "__main__":
    # MNIST
    plot3("MNIST_SVC_Time", svc_time_mnist, "Time (s)")
    plot3("MNIST_SVC_Error", svc_error_mnist, "Error")

    # Fashion MNIST
    plot3("Fashion_SVC_Time", svc_time_fashion, "Time (s)")
    plot3("Fashion_SVC_Error", svc_error_fashion, "Error")

    # MNIST
    # plot4("MNIST Time Comparison", svc_time_mnist, bag_time_mnist, "Time (s)")
    # plot4("MNIST Error Comparison", svc_error_mnist, bag_error_mnist, "Error")

    # # Fashion MNIST
    # plot4("Fashion MNIST Time Comparison", svc_time_fashion, bag_time_fashion, "Time (s)")
    # plot4("Fashion MNIST Error Comparison", svc_error_fashion, bag_error_fashion, "Error")