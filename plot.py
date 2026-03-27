import matplotlib.pyplot as plt
import numpy as np

kernels = ["Linear", "RBF", "Poly"]
pca_levels = ["PCA 50", "PCA 100", "PCA 200"]

import matplotlib.pyplot as plt
import numpy as np

kernels = ["Linear", "RBF", "Poly"]
pca_levels = ["PCA 50", "PCA 100", "PCA 200"]


# LDA MNIST
svc_time_mnist_lda = np.array([
    [5.85, 7.16, 5.37],
    [13.20, 16.81, 13.22],
    [24.73, 32.11, 27.15],
    [5.23 , 8.12 , 5.94]
])

svc_error_mnist_lda = np.array([
    [0.2466, 0.0289, 0.0356],
    [0.1780, 0.0248, 0.0302],
    [0.1433, 0.0265, 0.0259],
    [0.1726, 0.0778, 0.1725]
])

# LDA Fashion

svc_time_fashion_lda = np.array([
    [5.05, 15.77, 6.52],
    [11.36, 34.89, 26.89],
    [21.91, 62.65, 31.69],
    [4.03, 7.16, 5.94 ]
])

svc_error_fashion_lda = np.array([
    [0.4534, 0.1889, 0.3633],
    [0.3768, 0.1649, 0.2838],
    [0.2929, 0.1459, 0.2360],
    [0.2688, 0.2704, 0.1725]
])

# update, updated the times on the tables with info from the overleaf tables
svc_time_mnist = np.array([
    [5.85, 7.16, 5.37],
    [13.20, 16.81, 13.22],
    [24.73, 32.11, 27.15]
])

svc_error_mnist = np.array([
    [0.2466, 0.0289, 0.0356],
    [0.1780, 0.0248, 0.0302],
    [0.1433, 0.0265, 0.0259]
])

bag_time_mnist = np.array([
    [57.71, 65.71, 46.47],
    [115.08, 138.47, 106.37],
    [188.11, 232.19, 190.31]
])

bag_error_mnist = np.array([
    [0.1748, 0.0298, 0.0330],
    [0.1272, 0.0251, 0.0261],
    [0.0861, 0.0256, 0.0257]
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

def plot3_lda(title, data, ylabel, download=True):
    reductions = pca_levels + ["LDA"]
    x = np.arange(len(reductions))
    width = 0.25

    plt.figure(figsize=(8, 5))

    for i, kernel in enumerate(kernels):
        plt.bar(x + (i - 1) * width, data[:, i], width, label=kernel)

    plt.xticks(x, reductions)
    plt.xlabel("Reduction")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.tight_layout()

    if download == True:
        plt.savefig(f"figures/{title.replace(" ", "_") + "LDA"}.jpg")

    plt.show()

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
    # # MNIST
    # plot3("MNIST_SVC_Time", svc_time_mnist, "Time (s)")
    # plot3("MNIST_SVC_Error", svc_error_mnist, "Error")

    # # Fashion MNIST
    # plot3("Fashion_SVC_Time", svc_time_fashion, "Time (s)")
    # plot3("Fashion_SVC_Error", svc_error_fashion, "Error")

    # MNIST
    # plot4("MNIST Time Comparison", svc_time_mnist, bag_time_mnist, "Time (s)")
    # plot4("MNIST Error Comparison", svc_error_mnist, bag_error_mnist, "Error")

    # # # Fashion MNIST
    # plot4("Fashion MNIST Time Comparison", svc_time_fashion, bag_time_fashion, "Time (s)")
    # plot4("Fashion MNIST Error Comparison", svc_error_fashion, bag_error_fashion, "Error")

    # MNIST
    plot3_lda("MNIST_SVC_Time", svc_time_mnist_lda, "Time (s)")
    plot3_lda("MNIST_SVC_Error", svc_error_mnist_lda, "Error")

    # Fashion MNIST
    plot3_lda("Fashion_SVC_Time", svc_time_fashion_lda, "Time (s)")
    plot3_lda("Fashion_SVC_Error", svc_error_fashion_lda, "Error")