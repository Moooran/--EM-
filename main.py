import numpy as np
import matplotlib.pyplot as plt

def generate_heights(N, mu_M=176, sigma_M=8, mu_F=164, sigma_F=6):
    N_M = int(N * 3 / 5)
    N_F = N - N_M  # 确保男女总人数为N

    male_heights = np.random.normal(mu_M, sigma_M, N_M)
    female_heights = np.random.normal(mu_F, sigma_F, N_F)

    return np.concatenate((male_heights, female_heights))


def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))


def plot_gaussians(data, mu_M, sigma_M, mu_F, sigma_F, iteration):
    x = np.linspace(min(data), max(data), 1000)
    g_M = gaussian(x, mu_M, sigma_M ** 2)
    g_F = gaussian(x, mu_F, sigma_F ** 2)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data')
    plt.plot(x, g_M, label=f'Male Gaussian\n$\mu={mu_M:.2f}$, $\sigma={sigma_M:.2f}$', color='blue')
    plt.plot(x, g_F, label=f'Female Gaussian\n$\mu={mu_F:.2f}$, $\sigma={sigma_F:.2f}$', color='red')
    plt.title(f'Iteration {iteration}')
    plt.xlabel('Height')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'EM_iteration_{iteration}.png')
    plt.close()


def em_algorithm(data, max_iter=10000, tol=1e-6, plot_interval=1000):
    # 初始化参数
    N = len(data)
    pi_M, pi_F = 0.6, 0.4
    mu_M, mu_F = np.mean(data) + 10, np.mean(data) - 10
    sigma_M, sigma_F = np.std(data), np.std(data)

    for iteration in range(max_iter):
        # E步
        gamma_M = pi_M * gaussian(data, mu_M, sigma_M ** 2)
        gamma_F = pi_F * gaussian(data, mu_F, sigma_F ** 2)
        gamma_sum = gamma_M + gamma_F
        gamma_M /= gamma_sum
        gamma_F /= gamma_sum

        # M步
        N_M = np.sum(gamma_M)
        N_F = np.sum(gamma_F)
        pi_M_new = N_M / N
        pi_F_new = N_F / N
        mu_M_new = np.sum(gamma_M * data) / N_M
        mu_F_new = np.sum(gamma_F * data) / N_F
        sigma_M_new = np.sqrt(np.sum(gamma_M * (data - mu_M_new) ** 2) / N_M)
        sigma_F_new = np.sqrt(np.sum(gamma_F * (data - mu_F_new) ** 2) / N_F)

        # 判断收敛
        if (np.abs(mu_M_new - mu_M) < tol and np.abs(mu_F_new - mu_F) < tol and
                np.abs(sigma_M_new - sigma_M) < tol and np.abs(sigma_F_new - sigma_F) < tol and
                np.abs(pi_M_new - pi_M) < tol and np.abs(pi_F_new - pi_F) < tol):
            print(f"迭代次数：{iteration+1}")
            break

        pi_M, pi_F = pi_M_new, pi_F_new
        mu_M, mu_F = mu_M_new, mu_F_new
        sigma_M, sigma_F = sigma_M_new, sigma_F_new

        # 每100次迭代绘制一次图像
        if iteration % plot_interval == 0:
            plot_gaussians(data, mu_M, sigma_M, mu_F, sigma_F, iteration)

    return pi_M, pi_F, mu_M, mu_F, sigma_M, sigma_F


if __name__ == '__main__':
    N = 10000
    data = generate_heights(N)

    pi_M, pi_F, mu_M, mu_F, sigma_M, sigma_F = em_algorithm(data)

    print(f"男性混合系数: {pi_M:.4f}, 女性混合系数: {pi_F:.4f}")
    print(f"男性均值: {mu_M:.4f}, 标准差: {sigma_M:.4f}")
    print(f"女性均值: {mu_F:.4f}, 标准差: {sigma_F:.4f}")
