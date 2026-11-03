import os
import numpy as np
import matplotlib.pyplot as plt

def hooke_jeeves(func, x0, dx, q=2.0, p=2.0, eps1=1e-6, eps2=1e-6, max_iter=10000):
    """
    Реалізація методу нульового порядку Хука-Дживса
    """
    x_base = np.array(x0, dtype=float)
    step = np.array(dx, dtype=float)
    n = len(x_base)
    
    trajectory = [x_base.copy()]
    
    def exploratory_search(x_center, current_step):
        x_curr = x_center.copy()
        for i in range(n):
            f_center = func(x_curr)
            # Пробуємо крок вперед
            x_curr[i] += current_step[i]
            if func(x_curr) < f_center:
                continue
            # Пробуємо крок назад
            x_curr[i] -= 2.0 * current_step[i]
            if func(x_curr) < f_center:
                continue
            # Якщо успіху немає, повертаємо назад
            x_curr[i] += current_step[i]
        return x_curr

    x_curr = exploratory_search(x_base, step)
    trajectory.append(x_curr.copy())
    
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        if func(x_curr) < func(x_base):
            # Пошук за зразком
            x_pattern = x_curr + p * (x_curr - x_base)
            x_new = exploratory_search(x_pattern, step)
            
            x_base = x_curr.copy()
            x_curr = x_new.copy()
            trajectory.append(x_curr.copy())
        else:
            # Зменшення кроку
            if np.all(step < eps1):
                break
            step /= q
            x_curr = exploratory_search(x_base, step)
            trajectory.append(x_curr.copy())
            
        # Умови закінчення пошуку
        if np.linalg.norm(step) < eps1 or abs(func(x_curr) - func(x_base)) < eps2:
            if np.all(step < eps1):
                break
                
    return x_curr, func(x_curr), trajectory

#  Функція Розенброка для тестування 
def rosenbrock(X):
    return 100.0 * (X[0]**2 - X[1])**2 + (X[0] - 1.0)**2

# Система нелінійних рівнянь 
def f1(X):
    return X[0]**2 + X[1] - 2.0

def f2(X):
    return X[0] + X[1]**2 - 2.0

def objective_phi(X):
    """Цільова функція як сума квадратів рівнянь системи"""
    return f1(X)**2 + f2(X)**2

if __name__ == "__main__":
    
    print("\nТестування програми на функції Розенброка")
    x0_rosen = [-1.2, 0.0]
    dx_rosen = [0.5, 0.5]
    res_rosen, val_rosen, traj_rosen = hooke_jeeves(rosenbrock, x0_rosen, dx_rosen)
    print(f"Знайдений мінімум: X = [{res_rosen[0]:.4f}, {res_rosen[1]:.4f}]")
    print(f"Значення функції у мінімумі: {val_rosen:.6e}")
    print(f"Кількість кроків тесту: {len(traj_rosen)}")
    
    print("\nРозв'язання системи нелінійних рівнянь")
    x0_sys = [0.0, 0.0]  # Початкове наближення
    dx_sys = [0.2, 0.2]  # Початковий крок
    
    res_sys, val_sys, traj_sys = hooke_jeeves(objective_phi, x0_sys, dx_sys, eps1=1e-6, eps2=1e-6)
    
    file_path = "trajectory_steps.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Step\tX1\tX2\tPhi(X)\n")
        for idx, pt in enumerate(traj_sys):
            f.write(f"{idx}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{objective_phi(pt):.6e}\n")
            
    print(f"Координати траєкторії спуску успішно записані у файл: {file_path}")
    print(f"Знайдений розв'язок системи: x1 = {res_sys[0]:.6f}, x2 = {res_sys[1]:.6f}")
    print(f"Значення цільової функції Фі(X): {val_sys:.6e}")
    print(f"Загальна кількість кроків на траєкторії: {len(traj_sys)}")
    
    # Побудова графіка лінії рівня цільової функції та траєкторії спуску
    x1 = np.linspace(-1.5, 2.5, 200)
    x2 = np.linspace(-1.5, 2.5, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = objective_phi([X1, X2])
    
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X1, X2, Z, levels=np.logspace(-3, 2, 20), cmap='jet')
    plt.clabel(cp, inline=True, fontsize=8)
    
    traj_arr = np.array(traj_sys)
    plt.plot(traj_arr[:, 0], traj_arr[:, 1], 'ro-', markersize=4, linewidth=1.5, label="Траєкторія спуску")
    plt.scatter(res_sys[0], res_sys[1], color='black', marker='*', s=150, label="Мінімум (розв'язок)")
    
    plt.title("Метод Хука-Дживса для розв'язку системи нелінійних рівнянь")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.show()