import os
import numpy as np
import matplotlib.pyplot as plt

def mat_vec_mult(A, X):
    """Обчислення добутку матриці на вектор"""
    n = A.shape[0]
    result = np.zeros(n, dtype=A.dtype)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * X[j]
        result[i] = s
    return result

def vector_norm(v):
    """Обчислення норми вектора (максимум абсолютних значень)"""
    max_val = 0.0
    for i in range(len(v)):
        abs_v = abs(v[i])
        if abs_v > max_val:
            max_val = abs_v
    return float(max_val)

# Генерація матриці A (100x100) та вектора B для x_exact = 2.5
def generate_and_save_data(n=100, x_exact_val=2.5, dir_path="."):
    np.random.seed(42)  # Фіксуємо seed для відтворюваності досліджень
    # Генеруємо діагонально переважаючу матрицю для стійкості LU-розкладу
    A = np.random.uniform(-10, 10, size=(n, n))
    np.fill_diagonal(A, np.sum(np.abs(A), axis=1) + 10.0)
    
    x_exact = np.full(n, x_exact_val)
    B = mat_vec_mult(A, x_exact)
    
    file_a = os.path.join(dir_path, "matrix_A.txt")
    file_b = os.path.join(dir_path, "vector_B.txt")
    
    np.savetxt(file_a, A, fmt="%.8f")
    np.savetxt(file_b, B, fmt="%.8f")
    
    print("Сгенеровано та збережено в текстові файли:")
    print(f"Матриця A [{n}x{n}]: {file_a}")
    print(f"Вектор B [{n}]: {file_b}\n")
    return file_a, file_b, x_exact

# Допоміжні функції для зчитування, LU-розкладу та розв'язування
def load_matrix(filename):
    return np.loadtxt(filename)

def load_vector(filename):
    return np.loadtxt(filename)

def lu_decomposition(A):
    """Обчислення LU-розкладу без вибору головного елемента"""
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.eye(n)
    
    for k in range(n):
        for i in range(k, n):
            L[i, k] = A[i, k] - np.sum(L[i, :k] * U[:k, k])
        for i in range(k + 1, n):
            U[k, i] = (A[k, i] - np.sum(L[k, :k] * U[:k, i])) / L[k, k]
            
    return L, U

def save_lu(L, U, filename="lu_decomposition.txt"):
    n = L.shape[0]
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Матриця L\n")
        np.savetxt(f, L, fmt="%.6f")
        f.write("\nМатриця U\n")
        np.savetxt(f, U, fmt="%.6f")

def solve_lu(L, U, B):
    """Розв'язок системи LZ = B та UX = Z"""
    n = L.shape[0]
    
    # Прямий хід: LZ = B
    Z = np.zeros(n)
    Z[0] = B[0] / L[0, 0]
    for k in range(1, n):
        Z[k] = (B[k] - np.sum(L[k, :k] * Z[:k])) / L[k, k]
        
    # Зворотний хід: UX = Z
    X = np.zeros(n)
    X[n - 1] = Z[n - 1]
    for k in range(n - 2, -1, -1):
        X[k] = Z[k] - np.sum(U[k, k + 1:] * X[k + 1:])
        
    return X

def compute_residual_norm(A, X, B):
    """Оцінка точності за нормою максимум нев'язки"""
    A_ld = A.astype(np.longdouble)
    X_ld = X.astype(np.longdouble)
    B_ld = B.astype(np.longdouble)
    R = B_ld - mat_vec_mult(A_ld, X_ld)
    return vector_norm(R)


if __name__ == "__main__":
    n_dim = 100
    file_a, file_b, x_exact = generate_and_save_data(n=n_dim, x_exact_val=2.5)
    
    # Зчитування та виконання LU-розкладу
    A = load_matrix(file_a)
    B = load_vector(file_b)
    
    L, U = lu_decomposition(A)
    save_lu(L, U, "lu_decomposition.txt")
    
    # Розв'язання системи через LU
    X_0 = solve_lu(L, U, B)
    
    print("\nПервинний розв'язок та оцінка точності:")
    print(f"Перші 5 елементів знайденого X_0: {X_0[:5]}")
    
    # Оцінка початкової точності eps
    eps_initial = compute_residual_norm(A, X_0, B)
    err_exact_initial = vector_norm(X_0 - x_exact)
    print(f"Початкова похибка нев'язки (eps) : {eps_initial:.6e}")
    print(f"Фактична відхиленість від x=2.5   : {err_exact_initial:.6e}\n")

    
    # Ітераційний метод уточнення розв'язку
    eps_target = 1e-14
    max_iter = 50
    
    X_curr = X_0.copy()
    history_eps = [eps_initial]
    history_exact_err = [err_exact_initial]
    
    iterations_count = 0
    
    # Підготовка розширеної точності для обчислення нев'язки
    A_ld = A.astype(np.longdouble)
    B_ld = B.astype(np.longdouble)
    
    for k in range(1, max_iter + 1):
        # Обчислення вектора нев'язки R = B - A * X у підвищеній точності
        AX_ld = mat_vec_mult(A_ld, X_curr.astype(np.longdouble))
        R_k = np.array(B_ld - AX_ld, dtype=np.float64)
        
        # Перевірка умови зупинки
        current_eps = vector_norm(R_k)
        if current_eps <= eps_target:
            iterations_count = k - 1
            break
            
        # Розв'язання системи A * delta_X = R_k через той самий LU-розклад
        delta_X = solve_lu(L, U, R_k)
        
        # Уточнення розв'язку
        X_curr = X_curr + delta_X
        
        # Обчислення похибок
        new_eps = compute_residual_norm(A, X_curr, B)
        new_exact_err = vector_norm(X_curr - x_exact)
        
        history_eps.append(new_eps)
        history_exact_err.append(new_exact_err)
        iterations_count = k
        
        if new_eps <= eps_target:
            break

    print("\nІтераційне уточнення розв'язку:")
    print(f"Цільова точність eps0            : {eps_target:.1e}")
    print(f"Кількість виконаних ітерацій     : {iterations_count}")
    print(f"Фінальна похибка нев'язки        : {history_eps[-1]:.6e}")
    print(f"Фінальна відхиленість від x=2.5  : {history_exact_err[-1]:.6e}")
    print(f"Перші 5 елементів уточненого X  : {X_curr[:5]}\n")

    
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(len(history_eps)), history_eps, 'ro-', linewidth=2, label="Похибка нев'язки max|AX - B|")
    plt.axhline(eps_target, color='green', linestyle='--', label=f"Цільова точність eps0 = {eps_target:.0e}")
    plt.xlabel("Номер ітерації")
    plt.ylabel("Абсолютна похибка (логарифмічна шкала)")
    plt.title("Збіжність ітераційного методу уточнення розв'язку СЛАР")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()