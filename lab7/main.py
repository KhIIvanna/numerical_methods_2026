import os
import numpy as np
import matplotlib.pyplot as plt

def mat_vec_mult(A, X):
    """Обчислення добутку матриці на вектор вручну"""
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

def matrix_norm_c(C):
    """Обчислення норми матриці C"""
    max_row_sum = 0.0
    for i in range(C.shape[0]):
        row_sum = 0.0
        for j in range(C.shape[1]):
            row_sum += abs(C[i, j])
        if row_sum > max_row_sum:
            max_row_sum = row_sum
    return float(max_row_sum)

def load_matrix(filename):
    """Зчитування матриці A з текстового файлу"""
    return np.loadtxt(filename)

def load_vector(filename):
    """Зчитування вектора B з текстового файлу"""
    return np.loadtxt(filename)


def generate_and_save_data(n=100, x_exact_val=2.5, dir_path="."):
    np.random.seed(42)
    # Генерація матриці з сильним діагональним переважанням для збіжності ітерацій
    A = np.random.uniform(-1, 1, size=(n, n))
    diag_dominance = np.sum(np.abs(A), axis=1) + 20.0
    np.fill_diagonal(A, diag_dominance)
    
    x_exact = np.full(n, x_exact_val)
    B = mat_vec_mult(A, x_exact)
    
    file_a = os.path.join(dir_path, "matrix_A.txt")
    file_b = os.path.join(dir_path, "vector_B.txt")
    
    np.savetxt(file_a, A, fmt="%.8f")
    np.savetxt(file_b, B, fmt="%.8f")
    
    print("Сгенеровано та збережено в текстові файли:")
    print(f"Матриця A [{n}x{n}]: {file_a}")
    print(f"Вектор B [{n}]: {file_b}\n")
    return file_a, file_b


def simple_iteration_method(A, B, x0, eps=1e-14, max_iter=5000):
    """Метод простої ітерації: X^(k+1) = (E - tau*A)X^(k) + tau*B"""
    n = A.shape[0]
    # Вибір оптимального параметру tau
    norm_A = matrix_norm_c(A)
    tau = 1.0 / norm_A  # безпечний вибір параметра релаксації
    
    X_curr = x0.copy()
    history = [vector_norm(mat_vec_mult(A, X_curr) - B)]
    
    for k in range(1, max_iter + 1):
        X_next = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * X_curr[j]
            X_next[i] = X_curr[i] - tau * (s - B[i])
            
        diff = vector_norm(X_next - X_curr)
        X_curr = X_next
        
        residual = vector_norm(mat_vec_mult(A, X_curr) - B)
        history.append(residual)
        
        if diff <= eps or residual <= eps:
            return X_curr, k, history
            
    return X_curr, max_iter, history


def jacobi_method(A, B, x0, eps=1e-14, max_iter=5000):
    """Метод Якобі"""
    n = A.shape[0]
    X_curr = x0.copy()
    history = [vector_norm(mat_vec_mult(A, X_curr) - B)]
    
    for k in range(1, max_iter + 1):
        X_next = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * X_curr[j]
            X_next[i] = (B[i] - s) / A[i, i]
            
        diff = vector_norm(X_next - X_curr)
        X_curr = X_next
        
        residual = vector_norm(mat_vec_mult(A, X_curr) - B)
        history.append(residual)
        
        if diff <= eps or residual <= eps:
            return X_curr, k, history
            
    return X_curr, max_iter, history


def seidel_method(A, B, x0, eps=1e-14, max_iter=5000):
    """Метод Зейделя (Гауса-Зейделя)"""
    n = A.shape[0]
    X_curr = x0.copy()
    history = [vector_norm(mat_vec_mult(A, X_curr) - B)]
    
    for k in range(1, max_iter + 1):
        X_next = X_curr.copy()
        for i in range(n):
            s1 = 0.0
            for j in range(i):
                s1 += A[i, j] * X_next[j]  # вже уточнені значення
            s2 = 0.0
            for j in range(i + 1, n):
                s2 += A[i, j] * X_curr[j]  # попередні значення
                
            X_next[i] = (B[i] - s1 - s2) / A[i, i]
            
        diff = vector_norm(X_next - X_curr)
        X_curr = X_next
        
        residual = vector_norm(mat_vec_mult(A, X_curr) - B)
        history.append(residual)
        
        if diff <= eps or residual <= eps:
            return X_curr, k, history
            
    return X_curr, max_iter, history


if __name__ == "__main__":
    n_dim = 100
    file_a, file_b = generate_and_save_data(n=n_dim, x_exact_val=2.5)
    
    # Зчитування матриці та вектора з файлів
    A = load_matrix(file_a)
    B = load_vector(file_b)
    
    # Початкове наближення
    x0 = np.full(n_dim, 1.0)
    eps_target = 1e-14
    
    # Проста ітерація
    sol_si, iter_si, hist_si = simple_iteration_method(A, B, x0, eps=eps_target)
    print(f"Метод простої ітерації: ітерацій = {iter_si}, перші 3 ел. = {sol_si[:3]}")
    
    # Якобі
    sol_jac, iter_jac, hist_jac = jacobi_method(A, B, x0, eps=eps_target)
    print(f"Метод Якобі          : ітерацій = {iter_jac}, перші 3 ел. = {sol_jac[:3]}")
    
    # Зейдель
    sol_seid, iter_seid, hist_seid = seidel_method(A, B, x0, eps=eps_target)
    print(f"Метод Зейделя        : ітерацій = {iter_seid}, перші 3 ел. = {sol_seid[:3]}\n")
    
    # Побудова графіка збіжності
    plt.figure(figsize=(9, 5))
    plt.semilogy(hist_si, label=f"Проста ітерація ({iter_si} ітер.)", color='red')
    plt.semilogy(hist_jac, label=f"Якобі ({iter_jac} ітер.)", color='blue')
    plt.semilogy(hist_seid, label=f"Зейдель ({iter_seid} ітер.)", color='green')
    plt.axhline(eps_target, color='black', linestyle='--', label=f"Цільова точність eps = {eps_target:.0e}")
    
    plt.xlabel("Номер ітерації")
    plt.ylabel("Норма нев'язки")
    plt.title("Порівняння збіжності ітераційних методів розв'язку СЛАР")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()