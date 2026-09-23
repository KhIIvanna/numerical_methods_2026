import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# Допоміжні функції інтерполяції
def read_csv_data(file_name):
    x_val, y_val = [], []
    with open(file_name, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_val.append(float(row['n']))
            y_val.append(float(row['fps']))
    return np.array(x_val), np.array(y_val)


def divided_differences(x_arr, y_arr):
    n = len(y_arr)
    mat = np.zeros([n, n])
    mat[:, 0] = y_arr
    for j in range(1, n):
        for i in range(n - j):
            mat[i][j] = (mat[i + 1][j - 1] - mat[i][j - 1]) / (x_arr[i + j] - x_arr[i])
    return mat


def newton_poly(coef_mat, x_data, x_val):
    n = len(x_data)
    p = coef_mat[0, n - 1]
    for k in range(1, n):
        p = coef_mat[0, n - 1 - k] + (x_val - x_data[n - 1 - k]) * p
    return p


def forward_diff_table(y_arr):
    n = len(y_arr)
    table = np.zeros((n, n))
    table[:, 0] = y_arr
    for j in range(1, n):
        for i in range(n - j):
            table[i, j] = table[i + 1, j - 1] - table[i, j - 1]
    return table


def factorial_poly(table, x0, step_h, x_val):
    t = (x_val - x0) / step_h
    n = table.shape[0]
    res = table[0, 0]
    t_term = 1.0
    for k in range(1, n):
        t_term *= (t - (k - 1))
        res += (table[0, k] * t_term) / math.factorial(k)
    return res


def lagrange_poly(x_data, y_data, x_val):
    n = len(x_data)
    total = 0.0
    for i in range(n):
        term = y_data[i]
        for j in range(n):
            if i != j:
                term *= (x_val - x_data[j]) / (x_data[i] - x_data[j])
        total += term
    return total


if __name__ == "__main__":

    #1 Зчитування даних з CSV-файлу
    x_nodes, y_nodes = read_csv_data("data.csv")
    print("Вхідні дані (n, FPS):")
    for x_i, y_i in zip(x_nodes, y_nodes):
        print(f"n = {x_i:.0f}, FPS = {y_i:.0f}")

    #2 Побудова таблиці розділених різниць
    coef_matrix = divided_differences(x_nodes, y_nodes)
    print("\nТаблиця розділених різниць (перший рядок):")
    print(coef_matrix[0, :])

    #3 Обчислення прогнозу (методами Ньютона і факторіальними многочленами)
    target_n = 1000.0
    pred_newton = newton_poly(coef_matrix, x_nodes, target_n)

    diff_table = forward_diff_table(y_nodes)
    step = x_nodes[1] - x_nodes[0]
    pred_fact = factorial_poly(diff_table, x_nodes[0], step, target_n)

    print(f"\nПрогноз для n = {target_n:.0f}:")
    print(f"Метод Ньютона: FPS = {pred_newton:.2f}")
    print(f"Факторіальний многочлен: FPS = {pred_fact:.2f}")

    #4 Побудова графіку FPS(n)
    x_dense = np.linspace(100, 1600, 500)
    y_dense = np.array([newton_poly(coef_matrix, x_nodes, x) for x in x_dense])

    plt.figure(figsize=(8, 5))
    plt.plot(x_nodes, y_nodes, 'ro', label='Експериментальні точки')
    plt.plot(x_dense, y_dense, 'b-', label='Інтерполяція Ньютона')
    plt.axhline(60, color='g', linestyle='--', label='Поріг FPS = 60')
    plt.xlabel('n (об`єкти)')
    plt.ylabel('FPS')
    plt.title('Графік FPS(n)')
    plt.grid(True)
    plt.legend()
    plt.show()


    #5 Дослідження впливу кількості вузлів (n=5,10,20) та побудова графіку похибок для зазначеної кількості вузлів
    def test_f(x):
        return 120.0 / (1.0 + (x / 500.0) ** 2)


    x_test = np.linspace(100, 1600, 300)
    y_true = test_f(x_test)

    plt.figure(figsize=(8, 5))
    for n_nodes in [5, 10, 20]:
        nodes_x = np.linspace(100, 1600, n_nodes)
        nodes_y = test_f(nodes_x)
        c_mat = divided_differences(nodes_x, nodes_y)
        y_pred = np.array([newton_poly(c_mat, nodes_x, x) for x in x_test])
        err = np.abs(y_true - y_pred)
        plt.plot(x_test, err, label=f'Вузлів n = {n_nodes}')

    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Абсолютна похибка')
    plt.title('Похибка для n = 5, 10, 20 вузлів')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Дослідницька_частина_1 Дослідження впливу кроку: фіксований інтервал, різна кількість вузлів
    plt.figure(figsize=(8, 5))
    for n_nodes in [4, 8, 12, 16]:
        nodes_x = np.linspace(100, 1600, n_nodes)
        nodes_y = test_f(nodes_x)
        c_mat = divided_differences(nodes_x, nodes_y)
        y_pred = np.array([newton_poly(c_mat, nodes_x, x) for x in x_test])
        err = np.abs(y_true - y_pred)
        plt.plot(x_test, err, label=f'n = {n_nodes} вузлів (h = {(1500 / (n_nodes - 1)):.1f})')

    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Похибка')
    plt.title('Фіксований інтервал [100, 1600], різна кількість вузлів')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Дослідницька_частина_2 Дослідження впливу кількості вузлів, похибок. Фіксований крок, змінний інтервал.
    h_fixed = 100.0
    a_start = 100.0
    plt.figure(figsize=(8, 5))
    for n_nodes in [5, 10, 15]:
        b_end = a_start + h_fixed * (n_nodes - 1)
        nodes_x = np.linspace(a_start, b_end, n_nodes)
        nodes_y = test_f(nodes_x)
        c_mat = divided_differences(nodes_x, nodes_y)

        x_eval = np.linspace(a_start, b_end, 200)
        y_eval_true = test_f(x_eval)
        y_eval_pred = np.array([newton_poly(c_mat, nodes_x, x) for x in x_eval])

        err = np.abs(y_eval_true - y_eval_pred)
        plt.plot(x_eval, err, label=f'n = {n_nodes}, інтервал [{a_start:.0f}, {b_end:.0f}]')

    plt.yscale('log')
    plt.xlabel('n')
    plt.ylabel('Похибка')
    plt.title('Фіксований крок h = 100, змінний інтервал')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Дослідницька_частина_3 Аналіз ефекту Рунге
    def runge_f(x):
        return 1.0 / (1.0 + x ** 2)


    x_runge = np.linspace(-5, 5, 500)
    y_runge_true = runge_f(x_runge)

    plt.figure(figsize=(8, 5))
    plt.plot(x_runge, y_runge_true, 'k--', label='Еталон 1/(1+x^2)', linewidth=2)
    for n_nodes in [5, 10, 15]:
        nodes_x = np.linspace(-5, 5, n_nodes)
        nodes_y = runge_f(nodes_x)
        c_mat = divided_differences(nodes_x, nodes_y)
        y_pred = np.array([newton_poly(c_mat, nodes_x, x) for x in x_runge])
        plt.plot(x_runge, y_pred, label=f'Ньютон n = {n_nodes}')

    plt.ylim(-1, 2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аналіз ефекту Рунге')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Дослідницька_частина_4 Порівняння з методом Лагранжа
    pred_lagrange = lagrange_poly(x_nodes, y_nodes, target_n)
    diff_between_methods = abs(pred_newton - pred_lagrange)

    print("\nПорівняння методів для n = 1000:")
    print(f"Ньютон:   {pred_newton:.6f}")
    print(f"Лагранж:  {pred_lagrange:.6f}")
    print(f"Різниця між методами: {diff_between_methods:.6e}")