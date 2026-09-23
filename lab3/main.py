import csv
import os
import numpy as np
import matplotlib.pyplot as plt

#1 Програма для табуляції заданих даних про температуру
def generate_and_save_data(file_name="data.csv"):
    x0, xn = 1.0, 24.0
    n = 24
    h = (xn - x0) / (n - 1)

    x_nodes = np.array([x0 + i * h for i in range(n)])

    # Табуляція математичної моделі середньомісячної температури f(x)
    y_nodes = 10 + 12 * np.sin((x_nodes - 4) * np.pi / 6)
    y_nodes = np.round(y_nodes, 2)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_name)

    # Збереження вузлів у CSV
    with open(full_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Month', 'Temp'])
        for x_val, y_val in zip(x_nodes, y_nodes):
            writer.writerow([x_val, y_val])

    print(f"Протабульовано {n} вузлів та збережено у файл")
    return full_path

#2 Програма, яка для заданої табличної функції {xi, fi}
# Зчитування даних
def read_temperature_data(file_name="data.csv"):
    # Зчитування вхідних даних з текстового файлу
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_name)
    months, temps = [], []

    with open(full_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            months.append(float(row['Month']))
            temps.append(float(row['Temp']))

    return np.array(months), np.array(temps)

# Формування масивів В і С
def form_mnk_system(x_arr, y_arr, m_deg):
    size = m_deg + 1
    mat_c = np.zeros((size, size))
    vec_b = np.zeros(size)

    for i in range(size):
        for j in range(size):
            mat_c[i, j] = np.sum(x_arr ** (i + j))
        vec_b[i] = np.sum(y_arr * (x_arr ** i))

    return mat_c, vec_b

# Розв'язування СЛАР методом Гауса з вибором головного елемента
def gauss_solve_pivot(a_mat, b_vector):
    # Розв'язування СЛАР методом Гауса з вибором головного елемента по стовпцях
    a = a_mat.copy().astype(float)
    b = b_vector.copy().astype(float)
    n_size = len(b)

    # Прямий хід перестановки рядків та виключення змінних
    for k in range(n_size - 1):
        max_row = k + np.argmax(np.abs(a[k:, k]))
        if max_row != k:
            a[[k, max_row]] = a[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n_size):
            factor = a[i, k] / a[k, k]
            a[i, k:] -= factor * a[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід обчислення масиву коефіцієнтів A
    x_sol = np.zeros(n_size)
    for i in range(n_size - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(a[i, i + 1:] * x_sol[i + 1:])) / a[i, i]

    return x_sol

# Обчислення алгебраїчного многочлена по коефіцієнтах елемента
def eval_polynomial(x_val, coef_arr):
    y_val = np.zeros_like(x_val, dtype=float)
    for idx, c in enumerate(coef_arr):
        y_val += c * (x_val ** idx)
    return y_val


def calc_variance(y_true, y_approx):
    # Обчислення дисперсії
    return np.mean((y_true - y_approx) ** 2)


def calc_error_abs(y_true, y_approx):
    # Обчислення функції похибки
    return np.abs(y_true - y_approx)


if __name__ == "__main__":

    #1 Створення та табуляція даних у CSV
    generate_and_save_data("data.csv")

    #2 Зчитування вхідних даних
    x_months, y_temps = read_temperature_data("data.csv")
    print("\nЗчитано вхідні дані з CSV:")
    for m_i, t_i in zip(x_months, y_temps):
        print(f"Вузол x={m_i:2.0f}: Temp = {t_i:5.2f}")

    m_demo = 2
    # Формування C і B
    c_mat_demo, b_vec_demo = form_mnk_system(x_months, y_temps, m_demo)
    print("\nСформована матриця C:")
    print(c_mat_demo)

    print("\nСформований вектор B:")
    print(b_vec_demo)

    # Розв'язання методом Гауса
    coefs_demo = gauss_solve_pivot(c_mat_demo, b_vec_demo)
    print("\nЗнайдені коефіцієнти A (методом Гауса):")
    for idx, a_val in enumerate(coefs_demo):
        print(f"  a_{idx} = {a_val:.6f}")

    # Обчислення алгебраїчного многочлена та похибки
    y_approx_demo = eval_polynomial(x_months, coefs_demo)
    errors_demo = calc_error_abs(y_temps, y_approx_demo)

    print("\nТаблиця обчислених значень многочлена та похибок :")
    print("-" * 65)
    print(f"{'Вузол (x)':^10} | {'f(x) реальне':^14} | {'алгебраїчний многочлен':^14} | {'похибка':^14}")
    print("-" * 65)
    for x, y_t, y_p, err in zip(x_months, y_temps, y_approx_demo, errors_demo):
        print(f"{x:^10.0f} | {y_t:^14.2f} | {y_p:^14.4f} | {err:^14.4f}")
    print("-" * 65)

    #3 Знаходження дисперсії
    m = 1, ..., 10
    degrees = list(range(1, 11))
    variances = []
    coef_dict = {}

    print("\nРозрахунок дисперсії для m = 1,...,10")
    for deg in degrees:
        c_mat, b_vec = form_mnk_system(x_months, y_temps, deg)
        coefficients = gauss_solve_pivot(c_mat, b_vec)
        coef_dict[deg] = coefficients
        y_pred = eval_polynomial(x_months, coefficients)
        var_val = calc_variance(y_temps, y_pred)
        variances.append(var_val)
        print(f"Степінь m = {deg:2d} | Дисперсія = {var_val}")

    # Вибір оптимального степеня m
    opt_m = degrees[np.argmin(variances)]
    print(f"\nОптимальне значення степеня m = {opt_m}")

    # Друк коефіцієнтів та вигляду многочлена
    opt_coefs = coef_dict[opt_m]
    print(f"Коефіцієнти многочлена для m={opt_m}:")
    for i, c in enumerate(opt_coefs):
        print(f"  a_{i} = {c:.6f}")

    # Формування та вивід формули
    terms = [f"{c:+.4f}*x^{i}" if i > 0 else f"{c:.4f}" for i, c in enumerate(opt_coefs)]
    poly_str = " ".join(terms)
    print(f"\nВигляд апроксимуючого многочлена:\nP(x) = {poly_str}")

    # Побудова графіка залежності дисперсії від m
    plt.figure(figsize=(8, 4))
    plt.plot(degrees, variances, 'ro-', linewidth=2, markersize=6)
    plt.axvline(opt_m, color='g', linestyle='--', label=f'Оптимальний m = {opt_m}')
    plt.xlabel('Степінь многочлена (m)')
    plt.ylabel('Дисперсія')
    plt.title('Залежність дисперсії від степені апроксимуючого многочлена')
    plt.grid(True)
    plt.legend()
    plt.show()

    #4 Табуляція
    n_pts = len(x_months)
    # Сітка для табуляції похибки з кроком h1 = (xn-x0)/(20n)
    x_dense = np.linspace(x_months[0], x_months[-1], 20 * n_pts)
    y_dense_true = 10 + 12 * np.sin((x_dense - 4) * np.pi / 6)

    # Побудова графіків похибки
    plt.figure(figsize=(9, 5))
    for deg in [1, 2, 4, 6, 8, 10]:
        coefficients = coef_dict[deg]
        y_dense_pred = eval_polynomial(x_dense, coefficients)
        err_dense = calc_error_abs(y_dense_true, y_dense_pred)
        plt.plot(x_dense, err_dense, label=f'm = {deg}')

    plt.xlabel('Місяць (x)')
    plt.ylabel('Похибка e')
    plt.title('Графіки похибки апроксимації для різних степенів m')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Побудова графіка апроксимації та фактичних даних
    opt_coefs = coef_dict[opt_m]
    y_opt_dense = eval_polynomial(x_dense, opt_coefs)

    plt.figure(figsize=(9, 5))
    plt.plot(x_months, y_temps, 'ro', label='Фактичні дані')
    plt.plot(x_dense, y_opt_dense, 'b-', label=f'Апроксимація (m = {opt_m})', linewidth=2)
    plt.xlabel('Місяць (x)')
    plt.ylabel('Температура')
    plt.title(f'Графік апроксимації та фактичних даних (m = {opt_m})')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Екстраполяція прогнозу температури на наступні 3 місяці
    x_future = np.array([25, 26, 27])
    y_future = eval_polynomial(x_future, opt_coefs)

    print("\nЕкстраполяція прогнозу температури на 3 місяці")
    for m_fut, t_fut in zip(x_future, y_future):
        print(f"Місяць {m_fut}: {t_fut:.2f}")