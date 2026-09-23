import requests
import numpy as np
import urllib3
import matplotlib.pyplot as plt

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#1 Запит до Open-Elevation API
url = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|"
    "48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|"
    "48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|"
    "48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|"
    "48.160250,24.500106"
)

response = requests.get(url, verify=False)
data = response.json()
results = data["results"]
n = len(results)


# Формула Гаверсину для обчислення відстані між координатами
def haversine(lat1, lon1, lat2, lon2):
    r_earth = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r_earth * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

# Обчислення кумулятивної відстані
distances = [0.0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

# Табуляція у файл tabulation.txt
with open("tabulation.txt", "w", encoding="utf-8") as f:
    f.write(f"Кількість вузлів: {n}\n\n")
    f.write("Табуляція вузлів (GPS):\n")
    f.write(f"{'№':>2} | {'Latitude':>10} | {'Longitude':>10} | {'Elevation (m)':>13}\n")
    f.write("-" * 46 + "\n")
    for i, point in enumerate(results):
        f.write(f"{i:2d} | {point['latitude']:10.6f} | {point['longitude']:10.6f} | {point['elevation']:13.2f}\n")

    f.write("\n\nТабуляція (кумулятивна відстань, висота):\n")
    f.write(f"{'№':>2} | {'Distance (m)':>12} | {'Elevation (m)':>13}\n")
    f.write("-" * 35 + "\n")
    for i in range(n):
        f.write(f"{i:2d} | {distances[i]:12.2f} | {elevations[i]:13.2f}\n")

#2 Запис даних у файл input_data.txt
with open("input_data.txt", "w", encoding="utf-8") as f:
    f.write("distance_m,elevation_m\n")
    for i in range(n):
        f.write(f"{distances[i]:.2f},{elevations[i]:.2f}\n")

print("Файли збережені")

#3 Побудова графіка вхідних точок
plt.figure(figsize=(10, 5))
plt.plot(distances, elevations, 'ro--', label='Дискретні точки GPS (Заросляк — Говерла)')
plt.title('Профіль висоти маршруту (Вхідні дані)')
plt.xlabel('Кумулятивна відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#4 Формування системи лінійних алгебраїчних рівнянь трьохдіагональною матрицею
h = [distances[i] - distances[i - 1] for i in range(1, n)]

alpha = [0.0] * n
beta = [0.0] * n
gamma = [0.0] * n
delta = [0.0] * n

beta[0] = 1.0

for i in range(1, n - 1):
    alpha[i] = h[i - 1]
    beta[i] = 2.0 * (h[i - 1] + h[i])
    gamma[i] = h[i]
    delta[i] = 3.0 * ((elevations[i + 1] - elevations[i]) / h[i] - (elevations[i] - elevations[i - 1]) / h[i - 1])

beta[n - 1] = 1.0

# Друк коефіцієнтів матриці СЛАР
print("\nКоефіцієнти тридіагональної матриці:")
print("-" * 55)
print(f"{'i':>2} | {'alpha_i':>10} | {'beta_i':>10} | {'gamma_i':>10} | {'delta_i':>10}")
print("-" * 55)
for i in range(n):
    print(f"{i:2d} | {alpha[i]:10.2f} | {beta[i]:10.2f} | {gamma[i]:10.2f} | {delta[i]:10.4f}")

# Функція розв'язку СЛАР методом прогонки
def solve_tridiagonal(a, b, c, d, n_size):
    A_coeff = [0.0] * n_size
    B_coeff = [0.0] * n_size

    # Прямий хід прогонки
    A_coeff[0] = -c[0] / b[0]
    B_coeff[0] = d[0] / b[0]

    for i in range(1, n_size - 1):
        denom = a[i] * A_coeff[i - 1] + b[i]
        A_coeff[i] = -c[i] / denom
        B_coeff[i] = (d[i] - a[i] * B_coeff[i - 1]) / denom

    # Зворотний хід прогонки
    x = [0.0] * n_size
    denom_last = a[n_size - 1] * A_coeff[n_size - 2] + b[n_size - 1]
    x[n_size - 1] = (d[n_size - 1] - a[n_size - 1] * B_coeff[n_size - 2]) / denom_last

    for i in range(n_size - 2, -1, -1):
        x[i] = A_coeff[i] * x[i + 1] + B_coeff[i]

    return x, A_coeff, B_coeff

# Обчислення коефіцієнтів с_і за допомогою метеду прогонки
c_coeff, A_prog, B_prog = solve_tridiagonal(alpha, beta, gamma, delta, n)

print("\nПрогоничні коефіцієнти (A_i, B_i) та коефіцієнт c_i:")
print("-" * 50)
print(f"{'i':>2} | {'A_i':>12} | {'B_i':>12} | {'c_i':>12}")
print("-" * 50)
for i in range(n):
    print(f"{i:2d} | {A_prog[i]:12.6f} | {B_prog[i]:12.6f} | {c_coeff[i]:12.6f}")

# Обчисленння інших коефіцієнтів
a_coeff = [0.0] * n
b_coeff = [0.0] * n
d_coeff = [0.0] * n

for i in range(1, n):
    a_coeff[i] = elevations[i - 1]

    if i < n - 1:
        d_coeff[i] = (c_coeff[i + 1] - c_coeff[i]) / (3.0 * h[i - 1])
        b_coeff[i] = (elevations[i] - elevations[i - 1]) / h[i - 1] - (h[i - 1] / 3.0) * (
                    c_coeff[i + 1] + 2.0 * c_coeff[i])
    else:
        d_coeff[n - 1] = -c_coeff[n - 1] / (3.0 * h[n - 2])
        b_coeff[n - 1] = (elevations[n - 1] - elevations[n - 2]) / h[n - 2] - (2.0 / 3.0) * h[n - 2] * c_coeff[n - 1]

print("\nКоефіцієнти кубічних сплайнів:")
print("-" * 52)
print(f"{'i':>2} | {'a_i':>10} | {'b_i':>10} | {'c_i':>10} | {'d_i':>12}")
print("-" * 52)
for i in range(1, n):
    print(f"{i:2d} | {a_coeff[i]:10.2f} | {b_coeff[i]:10.4f} | {c_coeff[i]:10.6f} | {d_coeff[i]:12.8f}")

#5 Порівняння сплайнів для 10, 15, 21 вузлів та похибка
def build_spline_for_nodes(x_nodes, y_nodes):
    n_nodes = len(x_nodes)
    h_local = [x_nodes[k] - x_nodes[k - 1] for k in range(1, n_nodes)]

    a_l, b_l, g_l, d_l = [0.0] * n_nodes, [0.0] * n_nodes, [0.0] * n_nodes, [0.0] * n_nodes
    b_l[0] = 1.0
    for k in range(1, n_nodes - 1):
        a_l[k] = h_local[k - 1]
        b_l[k] = 2.0 * (h_local[k - 1] + h_local[k])
        g_l[k] = h_local[k]
        d_l[k] = 3.0 * ((y_nodes[k + 1] - y_nodes[k]) / h_local[k] - (y_nodes[k] - y_nodes[k - 1]) / h_local[k - 1])
    b_l[n_nodes - 1] = 1.0

    c_coeffs, _, _ = solve_tridiagonal(a_l, b_l, g_l, d_l, n_nodes)
    a_coeffs, b_coeffs, d_coeffs = [0.0] * n_nodes, [0.0] * n_nodes, [0.0] * n_nodes

    for k in range(1, n_nodes):
        a_coeffs[k] = y_nodes[k - 1]
        if k < n_nodes - 1:
            d_coeffs[k] = (c_coeffs[k + 1] - c_coeffs[k]) / (3.0 * h_local[k - 1])
            b_coeffs[k] = (y_nodes[k] - y_nodes[k - 1]) / h_local[k - 1] - (h_local[k - 1] / 3.0) * (
                        c_coeffs[k + 1] + 2.0 * c_coeffs[k])
        else:
            d_coeffs[n_nodes - 1] = -c_coeffs[n_nodes - 1] / (3.0 * h_local[n_nodes - 2])
            b_coeffs[n_nodes - 1] = (y_nodes[n_nodes - 1] - y_nodes[n_nodes - 2]) / h_local[n_nodes - 2] - (2.0 / 3.0) * \
                                    h_local[n_nodes - 2] * c_coeffs[n_nodes - 1]

    return a_coeffs, b_coeffs, c_coeffs, d_coeffs


def evaluate_spline(x_val, x_nodes, a_c, b_c, c_c, d_c):
    n_nodes = len(x_nodes)
    if x_val <= x_nodes[0]:
        return a_c[1]
    if x_val >= x_nodes[-1]:
        idx = n_nodes - 1
        dx = x_val - x_nodes[idx - 1]
        return a_c[idx] + b_c[idx] * dx + c_c[idx] * (dx ** 2) + d_c[idx] * (dx ** 3)

    for k in range(1, n_nodes):
        if x_nodes[k - 1] <= x_val <= x_nodes[k]:
            dx = x_val - x_nodes[k - 1]
            return a_c[k] + b_c[k] * dx + c_c[k] * (dx ** 2) + d_c[k] * (dx ** 3)
    return 0.0


xx = np.linspace(distances[0], distances[-1], 500)

indices_10 = np.linspace(0, n - 1, 10, dtype=int)
indices_15 = np.linspace(0, n - 1, 15, dtype=int)

x_10, y_10 = [distances[i] for i in indices_10], [elevations[i] for i in indices_10]
x_15, y_15 = [distances[i] for i in indices_15], [elevations[i] for i in indices_15]

a10, b10, c10, d10 = build_spline_for_nodes(x_10, y_10)
a15, b15, c15, d15 = build_spline_for_nodes(x_15, y_15)

yy_20 = [evaluate_spline(x, distances, a_coeff, b_coeff, c_coeff, d_coeff) for x in xx]
yy_15 = [evaluate_spline(x, x_15, a15, b15, c15, d15) for x in xx]
yy_10 = [evaluate_spline(x, x_10, a10, b10, c10, d10) for x in xx]

#5 Графік порівняння сплайнів
plt.figure(figsize=(12, 6))
plt.plot(xx, yy_10, label='Сплайн (10 вузлів)', linestyle='--')
plt.plot(xx, yy_15, label='Сплайн (15 вузлів)', linestyle='-.')
plt.plot(xx, yy_20, label='Сплайн (21 вузол - повний)', color='green')
plt.scatter(distances, elevations, color='red', zorder=5, label='Вхідні GPS-точки')
plt.title('Порівняння інтерполяції кубічними сплайнами при різній кількості вузлів')
plt.xlabel('Відстань (м)')
plt.ylabel('Висота (м)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#6 Графік похибки
error_10 = np.abs(np.array(yy_20) - np.array(yy_10))
plt.figure(figsize=(12, 4))
plt.plot(xx, error_10, color='purple', label='Похибка ε(x) = |S_21(x) - S_10(x)|')
plt.title('Графік абсолютної похибки між точним сплайном і спрощеним (10 вузлів)')
plt.xlabel('Відстань (м)')
plt.ylabel('Похибка (м)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#Додатково
print("\nХарактеристики маршруту:")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")

total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))

print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
print(f"Сумарний спуск (м): {total_descent:.2f}")

grad_full = np.gradient(yy_20, xx) * 100
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

mass = 80
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота підйому (кДж): {energy / 1000:.2f}")
print(f"Витрачена енергія (ккал): {energy / 4184:.2f}")