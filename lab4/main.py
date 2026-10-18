import numpy as np
import matplotlib.pyplot as plt

# 1. Аналітичне розв'язання
def M(t):
    """Функція вологості ґрунту M(t)"""
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
    """Точна аналітична похідна dM/dt"""
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

x0 = 1.0
exact_val = dM_exact(x0)

print("Аналітична похідна:")
print(f"Точна похідна у точці x0 = {x0}: y'(x0) = {exact_val:.12f}\n")


# 2. Дослідження залежності похибки R від кроку h 
def num_derivative_central(f, x, h):
    """Центральна різницева формула першої похідної"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Генеруємо логарифмічну сітку кроків від 10^-20 до 10^3
h_powers = np.linspace(-20, 3, 2000)
h_array = 10.0 ** h_powers

errors_array = []
num_derivs = []

for h in h_array:
    calc_d = num_derivative_central(M, x0, h)
    num_derivs.append(calc_d)
    errors_array.append(abs(calc_d - exact_val))

errors_array = np.array(errors_array)

# Пошук оптимального кроку h0
min_err_idx = np.argmin(errors_array)
h0 = h_array[min_err_idx]
R0 = errors_array[min_err_idx]

print("\nПошук оптимального кроку h0 (Регуляризація):")
print(f"Оптимальний крок h0: {h0:.6e}")
print(f"Досягнута точність R0: {R0:.6e}\n")


# 3. Позначення кроку
h_base = 1e-3

# 4. Обчислення для h та 2h
y_prime_h = num_derivative_central(M, x0, h_base)
y_prime_2h = num_derivative_central(M, x0, 2 * h_base)

# 5. Похибка R1 при кроці h
R1 = abs(y_prime_h - exact_val)

# 6. Метод Рунге-Ромберга
y_prime_R = y_prime_h + (y_prime_h - y_prime_2h) / 3.0
R2 = abs(y_prime_R - exact_val)

print("\nОбчислення для h = 10^-3 та Метод Рунге-Ромберга:")
print(f"y0'(h)   [h  = 0.001] = {y_prime_h:.12f} | Похибка R1 = {R1:.6e}")
print(f"y0'(2h)  [2h = 0.002] = {y_prime_2h:.12f}")
print(f"y_R' (Рунге-Ромберг) = {y_prime_R:.12f} | Похибка R2 = {R2:.6e}")
print(f"Покращення точності (R1 / R2): {R1 / R2:.2f} разів\n")


# 7. Метод Ейткена та оцінка порядку точності p (для кроків h, 2h, 4h)
y_prime_4h = num_derivative_central(M, x0, 4 * h_base)

# Формула Ейткена
denom_aitken = 2 * y_prime_2h - (y_prime_4h + y_prime_h)
y_prime_E = ((y_prime_2h) ** 2 - y_prime_4h * y_prime_h) / denom_aitken
R3 = abs(y_prime_E - exact_val)

# Оцінка порядку точності p
p_est = (1.0 / np.log(2)) * np.log(abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h)))

print("\nМетод Ейткена та оцінка порядку точності:")
print(f"y0'(4h)  [4h = 0.004] = {y_prime_4h:.12f}")
print(f"y_E' (Ейткен)        = {y_prime_E:.12f} | Похибка R3 = {R3:.6e}")
print(f"Оцінений порядок точності p: {p_est:.4f} (Теоретичний p = 2)")
print(f"Покращення точності відносно R1 (R1 / R3): {R1 / R3:.2f} разів\n")

# Графік залежності похибки R від кроку h (визначення h0)
plt.figure(figsize=(9, 5))
plt.loglog(h_array, errors_array, label='Повна похибка R(h)', color='blue', linewidth=1.5)
plt.axvline(h0, color='red', linestyle='--', label=f'Оптимальний крок h0 = {h0:.2e}')
plt.scatter([h0], [R0], color='red', zorder=5, label=f'R0 min = {R0:.2e}')
plt.xlabel('Крок h (логарифмічна шкала)')
plt.ylabel('Похибка R (логарифмічна шкала)')
plt.title('Залежність похибки чисельного диференціювання від кроку h')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

# Графік моделі M(t) та точка t0 = 1
t_vals = np.linspace(0, 20, 500)
plt.figure(figsize=(9, 5))
plt.plot(t_vals, M(t_vals), label='M(t) = 50e^{-0.1t} + 5sin(t)', color='green', linewidth=2)
plt.scatter([x0], [M(x0)], color='red', zorder=5, label=f'Точка t0 = {x0}')
plt.xlabel('Час t')
plt.ylabel('Вологість ґрунту')
plt.title('Графік моделі вологості ґрунту')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Графік порівняння похибок методів при h = 10^-3
methods = ['Центральна різниця (h)', 'Рунге-Ромберг', 'Ейткен']
errors = [R1, R2, R3]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(methods, errors, color=['skyblue', 'orange', 'green'])
plt.yscale('log')
plt.ylabel('Абсолютна похибка (логарифмічна шкала)')
plt.title('Порівняння точності методів чисельного диференціювання')
plt.grid(True, which="both", axis='y', ls="--")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f'{yval:.2e}', ha='center', va='bottom')

plt.tight_layout()
plt.show()