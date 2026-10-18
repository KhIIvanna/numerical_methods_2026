import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Математична модель та точне значення інтеграла I0
def f(x):
    """Підінтегральна функція навантаження на сервер f(x)"""
    return 50.0 + 20.0 * np.sin(np.pi * x / 12.0) + 5.0 * np.exp(-0.2 * (x - 12.0)**2)

a, b = 0.0, 24.0

def exact_integral(a, b):
    """
    Аналітичне обчислення інтеграла від f(x) на [a, b]:
    1) int(50) dx = 50*x
    2) int(20*sin(pi*x/12)) dx = -20*(12/pi)*cos(pi*x/12)
    3) int(5*exp(-0.2*(x-12)^2)) dx = 5 * sqrt(pi/0.2)/2 * erf(sqrt(0.2)*(x-12))
    """
    def F(x):
        term1 = 50.0 * x
        term2 = -20.0 * (12.0 / np.pi) * np.cos(np.pi * x / 12.0)
        term3 = 5.0 * 0.5 * np.sqrt(np.pi / 0.2) * erf(np.sqrt(0.2) * (x - 12.0))
        return term1 + term2 + term3

    return F(b) - F(a)

I0 = exact_integral(a, b)

print("Точне значення інтеграла:")
print(f"Точне значення інтеграла I0 = {I0:.15f}\n")


# Складова квадратурна формула Сімпсона
def simpson_composite(f, a, b, N):
    """
    Складова формула Сімпсона для парного N розбиттів
    """
    if N % 2 != 0:
        N += 1  # Кількість розбиттів N має бути парною
    
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    
    I_simps = (h / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1])
    return I_simps


# Дослідження залежності точності від N та пошук N_opt
N_vals = np.arange(10, 1002, 2)
eps_N = []

for N_i in N_vals:
    I_calc = simpson_composite(f, a, b, N_i)
    eps_N.append(abs(I_calc - I0))

eps_N = np.array(eps_N)

target_eps = 1e-12
valid_indices = np.where(eps_N <= target_eps)[0]

if len(valid_indices) > 0:
    N_opt = N_vals[valid_indices[0]]
    eps_opt = eps_N[valid_indices[0]]
else:
    N_opt = N_vals[np.argmin(eps_N)]
    eps_opt = np.min(eps_N)

print("\nПошук N_opt для досягнення точності eps = 1e-12:")
print(f"Оптимальна кількість розбиттів N_opt = {N_opt}")
print(f"Досягнута точність eps_opt          = {eps_opt:.6e}\n")


# Обчислення похибки при N0 
N0 = int(np.ceil((N_opt / 10) / 8.0) * 8)
if N0 < 8:
    N0 = 8

I_N0 = simpson_composite(f, a, b, N0)
eps0 = abs(I_N0 - I0)

print(f"\nБазове обчислення для N0 = {N0}:")
print(f"I(N0) = {I_N0:.15f}")
print(f"Похибка eps0 = {eps0:.6e}\n")


# Уточнення методом Рунге-Ромберга 
I_N0_half = simpson_composite(f, a, b, N0 // 2)

# Для формули Сімпсона порядок точності p = 4, тому 2^4 - 1 = 15
I_R = I_N0 + (I_N0 - I_N0_half) / 15.0
epsR = abs(I_R - I0)

print("\nМетод Рунге-Ромберга:")
print(f"I(N0/2) [N = {N0//2}]  = {I_N0_half:.15f}")
print(f"I_R (Рунге-Ромберг) = {I_R:.15f}")
print(f"Похибка epsR        = {epsR:.6e}")
print(f"Покращення точності (eps0 / epsR): {eps0 / epsR:.2f} разів\n")


# Метод Ейткена та оцінка порядку точності p
I_N0_quarter = simpson_composite(f, a, b, N0 // 4)

# Формула Ейткена
denom_aitken = 2.0 * I_N0_half - (I_N0 + I_N0_quarter)
I_E = ((I_N0_half)**2 - I_N0 * I_N0_quarter) / denom_aitken
epsE = abs(I_E - I0)

# Оцінка порядку точності p
p_est = (1.0 / np.log(2.0)) * np.log(abs((I_N0_quarter - I_N0_half) / (I_N0_half - I_N0)))

print("\nМетод Ейткена та порядок точності p:")
print(f"I(N0/4) [N = {N0//4}]  = {I_N0_quarter:.15f}")
print(f"I_E (Ейткен)        = {I_E:.15f}")
print(f"Похибка epsE        = {epsE:.6e}")
print(f"Оцінений порядок точності p: {p_est:.4f} (Теоретичний p = 4)")
print(f"Покращення точності (eps0 / epsE): {eps0 / epsE:.2f} разів\n")


# Адаптивний алгоритм чисельного інтегрування
func_eval_count = 0

def f_eval(x):
    """Обгортка для підрахунку кількості обчислень функції"""
    global func_eval_count
    func_eval_count += 1
    return f(x)

def adaptive_simpson_rec(f_func, a_i, b_i, tol, S_i):
    """
    Рекурсивна функція адаптивного методу Сімпсона
    """
    c_i = (a_i + b_i) / 2.0
    h_i = (b_i - a_i) / 2.0
    
    # Обчислення Сімпсона на лівій і правій половині
    d_i = (a_i + c_i) / 2.0
    e_i = (c_i + b_i) / 2.0
    
    S_left = (h_i / 6.0) * (f_func(a_i) + 4.0 * f_func(d_i) + f_func(c_i))
    S_right = (h_i / 6.0) * (f_func(c_i) + 4.0 * f_func(e_i) + f_func(b_i))
    S_two = S_left + S_right
    
    # Перевірка умови збіжності (з врахуванням правила Рунге |S2 - S1| / 15 <= tol)
    if abs(S_two - S_i) <= 15.0 * tol:
        return S_two + (S_two - S_i) / 15.0
    else:
        return (adaptive_simpson_rec(f_func, a_i, c_i, tol / 2.0, S_left) +
                adaptive_simpson_rec(f_func, c_i, b_i, tol / 2.0, S_right))

def adaptive_simpson(f_func, a_val, b_val, tol):
    """Головний виклик адаптивного алгоритму"""
    global func_eval_count
    func_eval_count = 0
    
    c_val = (a_val + b_val) / 2.0
    h_val = (b_val - a_val) / 2.0
    S_initial = (h_val / 3.0) * (f_func(a_val) + 4.0 * f_func(c_val) + f_func(b_val))
    
    I_adaptive = adaptive_simpson_rec(f_func, a_val, b_val, tol, S_initial)
    return I_adaptive, func_eval_count

delta_list = [1e-3, 1e-6, 1e-9, 1e-12]
adaptive_results = []

print("\nАдаптивний алгоритм чисельного інтегрування:")
print(f"{'delta':^10} | {'Обчислений I':^18} | {'Похибка':^12} | {'Викликів f(x)':^12}")
print("-" * 60)

for d in delta_list:
    I_adap, n_evals = adaptive_simpson(f_eval, a, b, d)
    err_adap = abs(I_adap - I0)
    adaptive_results.append((d, I_adap, err_adap, n_evals))
    print(f"{d:^10.1e} | {I_adap:^18.12f} | {err_adap:^12.4e} | {n_evals:^12d}")
print("\n")


# Графік залежності похибки складової формули Сімпсона від N
plt.figure(figsize=(9, 5))
plt.loglog(N_vals, eps_N, label='Похибка ε(N) = |I(N) - I0|', color='blue')
plt.axhline(target_eps, color='red', linestyle='--', label=f'Цільова точність ε = {target_eps:.0e}')
plt.scatter([N_opt], [eps_opt], color='green', zorder=5, label=f'N_opt = {N_opt} (ε = {eps_opt:.2e})')
plt.xlabel('Кількість розбиттів N (логарифмічна шкала)')
plt.ylabel('Похибка чисельного інтегрування')
plt.title('Залежність точності формули Сімпсона від кількості розбиттів N')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()

# Графік порівняння методів підвищення точності при N0
methods = [f'Сімпсон (N0={N0})', 'Рунге-Ромберг', 'Ейткен']
errors = [eps0, epsR, epsE]

plt.figure(figsize=(8, 4.5))
bars = plt.bar(methods, errors, color=['skyblue', 'orange', 'green'])
plt.yscale('log')
plt.ylabel('Абсолютна похибка (логарифмічна шкала)')
plt.title('Порівняння точності методів чисельного інтегрування')
plt.grid(True, which="both", axis='y', ls="--")

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval * 1.5, f'{yval:.2e}', ha='center', va='bottom')

plt.tight_layout()

# Графік адаптивного алгоритму 
deltas_plot = [r[0] for r in adaptive_results]
evals_plot = [r[3] for r in adaptive_results]

plt.figure(figsize=(8, 4.5))
plt.loglog(deltas_plot, evals_plot, 'ro-', linewidth=2, markersize=7)
plt.gca().invert_xaxis()  # Інвертуємо вісь, щоб delta зменшувалася зліва направо
plt.xlabel('Задана точність delta (логарифмічна шкала)')
plt.ylabel('Кількість обчислень функції f(x)')
plt.title('Залежність обчислювальної складності адаптивного алгоритму від delta')
plt.grid(True, which="both", ls="--")
plt.tight_layout()

plt.show()