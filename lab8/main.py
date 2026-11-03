import os
import numpy as np
import matplotlib.pyplot as plt

def F(x):
    """Вихідна трансендентна функція"""
    return np.exp(x) - np.sin(x) - 2.0

def dF(x):
    """Перша похідна F'(x)"""
    return np.exp(x) - np.cos(x)

def ddF(x):
    """Друга похідна F''(x)"""
    return np.exp(x) + np.sin(x)

# Табуляція функції
def tabulate_function(a=0.0, b=3.0, h=0.1, filename="tabulation.txt"):
    x_vals = np.arange(a, b + h/2, h)
    y_vals = F(x_vals)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("x\t\tF(x)\n")
        for xv, yv in zip(x_vals, y_vals):
            f.write(f"{xv:.4f}\t{yv:.6f}\n")
            
    print(f"Табуляція записана у файл: {filename}")
    return x_vals, y_vals

# Ітераційні методи для нелінійних рівнянь
def simple_iteration(x0, eps=1e-10, max_iter=1000):
    tau = -0.15
    x_curr = x0
    history = [x_curr]
    for k in range(1, max_iter + 1):
        x_next = x_curr + tau * F(x_curr)
        history.append(x_next)
        if abs(F(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, k
        x_curr = x_next
    return x_curr, max_iter

def newton_method(x0, eps=1e-10, max_iter=1000):
    x_curr = x0
    for k in range(1, max_iter + 1):
        df = dF(x_curr)
        if abs(df) < 1e-14:
            break
        x_next = x_curr - F(x_curr) / df
        if abs(F(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, k
        x_curr = x_next
    return x_curr, max_iter

def chebyshev_method(x0, eps=1e-10, max_iter=1000):
    x_curr = x0
    for k in range(1, max_iter + 1):
        fx = F(x_curr)
        df = dF(x_curr)
        ddf = ddF(x_curr)
        if abs(df) < 1e-14:
            break
        x_next = x_curr - fx / df - 0.5 * (fx**2 * ddf) / (df**3)
        if abs(F(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, k
        x_curr = x_next
    return x_curr, max_iter

def chord_method(x0, x1, eps=1e-10, max_iter=1000):
    x_prev = x0
    x_curr = x1
    for k in range(1, max_iter + 1):
        f_curr = F(x_curr)
        f_prev = F(x_prev)
        if abs(f_curr - f_prev) < 1e-14:
            break
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        if abs(F(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, k
        x_prev = x_curr
        x_curr = x_next
    return x_curr, max_iter

def parabolas_method(x0, x1, x2, eps=1e-10, max_iter=1000):
    xn2, xn1, xn = x0, x1, x2
    for k in range(1, max_iter + 1):
        f0, f1, f2 = F(xn2), F(xn1), F(xn)
        h1 = xn1 - xn2
        h2 = xn - xn1
        if abs(h1) < 1e-14 or abs(h2) < 1e-14:
            break
        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2
        
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            delta = -c / b if abs(b) > 1e-14 else 0
        else:
            sd = np.sqrt(discriminant)
            den = b + sd if abs(b + sd) > abs(b - sd) else b - sd
            delta = -2 * c / den if abs(den) > 1e-14 else 0
            
        x_next = xn + delta
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, k
        xn2, xn1, xn = xn1, xn, x_next
    return xn, max_iter

def inverse_interpolation(x0, x1, x2, eps=1e-10, max_iter=1000):
    xn2, xn1, xn = x0, x1, x2
    for k in range(1, max_iter + 1):
        y0, y1, y2 = F(xn2), F(xn1), F(xn)
        # Квадратна зворотна інтерполяція (формула Лагранжа для 3 вузлів при y=0)
        denom1 = (y0 - y1) * (y0 - y2)
        denom2 = (y1 - y0) * (y1 - y2)
        denom3 = (y2 - y0) * (y2 - y1)
        
        if abs(denom1) < 1e-14 or abs(denom2) < 1e-14 or abs(denom3) < 1e-14:
            break
            
        x_next = (y1 * y2 * xn2) / denom1 + (y0 * y2 * xn1) / denom2 + (y0 * y1 * xn/ denom3 if denom3 != 0 else xn)
        # Виправлена формула для x_next
        x_next = (y1 * y2 * xn2) / denom1 + (y0 * y2 * xn1) / denom2 + (y0 * y1 * xn) / denom3
        
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, k
        xn2, xn1, xn = xn1, xn, x_next
    return xn, max_iter


# Збереження коефіцієнтів у файл
def save_poly_coefficients(filename="poly_coeffs.txt"):
    coeffs = [-2.0, 1.0, -2.0, 1.0]
    with open(filename, "w", encoding="utf-8") as f:
        for c in coeffs:
            f.write(f"{c}\n")
    return coeffs

def load_poly_coefficients(filename="poly_coeffs.txt"):
    return np.loadtxt(filename)

# Оцінка многочлена через схему Горнера
def horner_eval(coeffs, x):
    m = len(coeffs) - 1
    b = coeffs[m]
    for i in range(m - 1, -1, -1):
        b = coeffs[i] + x * b
    return b

#  Метод Ньютона за схемою Горнера для дійсного кореня
def newton_horner(coeffs, x0, eps=1e-10, max_iter=1000):
    m = len(coeffs) - 1
    xn = x0
    for k in range(1, max_iter + 1):
        b = [0.0] * (m + 1)
        b[m] = coeffs[m]
        for i in range(m - 1, -1, -1):
            b[i] = coeffs[i] + xn * b[i+1]
        b0 = b[0]
        
        c = [0.0] * m
        c[m-1] = b[m]
        for i in range(m - 2, 0, -1):
            c[i] = b[i+1] + xn * c[i+1]
        # с1 для похідної
        c1 = b[1] + xn * c[2] if m >= 2 else b[1]
        
        if abs(c1) < 1e-14:
            break
            
        x_next = xn - b0 / c1
        if abs(b0) < eps and abs(x_next - xn) < eps:
            return x_next, k
        xn = x_next
    return xn, max_iter

# Метод Ліна для комплексних коренів многочлена 3-го порядку
def lin_method(coeffs, alpha0, beta0, eps=1e-10, max_iter=1000):
    a0, a1, a2, a3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    alpha, beta = alpha0, beta0
    
    for k in range(1, max_iter + 1):
        p = -2 * alpha
        q = alpha**2 + beta**2
        
        # Обчислення неповних часток b за схемою Ліна для m=3
        # F(x) = (x^2 + px + q)(b_3 x + b_2) + b_1 x + b_0
        b3 = a3
        b2 = a2 - p * b3
        b1 = a1 - p * b2 - q * b3
        b0 = a0 - q * b2  # тут b0 залишок від ділення
        
        if abs(b2) < 1e-14:
            break
            
        q1 = a0 / b2
        p1 = (a1 * b2 - a0 * b3) / (b2**2)
        
        alpha1 = -p1 / 2.0
        val_under_sqrt = q1 - alpha1**2
        beta1 = np.sqrt(abs(val_under_sqrt)) if val_under_sqrt >= 0 else 0.0
        
        if abs(alpha1 - alpha) < eps and abs(beta1 - beta) < eps:
            return alpha1, beta1, k
            
        alpha, beta = alpha1, beta1
        
    return alpha, beta, max_iter


if __name__ == "__main__":
    
    tabulate_function()
    
    x_root1_approx = 0.5
    x_root2_approx = 2.1
    
    print("\nУточнення першого Кореня")
    r1_si, it_si = simple_iteration(x_root1_approx)
    r1_newt, it_newt = newton_method(x_root1_approx)
    r1_cheb, it_cheb = chebyshev_method(x_root1_approx)
    r1_chd, it_chd = chord_method(0.1, x_root1_approx)
    r1_par, it_par = parabolas_method(0.1, 0.3, x_root1_approx)
    r1_inv, it_inv = inverse_interpolation(0.1, 0.3, x_root1_approx)
    
    print(f"Проста ітерація    : корінь = {r1_si:.8f}, ітерацій = {it_si}")
    print(f"Метод Ньютона      : корінь = {r1_newt:.8f}, ітерацій = {it_newt}")
    print(f"Метод Чебишева     : корінь = {r1_cheb:.8f}, ітерацій = {it_cheb}")
    print(f"Метод хорд         : корінь = {r1_chd:.8f}, ітерацій = {it_chd}")
    print(f"Метод парабол      : корінь = {r1_par:.8f}, ітерацій = {it_par}")
    print(f"Зворотна інтерпол. : корінь = {r1_inv:.8f}, ітерацій = {it_inv}")

    print("\nУточнення другого Кореня 2")
    r2_newt, it2_newt = newton_method(x_root2_approx)
    r2_cheb, it2_cheb = chebyshev_method(x_root2_approx)
    print(f"Метод Ньютона      : корінь = {r2_newt:.8f}, ітерацій = {it2_newt}")
    print(f"Метод Чебишева     : корінь = {r2_cheb:.8f}, ітерацій = {it2_cheb}")

    coeffs_file = "poly_coeffs.txt"
    coeffs = save_poly_coefficients(coeffs_file)
    loaded_coeffs = load_poly_coefficients(coeffs_file)
    
    print("\nРозв'язок алгебраїчного рівняння 3-го порядку")
    print(f"Коефіцієнти многочлена: {loaded_coeffs}")
    
    real_root, iter_nh = newton_horner(loaded_coeffs, 2.0)
    print(f"Дійсний корінь (Ньютон-Горнер): {real_root:.8f}, ітерацій = {iter_nh}")
    
    alpha_res, beta_res, iter_lin = lin_method(loaded_coeffs, 0.0, 1.0)
    print(f"Комплексні корені (Метод Ліна): {alpha_res:.8f} ± i*{beta_res:.8f}, ітерацій = {iter_lin}")