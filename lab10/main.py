import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    """Права частина рівняння"""
    return y - x

def exact_solution(x):
    """Точний аналітичний розв'язок y(x) на відрізку [a, b] з початковою умовою"""
    return x + 1.0 + np.exp(x)


def runge_kutta_4(f, x0, y0, x_end, h):
    """Реалізація класичного методу Рунге-Кутта 4-го порядку"""
    x_vals = np.arange(x0, x_end + h/2, h)
    y_vals = np.zeros(len(x_vals))
    y_vals[0] = y0
    
    for i in range(len(x_vals) - 1):
        xn = x_vals[i]
        yn = y_vals[i]
        
        k1 = f(xn, yn)
        k2 = f(xn + 0.5*h, yn + 0.5*h*k1)
        k3 = f(xn + 0.5*h, yn + 0.5*h*k2)
        k4 = f(xn + h, yn + h*k3)
        
        y_vals[i+1] = yn + (h / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        
    return x_vals, y_vals


def runge_kutta_adaptive(f, x0, y0, x_end, h_init=0.1, eps=1e-5):
    """Адаптивний метод Рунге-Кутта з автоматичною зміною кроку за точністю"""
    x_curr = x0
    y_curr = y0
    h = h_init

    x_path = [x_curr]
    y_path = [y_curr]
    h_path = []

    while x_curr < x_end:
        if x_curr + h > x_end:
            h = x_end - x_curr

        _, y_h = runge_kutta_4(f, x_curr, y_curr, x_curr + h, h)
        y_next_h = y_h[-1]

        _, y_half1 = runge_kutta_4(
            f, x_curr, y_curr, x_curr + h / 2, h / 2
        )

        _, y_half2 = runge_kutta_4(
            f, x_curr + h / 2, y_half1[-1], x_curr + h, h / 2
        )

        y_next_half = y_half2[-1]

        error = (16.0 / 15.0) * abs(y_next_half - y_next_h)

        if error <= eps:
            x_curr += h
            y_curr = y_next_half

            x_path.append(x_curr)
            y_path.append(y_curr)
            h_path.append(h)

            if error < eps / 32.0:
                h *= 2.0
        else:
            h /= 2.0

            if h < 1e-12:
                break

    return np.array(x_path), np.array(y_path), np.array(h_path)


def adams_bashforth_moulton_2(f, x0, y0, x_end, h, eps=1e-6):
    """Багатокроковий метод Адамса (прогноз + корекція)"""
    x_init, y_init = runge_kutta_4(f, x0, y0, x0 + h, h)
    
    x_vals = list(x_init)
    y_vals = list(y_init)
    
    np_err_list = []
    
    while x_vals[-1] < x_end - h/2:
        xn = x_vals[-1]
        yn = y_vals[-1]
        yn_prev = y_vals[-2]
        xn_prev = x_vals[-2]
        
        fn = f(xn, yn)
        fn_prev = f(xn_prev, yn_prev)
        
        # Етап прогнозу (екстраполяція)
        y_np = yn + 0.5 * h * (3.0 * fn - fn_prev)
        
        # Етап корекції (уточнення через ітерації)
        xn_next = xn + h
        y_kop = y_np
        for _ in range(2):
            y_kop = yn + 0.5 * h * (f(xn_next, y_kop) + fn)
            
        # Оцінка похибки через різницю (y_kop - y_np)
        local_err = (1.0 / 6.0) * abs(y_kop - y_np)
        
        x_vals.append(xn_next)
        y_vals.append(y_kop)
        np_err_list.append(local_err)
        
    return np.array(x_vals), np.array(y_vals), np.array(np_err_list)


if __name__ == "__main__":
    a, b = 0.0, 1.0
    y0 = 2.0
    h_fixed = 0.01
    
    # Обчислення точного розв'язку
    x_exact = np.linspace(a, b, 200)
    y_exact = exact_solution(x_exact)
    
    # Чисельний розв'язок методом Рунге-Кутта 4
    x_rk, y_rk = runge_kutta_4(f, a, y0, b, h_fixed)
    exact_rk = exact_solution(x_rk)
    
    # Обчислення локальної похибки через точне значення phi_n = y_n - y(x_n)
    varphi_rk = y_rk - exact_rk
    
    # Автоматичний вибір кроку h(x)
    x_ad, y_ad, h_ad = runge_kutta_adaptive(f, a, y0, b, h_init=0.1, eps=1e-5)
    
    # Метод Адамса 2-го порядку
    x_adams, y_adams, adams_err = adams_bashforth_moulton_2(f, a, y0, b, h_fixed)
    varphi_adams = y_adams - exact_solution(x_adams)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    
    axs[0, 0].plot(x_exact, y_exact, 'k-', label="Точний розв'язок", linewidth=2)
    axs[0, 0].plot(x_rk, y_rk, 'r--', label="Рунге-Кутт 4", alpha=0.8)
    axs[0, 0].set_title("Чисельні та точний розв'язки")
    axs[0, 0].grid(True, linestyle="--")
    axs[0, 0].legend()
    
    axs[0, 1].plot(x_rk, varphi_rk, 'b-', label="Похибка phi_n")
    axs[0, 1].set_title("Локальна похибка Рунге-Кутта 4")
    axs[0, 1].grid(True, linestyle="--")
    axs[0, 1].legend()
    
    axs[1, 0].plot(x_ad[1:], h_ad, 'g-o', markersize=3, label="Крок h(x)")
    axs[1, 0].set_title("Автоматичний вибір кроку")
    axs[1, 0].grid(True, linestyle="--")
    axs[1, 0].legend()
    
    axs[1, 1].plot(x_adams, varphi_adams, 'm-', label="Похибка Адамса")
    axs[1, 1].set_title("Локальна похибка методу Адамса 2")
    axs[1, 1].grid(True, linestyle="--")
    axs[1, 1].legend()
    
    plt.tight_layout()
    plt.show()