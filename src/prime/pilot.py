import math
from statistics import NormalDist


z = NormalDist().inv_cdf


def n_per_group(
    sigma: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> int:
    """Количество клиентов на каждую группу."""

    if sigma < 0 or mde <= 0:
        raise ValueError("sigma должна быть >= 0, mde > 0")

    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha и power должны лежать между 0 и 1")

    if comparisons < 1:
        raise ValueError("comparisons должно быть >= 1")

    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)

    return math.ceil(
        2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2
    )


def mde_for_n(
    sigma: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> float:
    """Минимальный обнаруживаемый эффект для заданной выборки."""

    if sigma < 0 or n <= 0:
        raise ValueError("sigma должна быть >= 0, n > 0")

    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha и power должны лежать между 0 и 1")

    if comparisons < 1:
        raise ValueError("comparisons должно быть >= 1")

    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)

    return math.sqrt(2) * (za + zb) * sigma / math.sqrt(n)


def diff_ci(t, c, alpha: float = 0.05) -> dict:
    """Разность средних test-control, доверительный интервал и p-value."""

    if len(t) == 0 or len(c) == 0:
        raise ValueError("Группы не должны быть пустыми")

    if not 0 < alpha < 1:
        raise ValueError("alpha должен лежать между 0 и 1")

    d = t.mean() - c.mean()

    se = math.sqrt(
        t.var() / len(t) + c.var() / len(c)
    )

    q = NormalDist().inv_cdf(1 - alpha / 2)

    if se == 0:
        p = 1.0 if d == 0 else 0.0
    else:
        p = 2 * (1 - NormalDist().cdf(abs(d / se)))

    return {
        "diff": d,
        "lo": d - q * se,
        "hi": d + q * se,
        "p": p,
    }


def srm_pvalue(sizes) -> float:
    """P-value проверки Sample Ratio Mismatch для 2 или 3 групп."""

    sizes = list(sizes)

    if len(sizes) not in (2, 3):
        raise ValueError("Поддерживаются только 2 или 3 группы")

    if any(s < 0 for s in sizes) or sum(sizes) <= 0:
        raise ValueError("Размеры групп должны быть неотрицательными")

    n = sum(sizes)
    expected = n / len(sizes)

    chi2 = sum(
        (s - expected) ** 2 / expected
        for s in sizes
    )

    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))

    return math.exp(-chi2 / 2)


def cuped(y, x):
    """CUPED-коррекция метрики и коэффициент theta."""

    if len(y) != len(x) or len(y) == 0:
        raise ValueError("y и x должны иметь одинаковую ненулевую длину")

    variance = x.var()

    if variance == 0:
        raise ValueError("Дисперсия ковариаты равна нулю")

    theta = y.cov(x) / variance

    return y - theta * (x - x.mean()), theta