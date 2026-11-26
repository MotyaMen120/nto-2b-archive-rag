"""Exact output-unit aliases; unknown units are preserved."""
ALIASES = {"rub": "руб", "rubles": "руб", "руб.": "руб", "руб": "руб",
           "mln rub": "млн руб", "млн руб.": "млн руб", "млн руб": "млн руб",
           "тыс. руб.": "тыс. руб", "млрд руб.": "млрд руб", "процентов": "%", "м2": "кв. м",
           "days": "дней", "months": "месяцев", "times": "раз"}

def normalize_unit(unit):
    """Normalize only controlled aliases, without trimming or changing case."""
    return ALIASES.get(unit, unit)
