# -*- coding: utf-8 -*-
"""
MedDose Calculator v3.6
Улучшенная визуализация: фокус на эффективности (MEC), без перегрузки MSC на графике.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ───────────────────────────────────────
# БАЗА ПРЕПАРАТОВ С КЛИНИЧЕСКИ РЕАЛИСТИЧНЫМИ ПАРАМЕТРАМИ
# ───────────────────────────────────────
DRUGS = {
    "Парацетамол": {
        "name": "Парацетамол",
        "Vd_per_kg": 0.95,
        "T_half_base": 2.0,
        "T_half_factor_female": 1.15,
        "T_half_factor_elderly": 1.3,
        "MEC": 5,                   # мг/л — реалистичное значение
        "MSC": 150,
        "max_single_dose_mg_per_kg": 15,
        "max_daily_dose_mg_per_kg": 60,
        "absolute_max_daily_mg": 4000,
        "min_interval_h": 4,
        "unit": "мг"
    },
    "Ибупрофен": {
        "name": "Ибупрофен",
        "Vd_per_kg": 0.7,
        "T_half_base": 2.0,
        "T_half_factor_female": 1.2,
        "T_half_factor_elderly": 1.4,
        "MEC": 4,                   # мг/л — реалистичное значение
        "MSC": 100,
        "max_single_dose_mg_per_kg": 10,
        "max_daily_dose_mg_per_kg": 30,
        "absolute_max_daily_mg": 2400,
        "min_interval_h": 6,
        "unit": "мг"
    }
}

def calculate_concentration(drug, mass_kg, age, sex, dose_mg, interval_h, total_hours=24):
    Vd = drug["Vd_per_kg"] * mass_kg
    C0 = dose_mg / Vd
    T_half = drug["T_half_base"]
    if sex == "Женский":
        T_half *= drug["T_half_factor_female"]
    if age >= 65:
        T_half *= drug["T_half_factor_elderly"]
    k = np.log(2) / T_half

    t = np.arange(0, total_hours + 0.1, 0.1)
    C = np.zeros_like(t)
    num_doses = int(total_hours // interval_h) + 1
    for i in range(num_doses):
        dose_time = i * interval_h
        mask = t >= dose_time
        C[mask] += C0 * np.exp(-k * (t[mask] - dose_time))
    return t, C

def calculate_safe_dosing(drug, mass_kg, age):
    single_dose_max = drug["max_single_dose_mg_per_kg"] * mass_kg
    daily_dose_by_weight = drug["max_daily_dose_mg_per_kg"] * mass_kg
    daily_dose_absolute = drug["absolute_max_daily_mg"]
    
    daily_dose_max = min(daily_dose_by_weight, daily_dose_absolute) if mass_kg < 50 else daily_dose_absolute
    
    min_interval = drug["min_interval_h"]
    max_doses_per_day = 24 // min_interval
    
    single_dose_from_daily = daily_dose_max / max_doses_per_day
    recommended_single = min(single_dose_max, single_dose_from_daily)
    
    return {
        "recommended_single_mg": round(recommended_single, 1),
        "max_daily_mg": round(daily_dose_max, 1),
        "min_interval_h": min_interval,
        "max_doses_per_day": int(max_doses_per_day)
    }

def main():
    st.set_page_config(page_title="MedDose Calculator", page_icon="💊", layout="centered")
    st.title("💊 MedDose Calculator")
    st.markdown("### Персонализированный расчёт дозировки лекарств")

    st.sidebar.header("Данные пациента")
    mass_kg = st.sidebar.number_input("Масса тела (кг)", min_value=5.0, max_value=200.0, value=70.0, step=1.0)
    age = st.sidebar.number_input("Возраст (лет)", min_value=1, max_value=120, value=30)
    sex = st.sidebar.radio("Пол", ["Мужской", "Женский"])

    drug_name = st.selectbox("Выберите препарат", list(DRUGS.keys()))
    drug = DRUGS[drug_name]

    dosing = calculate_safe_dosing(drug, mass_kg, age)
    
    st.subheader("📋 Рекомендуемая дозировка")
    st.info(f"""
    **Разовая доза**: до **{dosing['recommended_single_mg']} {drug['unit']}**  
    **Суточная доза**: не более **{dosing['max_daily_mg']} {drug['unit']}**  
    **Минимальный интервал**: **{dosing['min_interval_h']} ч**  
    **Максимум приёмов в сутки**: **{dosing['max_doses_per_day']} раз**
    """)

    st.markdown("---")
    st.subheader("🔍 Проверка вашего режима приёма препарата")

    col1, col2 = st.columns(2)
    with col1:
        user_dose = st.number_input(
            f"Разовая доза ({drug['unit']})",
            min_value=1.0,
            max_value=float(dosing["max_daily_mg"]),
            value=dosing["recommended_single_mg"]
        )
    with col2:
        user_interval = st.number_input(
            "Интервал между приёмами (ч)",
            min_value=float(drug["min_interval_h"]),
            max_value=12.0,
            value=float(dosing["min_interval_h"]),
            step=1.0
        )

    doses_per_day = 24 / user_interval
    user_daily_dose = user_dose * doses_per_day

    safe_single = user_dose <= dosing["recommended_single_mg"]
    safe_daily = user_daily_dose <= dosing["max_daily_mg"]
    safe_interval = user_interval >= drug["min_interval_h"]

    st.markdown("### 📊 Анализ режима дозирования")
    if safe_single and safe_daily and safe_interval:
        st.success("✅ Режим дозирования безопасен и эффективен!")
        show_plot = True
    else:
        st.error("❌ Обнаружены нарушения рекомендаций:")
        if not safe_single:
            st.warning(f"• Разовая доза превышает рекомендованную ({dosing['recommended_single_mg']} {drug['unit']})")
        if not safe_daily:
            st.warning(f"• Суточная доза ({user_daily_dose:.1f} {drug['unit']}) превышает допустимый максимум ({dosing['max_daily_mg']} {drug['unit']})")
        if not safe_interval:
            st.warning(f"• Интервал короче минимально допустимого ({drug['min_interval_h']} ч)")
        show_plot = False

    # ───────────────────────────────────────
    # ОСНОВНОЙ ГРАФИК С УЛУЧШЕННОЙ ВИЗУАЛИЗАЦИЕЙ
    # ───────────────────────────────────────
    if show_plot:
        t, C = calculate_concentration(drug, mass_kg, age, sex, user_dose, user_interval)
        MEC = drug["MEC"]
        MSC = drug["MSC"]
        min_steady = np.min(C[int(user_interval * 10):])
        max_steady = np.max(C[int(user_interval * 10):])

        # Масштаб: от 0 до 1.3 × максимума на графике
        max_conc = np.max(C)
        y_max = max_conc * 1.3

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, C, 'b-', linewidth=2.5, label='Концентрация в крови')
        ax.axhline(MEC, color='orange', linestyle='--', linewidth=2, label=f'MEC = {MEC} мг/л')
        # MSC НЕ отображается на графике!
        ax.fill_between(t, MEC, y_max, color='green', alpha=0.15, label='Эффективная зона')
        ax.set_xlim(0, 24)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Время (часы)", fontsize=12)
        ax.set_ylabel("Концентрация (мг/л)", fontsize=12)
        ax.set_title(f"{drug_name}: {user_dose} {drug['unit']} каждые {user_interval} ч", fontsize=13)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.legend()
        st.pyplot(fig)

        # Анализ эффективности
        if min_steady >= MEC:
            st.success("✅ Концентрация препарата остаётся выше минимальной эффективной (MEC).")
        else:
            st.warning(f"⚠️ Минимальная концентрация ({min_steady:.1f} мг/л) ниже MEC ({MEC} мг/л).")

        # Словесное упоминание безопасности (MSC)
        if max_steady > MSC:
            st.error(f"❗ Пиковая концентрация ({max_steady:.1f} мг/л) превышает максимальную безопасную ({MSC} мг/л). Риск токсичности!")
        else:
            st.info(f"ℹ️ Пиковая концентрация ({max_steady:.1f} мг/л) в пределах безопасного уровня (≤ {MSC} мг/л).")

        st.caption(f"Суточная доза: {user_daily_dose:.1f} {drug['unit']} (макс. {dosing['max_daily_mg']} {drug['unit']})")

    # ───────────────────────────────────────
    # СРАВНИТЕЛЬНЫЙ ГРАФИК С УЛУЧШЕННОЙ ВИЗУАЛИЗАЦИЕЙ
    # ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Сравнительный анализ: ваш режим приёма препарата и рекомендуемый")

    dose_user = user_dose
    interval_user = user_interval
    dose_recommended = dosing["recommended_single_mg"]
    interval_recommended = float(dosing["min_interval_h"])

    t_user, C_user = calculate_concentration(drug, mass_kg, age, sex, dose_user, interval_user)
    t_rec, C_rec = calculate_concentration(drug, mass_kg, age, sex, dose_recommended, interval_recommended)

    # Общий масштаб для обоих кривых
    max_conc = max(np.max(C_user), np.max(C_rec))
    y_max = max_conc * 1.3

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(t_user, C_user, 'b-', linewidth=2, label=f'Ваш режим: {dose_user} мг каждые {interval_user} ч')
    ax2.plot(t_rec, C_rec, 'g-', linewidth=2, label=f'Рекомендованный: {dose_recommended} мг каждые {interval_recommended} ч')
    ax2.axhline(drug["MEC"], color='orange', linestyle='--', linewidth=2, label=f'MEC = {drug["MEC"]} мг/л')
    ax2.fill_between(t_user, drug["MEC"], y_max, color='green', alpha=0.15, label='Эффективная зона')
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, y_max)
    ax2.set_xlabel("Время (часы)", fontsize=12)
    ax2.set_ylabel("Концентрация (мг/л)", fontsize=12)
    ax2.set_title(f"Сравнение: ваш выбор vs рекомендация ({drug_name})", fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    st.pyplot(fig2)

    min_user = np.min(C_user[int(interval_user*10):]) if len(C_user[int(interval_user*10):]) > 0 else 0
    min_rec = np.min(C_rec[int(interval_recommended*10):]) if len(C_rec[int(interval_recommended*10):]) > 0 else 0

    st.info(f"""
    **Ваш режим**:  
    Мин. концентрация = {min_user:.1f} мг/л → {'в' if min_user >= drug['MEC'] else 'ниже'} эффективной зоны {'✅' if min_user >= drug['MEC'] else '❌'}

    **Рекомендованный режим**:  
    Мин. концентрация = {min_rec:.1f} мг/л → {'в' if min_rec >= drug['MEC'] else 'ниже'} эффективной зоны {'✅' if min_rec >= drug['MEC'] else '❌'}
    """)

    # ───────────────────────────────────────
    # ПОЯСНЕНИЕ О МОДЕЛИ
    # ───────────────────────────────────────
    st.markdown("---")
    st.caption("""
    ℹ️ **Примечание о модели**:  
    График фокусируется на **минимальной эффективной концентрации (MEC)** — ниже неё лекарство не работает.  
    **Максимальная безопасная концентрация (MSC)** не отображается на графике для лучшей наглядности,  
    но проверяется программно и сообщается текстом.  
    В реальности анальгетики могут действовать дольше, чем обнаруживаются в крови.  
    Программа — образовательный инструмент, не заменяющий врача.
    """)

if __name__ == "__main__":
    main()