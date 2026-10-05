"""
AutoPlot Civil.

Application for concrete compressive strength, sieve analysis, Proctor
compaction, basic statistics, and unit conversion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.engineering import (
    calculate_compressive_strength,
    process_sieve_data,
    analyze_compaction,
    calculate_statistics,
)
from src.visualization import create_grain_size_chart, create_compaction_chart

st.set_page_config(page_title="AutoPlot Civil", layout="wide")
st.title("AutoPlot Civil")
st.caption("Civil engineering calculations and charts")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Concrete Strength", "Sieve Analysis", "Proctor Compaction", "Statistics"]
)

with tab1:
    st.subheader("Concrete Compressive Strength")
    st.caption("The h/d ratio correction follows SNI 1974:2011.")
    c1, c2, c3 = st.columns(3)
    load = c1.number_input("Maximum load (kN)", value=450.0, step=10.0)
    height = c2.number_input("Specimen height (mm)", value=300.0, step=5.0)
    diameter = c3.number_input("Diameter (mm)", value=150.0, step=5.0)
    if st.button("Compute", key="concrete"):
        area = np.pi * (diameter ** 2) / 4
        result = calculate_compressive_strength(load, area, height, diameter)
        st.metric("Corrected compressive strength (MPa)", f"{result.value:.2f}")
        st.write(result)

with tab2:
    st.subheader("Sieve Analysis")
    st.caption("Grain-size distribution curve and USCS/AASHTO classification.")
    sample = pd.DataFrame({
        "aperture_mm": [37.5, 25.0, 19.0, 9.5, 4.75, 2.0, 0.85, 0.425, 0.25, 0.15, 0.075],
        "pct_finer":   [100.0, 98.0, 95.0, 88.0, 76.0, 61.0, 48.0, 35.0, 24.0, 15.0, 6.0],
    })
    edited = st.data_editor(sample, num_rows="dynamic", key="sieve")
    if st.button("Process", key="sieve_btn"):
        result = process_sieve_data(edited)
        st.write(result)
        st.plotly_chart(create_grain_size_chart(edited), use_container_width=True)

with tab3:
    st.subheader("Proctor Compaction")
    st.caption("Compaction curve, MDD, OMC, and the zero air voids line.")
    sample = pd.DataFrame({"water_content": [8, 10, 12, 14, 16, 18],
                           "dry_density": [1.62, 1.71, 1.78, 1.80, 1.76, 1.68]})
    edited = st.data_editor(sample, num_rows="dynamic", key="comp")
    if st.button("Analyse", key="comp_btn"):
        result = analyze_compaction(edited)
        st.write(result)
        st.plotly_chart(create_compaction_chart(edited), use_container_width=True)

with tab4:
    st.subheader("Basic Statistics")
    st.caption("Mean, standard deviation, coefficient of variation, and characteristic strength.")
    text = st.text_area("Data (one value per line)",
                        "32.5\\n34.1\\n31.8\\n33.7\\n35.0\\n32.9\\n33.2")
    if st.button("Compute", key="stat"):
        try:
            values = [float(x) for x in text.split() if x.strip()]
            st.write(calculate_statistics(values))
        except ValueError:
            st.error("Enter valid numbers.")