# utils.py

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Reference ranges for lab values
REFERENCE_RANGES = {
    "Hemoglobin": (13, 17),
    "Total leukocyte count": (4000, 11000),
    "GRBS/ random blood sugar (mg/dL)": (70, 140),
    "Total protein (g/dl)": (6.6, 8.7),
    "Serum albumin (g/dl)": (3.5, 5.2),
    "Total bilirubin": (0, 1.2),
    "Direct bilirubin": (0, 0.2),
    "AST": (0, 41),
    "ALT": (0, 40),
    "Prothrombin time (PT)": (11, 16),
    "Activated partial thromboplastin time (APTT)": (26, 40),
    "Urea (mg/dL)": (16.6, 48.5),
    "Creatinine": (0.7, 1.2),
    "Platelet": (50000, 150000),
    "Sodium (mM/l)": (136, 145),
    "Potassium (mM/l)": (3.5, 5.1),
    "pH": (7.35, 7.45),
    "PO2 (mmHg)": (60, 100),
    "Bicarb (mmol/l)": (20, 29),
    "Lactate (mol/L)": (1.3, 2.0),
    "CPK (U/L)": (0, 190),
    "CPK-MB (U/L)": (0, 25),
    "CRP (mg/L)": (0, 6),
    "Procalcitonin (ng/ml)": (0, 0.5),
    "Serum ferritin (ng/ml)": (30, 400),
    "LDH (U/L)": (0, 250),
    "d Dimer (mcg/ml)": (0, 0.5)
}

NUMERICAL_ACTION_HINTS = {
    "Hemoglobin": "Suggests anemia or blood loss — consider CBC review and bleeding source.",
    "Total leukocyte count": "May indicate infection or inflammation — evaluate infection markers and cultures.",
    "Serum albumin (g/dl)": "Low values may suggest malnutrition or hepatic dysfunction — consider nutritional support and LFT review.",
    "Urea (mg/dL)": "High urea may suggest dehydration or renal failure — check creatinine, electrolytes, hydration status.",
    "Creatinine": "Elevated levels suggest impaired renal function — monitor renal profile and adjust medications.",
    "CRP (mg/L)": "Indicates systemic inflammation — evaluate for infection, sepsis, or autoimmune process.",
    "Procalcitonin (ng/ml)": "May signal bacterial sepsis — start empirical antibiotics and monitor trend.",
    "LDH (U/L)": "Suggests tissue breakdown — assess for hemolysis, liver damage, or tumor lysis.",
    "d Dimer (mcg/ml)": "May indicate thromboembolic event — consider Doppler/CT angiography for PE/DVT.",
    "Platelet": "Low platelets increase bleeding risk — evaluate for DIC, sepsis, or marrow suppression.",
    "Sodium (mM/l)": "Abnormal sodium may reflect SIADH, dehydration, or adrenal dysfunction — treat accordingly.",
    "Potassium (mM/l)": "Imbalance can cause arrhythmias — correct cautiously and monitor ECG.",
    "pH": "Deranged pH may point to metabolic or respiratory acidosis/alkalosis — review ABG and lactate.",
    "Lactate (mol/L)": "High lactate = poor perfusion or sepsis — consider fluid resuscitation and antibiotics.",
    "ALT": "Elevated ALT may suggest hepatocellular injury — evaluate hepatitis, ischemia, or drugs.",
    "AST": "Elevated AST may indicate liver, cardiac, or muscle injury — compare with ALT and CPK.",
    "Direct bilirubin": "Elevation indicates possible cholestasis — evaluate for biliary obstruction or sepsis.",
    "Total bilirubin": "Hyperbilirubinemia may reflect liver dysfunction or hemolysis — check LFT and hemolysis labs."
}

CATEGORICAL_HINTS = {
    "Airway & breathing": "Severe derangements may require mechanical ventilation.",
    "Renal": "Dialysis may be needed in worsening AKI or CKD.",
    "Hematological": "Consider blood product transfusion if bleeding risk high.",
    "Gastrointestinal and Hepatic": "Evaluate for liver failure or ileus.",
    "CO-MORBIDITY": "Multiple comorbidities may worsen prognosis—multi-disciplinary review advised.",
    "Complications": "Critical complications like ARDS or CKD demand aggressive management.",
    "ECHO": "Severe dysfunction indicates need for cardiology consult."
}



def generate_pdf_table(explanation_df, mode="Model View"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=40,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    elements = []

    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    title_style = ParagraphStyle(
    name='SimpleBoldTitle',
    fontSize=16,
    leading=20,
    alignment=1,  # Centered
    fontName='Helvetica-Bold',
    textColor=colors.HexColor("#003366"),
    spaceAfter=12
    )

    elements.append(Paragraph("<u>Patient-Specific Mortality Risk Explanation</u>", title_style))
    elements.append(Spacer(1, 6))

    explanation_df_sorted = explanation_df.sort_values(by="Contribution", ascending=True)
    top_contributors = explanation_df_sorted.head(5)
    top_summary = "<br/>".join([
        f"• <b>{row['Feature']}</b>: {float(row['Contribution']):+.2f}"
        for _, row in top_contributors.iterrows()
    ])
    elements.append(Paragraph("<b>Top Contributors to Mortality Risk:</b>", styles["Normal"]))
    elements.append(Paragraph(top_summary, ParagraphStyle("Summary", fontSize=9, leading=12)))
    elements.append(Spacer(1, 12))

    wrap_style = ParagraphStyle(name='Wrap', fontSize=8, leading=10, wordWrap='LTR')

    # Dynamic columns
    if mode == "Model View":
        table_data = [["Feature", "Input Value", "Risk Weight", "Risk Impact"]]
        col_widths = [140, 140, 90, 170]
    else:
        table_data = [["Feature", "Input Value", "Risk Impact"]]
        col_widths = [160, 180, 200]

    row_colors = [colors.HexColor("#4B8BBE")]
    max_impact = max(abs(explanation_df_sorted["Contribution"].min()), abs(explanation_df_sorted["Contribution"].max()))

    def make_risk_bar(value, width=50):
        if abs(value) < 0.01:
            return " "
        symbol = "█" * int((abs(value) / max_impact) * width)
        return f"{symbol} {'+' if value > 0 else '-'}"

    for _, row in explanation_df_sorted.iterrows():
        impact = row["Contribution"]
        bar = make_risk_bar(impact, width=15)

        if impact < 0:
            bg_color = colors.HexColor("#ffe5e5")
        elif impact > 0:
            bg_color = colors.HexColor("#e5ffe5")
        else:
            bg_color = colors.beige

        if mode == "Model View":
            row_data = [
                Paragraph(str(row["Feature"]), wrap_style),
                Paragraph(str(row["Input Value"]), wrap_style),
                Paragraph(f"{row['Score Weight']:.4f}", wrap_style),
                Paragraph(f"{impact:+.2f} {bar}", wrap_style)
            ]
        else:
            row_data = [
                Paragraph(str(row["Feature"]), wrap_style),
                Paragraph(str(row["Input Value"]), wrap_style),
                Paragraph(f"{impact:+.2f} {bar}", wrap_style)
            ]
        table_data.append(row_data)
        row_colors.append(bg_color)

    table = Table(table_data, repeatRows=1, colWidths=col_widths)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4B8BBE")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        ('ALIGN', (2, 1), (2, -1), 'LEFT') if mode == "Model View" else ('ALIGN', (1, 1), (1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
    ])

    for i, bg in enumerate(row_colors[1:], start=1):
        style.add('BACKGROUND', (0, i), (-1, i), bg)

    table.setStyle(style)
    elements.append(table)

    elements.append(Spacer(1, 8))
    legend_text = (
    "<font backColor='#ffcccc'>&nbsp;&nbsp;&nbsp;&nbsp;</font> = Increased mortality risk  "
    "<font backColor='#ccffcc'>&nbsp;&nbsp;&nbsp;&nbsp;</font> = Reduced mortality risk"
    )
    
    legend = Paragraph(
        legend_text,
        ParagraphStyle("Legend", fontSize=8, textColor=colors.black)
    )
    elements.append(legend)

    # Footnotes
    elements.append(Spacer(1, 14))
    footnotes = [
    "* Risk Impact reflects the model’s contribution — negative values indicate increased mortality risk."
    ]
    
    if mode == "Model View":
        footnotes.append("* Risk Weight reflects model sensitivity — higher absolute values indicate stronger influence.")
    
    footnotes.append("This is a computer-generated document; no signature is required.")

    for note in footnotes:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(note, ParagraphStyle("Footer", fontSize=8, leading=10, alignment=0)))

    doc.build(elements)
    buffer.seek(0)
    return buffer









def explain_score_model(score_weights, input_df, reference_order=None, threshold=0.05, original_input_df=None, original_text_map=None, mode="Model View"):
    input_array = input_df.values.flatten()
    contributions = input_array * score_weights
    abs_contributions = np.abs(contributions)
    feature_names = reference_order if reference_order else input_df.columns.tolist()

    if original_input_df is not None:
        original_input_df = original_input_df[feature_names]
        input_display = original_input_df.values.flatten()
    else:
        input_display = input_array

    input_display_readable = []
    for feat, val in zip(feature_names, input_display):
        if original_text_map and feat in original_text_map:
            input_display_readable.append(original_text_map[feat])
        else:
            input_display_readable.append(val)

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Input Value": input_display_readable,
        "Score Weight": score_weights,
        "Contribution": contributions,
        "Abs Contribution": abs_contributions
    }).sort_values("Abs Contribution", ascending=False)

    explanation_df["Risk Impact"] = explanation_df["Contribution"].apply(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}")
    explanation_df["Risk Weight"] = explanation_df["Score Weight"]

    display_df = explanation_df[["Feature", "Input Value", "Risk Weight", "Risk Impact"]]

    if mode == "Model View":
        st.write("### 🔢 Feature Contributions (Model View)")
        st.data_editor(
            display_df,
            use_container_width=True,
            disabled=True,
            column_config={
                "Risk Weight": st.column_config.NumberColumn("Risk Weight", format="%.4f", width="medium")
            }
        )
    else:
        st.write("### 🧠 Key Clinical Contributors (Simplified View)")
        st.table(display_df[["Feature", "Input Value", "Risk Impact"]])

    st.caption("🔎 Negative Risk Impact indicates increased mortality risk. Positive indicates reduced risk.")

    st.write("### 🧾 Clinical Action Recommendations")
    for _, row in explanation_df.iterrows():
        if float(row["Contribution"]) < -threshold:
            feat = row["Feature"]
            val = row["Input Value"]
            message = None

            if feat in REFERENCE_RANGES:
                low, high = REFERENCE_RANGES[feat]
                try:
                    val_float = float(val)
                    if val_float < low:
                        status = "low"
                    elif val_float > high:
                        status = "high"
                    else:
                        continue
                    action = NUMERICAL_ACTION_HINTS.get(feat, "May indicate clinical abnormality — suggest further diagnostic evaluation based on context.")
                    message = f"- **{feat}** is **{status}** ({val}). {action}"
                except:
                    pass
            elif feat in CATEGORICAL_HINTS:
                message = f"- **{feat}** contributed significantly → {CATEGORICAL_HINTS[feat]}"

            if message:
                st.markdown(message)

    # Pass mode to PDF
    pdf_buf = generate_pdf_table(explanation_df, mode=mode)
    st.download_button(
        label="⬇️ Download Explanation Summary (PDF)",
        data=pdf_buf,
        file_name="score_model_explanation.pdf",
        mime="application/pdf"
    )

    # SHAP-style plot
    st.write("### 📉 Feature Impact Visualization")
    sorted_df = explanation_df.sort_values("Contribution")
    fig, ax = plt.subplots(figsize=(10, len(sorted_df) * 0.25))
    colors = ['green' if val > 0 else 'red' for val in sorted_df["Contribution"]]
    ax.barh(sorted_df["Feature"], sorted_df["Contribution"], color=colors)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel("Contribution to Risk Score")
    ax.set_title("All Feature Contributions (Score-Based)")
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="⬇️ Download Explanation Plot",
        data=buf,
        file_name="score_feature_contributions.png",
        mime="image/png"
    )

    return display_df



