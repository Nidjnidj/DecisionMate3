import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import io
from datetime import datetime
import networkx as nx
from fpdf import FPDF
import zipfile

# === Translation Dictionary ===
translations = {
    "English": {
        "login_title": "🔐 Login",
        "username": "Username",
        "login_button": "Login",
        "login_warning": "Please enter your name to access the application.",
        "theme": "🌓 Theme",
        "language": "🌐 Language",
        "modules": "📂 Modules",
        "title": "📊 DecisionMate3 - Unified Smart Decision App",
        "select_module": "Select Module",
    },
    "Azerbaijani": {
        "login_title": "🔐 Giriş",
        "username": "İstifadəçi adı",
        "login_button": "Daxil ol",
        "login_warning": "Zəhmət olmasa daxil olmaq üçün adınızı yazın.",
        "theme": "🌓 Mövzu",
        "language": "🌐 Dil",
        "modules": "📂 Modullar",
        "title": "📊 DecisionMate3 - Ağıllı Qərar Tətbiqi",
        "select_module": "Modul seçin",
    },
    "Russian": {
        "login_title": "🔐 Вход",
        "username": "Имя пользователя",
        "login_button": "Войти",
        "login_warning": "Пожалуйста, введите имя для доступа к приложению.",
        "theme": "🌓 Тема",
        "language": "🌐 Язык",
        "modules": "📂 Модули",
        "title": "📊 DecisionMate3 - Приложение для умных решений",
        "select_module": "Выберите модуль",
    }
}

# === Set Language Code and Key Only Once ===
if "selected_language" not in st.session_state:
    st.session_state.selected_language = "English"

language = st.sidebar.radio("🌐 Language", ["English", "Azerbaijani", "Russian"],
                            index=["English", "Azerbaijani", "Russian"].index(st.session_state.selected_language),
                            key="language_select")
st.session_state.selected_language = language
T = translations[language]

# === Streamlit Config ===
st.set_page_config(
    page_title="DecisionMate3",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# === Sidebar Settings ===
st.sidebar.image("nijat_logo.png", width=200)
st.sidebar.subheader(T["theme"])
theme_mode = st.sidebar.radio("", ["Light", "Dark"], key="theme_mode")
st.sidebar.subheader(T["modules"])

# === Simple Login ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    with st.sidebar.expander(T["login_title"]):
        username = st.text_input(T["username"])
        login_button = st.button(T["login_button"])
        if username and login_button:
            st.session_state.logged_in = True
            st.session_state.username = username
        else:
            st.warning(T["login_warning"])
            st.stop()

# === Page Title ===
st.title(T["title"])

# === Module Selection ===
selected_group = st.sidebar.radio("Module Category", [
    "🧠 Personal Decisions",
    "📊 Business & Financial Tools",
    "📅 Planning Tools",
    "🚨 Risk Management"
])

if selected_group == "🧠 Personal Decisions":
    selected_module = st.sidebar.radio(T["select_module"], [
        "Life & Career Decisions",
        "Pros & Cons Evaluator",
        "SWOT Analysis"
    ])
elif selected_group == "📊 Business & Financial Tools":
    selected_module = st.sidebar.radio(T["select_module"], [
        "Rent vs Buy",
        "CAPEX/OPEX/NPV Calculator",
        "IRR Analysis",
        "Tornado Chart"
    ])
elif selected_group == "📅 Planning Tools":
    selected_module = st.sidebar.radio(T["select_module"], [
        "Critical Path Planner",
        "S-Curve & Schedule Planner"
    ])
elif selected_group == "🚨 Risk Management":
    selected_module = st.sidebar.radio(T["select_module"], [
        "Risk Management"
    ])

# === Shared State for Business Tools ===
if "capex" not in st.session_state:
    st.session_state.capex = 100000
if "opex" not in st.session_state:
    st.session_state.opex = 10000
if "revenue" not in st.session_state:
    st.session_state.revenue = 20000
if "discount_rate" not in st.session_state:
    st.session_state.discount_rate = 10
if "years" not in st.session_state:
    st.session_state.years = 10

# === CAPEX / OPEX / NPV Calculator ===
# Enhanced with professional layout and breakdowns
def capex_opex():
    st.header("💼 Capital Investment Evaluation")
    st.markdown("""
Use this tool to evaluate the Net Present Value (NPV) of an investment based on capital and operational costs,
projected revenue, discount rate, and project duration.
""")
    st.session_state.capex = st.number_input("CAPEX ($)", value=st.session_state.capex, key="capex_input")
    st.session_state.opex = st.number_input("Annual OPEX ($)", value=st.session_state.opex, key="opex_input")
    st.session_state.revenue = st.number_input("Annual Revenue ($)", value=st.session_state.revenue, key="revenue_input")
    st.session_state.discount_rate = st.slider("Discount Rate (%)", 0, 20, st.session_state.discount_rate, key="discount_input")
    st.session_state.years = st.slider("Project Life (Years)", 1, 30, st.session_state.years, key="years_input")

    cash_flows = [(st.session_state.revenue - st.session_state.opex)] * st.session_state.years
    npv = npf.npv(st.session_state.discount_rate/100, cash_flows) - st.session_state.capex

    st.metric(label="📊 Net Present Value (NPV)", value=f"${npv:,.2f}")

# === Tornado Chart ===
# Enhanced for sensitivity analysis insights
def tornado_chart():
    st.header("🌪️ Sensitivity Analysis: Tornado Diagram")
    st.markdown("""
This chart shows how variations in key inputs affect project outcomes. Use it to identify
which parameters most significantly impact your decision.
""")
    st.write("Automatically generated sensitivity chart from CAPEX data")
    variations = [
        {"Variable": "CAPEX", "Low": st.session_state.capex * 0.8, "Base": st.session_state.capex, "High": st.session_state.capex * 1.2},
        {"Variable": "OPEX", "Low": st.session_state.opex * 0.8, "Base": st.session_state.opex, "High": st.session_state.opex * 1.2},
        {"Variable": "Revenue", "Low": st.session_state.revenue * 0.8, "Base": st.session_state.revenue, "High": st.session_state.revenue * 1.2},
        {"Variable": "Years", "Low": st.session_state.years - 2, "Base": st.session_state.years, "High": st.session_state.years + 2},
        {"Variable": "Discount Rate", "Low": st.session_state.discount_rate - 2, "Base": st.session_state.discount_rate, "High": st.session_state.discount_rate + 2},
    ]

    df = pd.DataFrame(variations)
    df["Impact"] = df["High"] - df["Low"]
    df.sort_values("Impact", ascending=True, inplace=True)

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            y=[row["Variable"]],
            x=[row["High"] - row["Base"]],
            orientation='h',
            name="High",
            marker=dict(color="green")
        ))
        fig.add_trace(go.Bar(
            y=[row["Variable"]],
            x=[row["Low"] - row["Base"]],
            orientation='h',
            name="Low",
            marker=dict(color="red")
        ))
    fig.update_layout(barmode='overlay')
    st.plotly_chart(fig, key="tornado_chart")

# === IRR Analysis ===
# Enhanced with interpretation and validation
def irr_analysis():
    st.header("📈 Internal Rate of Return (IRR) Analysis")
    st.markdown("""
This tool calculates the IRR to determine the profitability of an investment. IRR is the discount rate
at which the NPV equals zero.
""")
    st.write("Uses same CAPEX and years from shared input")
    inflows = [st.session_state.revenue - st.session_state.opex] * st.session_state.years
    irr = npf.irr([-st.session_state.capex] + inflows)
    st.metric(label="📉 Internal Rate of Return (IRR)", value=f"{irr*100:.2f}%")
    if irr*100 < st.session_state.discount_rate:
        st.warning("IRR is below your discount rate – the project may not be profitable.")
    else:
        st.success("IRR exceeds your discount rate – the project is likely profitable.")

# === Risk Management ===
def risk_management():
    st.header("🚨 Risk Management Matrix")
    st.markdown("""
Use this matrix to assess and visualize risks based on their probability and impact.
Select the severity level of each risk to prioritize mitigations.
""")

    st.subheader("Define Risks")
    risk_data = st.text_area("Enter risks (one per line, format: Risk Description)")
    risks = risk_data.splitlines()

    if risks:
        matrix_data = []
        st.subheader("Rate Each Risk")
        for risk in risks:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(risk)
            with col2:
                probability = st.selectbox(f"Probability - {risk}", ["Low", "Medium", "High"], key=f"prob_{risk}")
                impact = st.selectbox(f"Impact - {risk}", ["Low", "Medium", "High"], key=f"impact_{risk}")
            mitigation = st.text_input(f"Mitigation Plan - {risk}", key=f"mit_{risk}")
            owner = st.text_input(f"Owner - {risk}", key=f"owner_{risk}")
            category = st.selectbox(f"Category - {risk}", ["Strategic", "Operational", "Financial", "Compliance"], key=f"cat_{risk}")
            review_date = st.date_input(f"Review Date - {risk}", key=f"date_{risk}")
            matrix_data.append({"Risk": risk, "Probability": probability, "Impact": impact, "Mitigation": mitigation, "Owner": owner, "Category": category, "Review Date": review_date})

        df = pd.DataFrame(matrix_data)
        st.subheader("📋 Risk Register")
        st.dataframe(df[["Risk", "Probability", "Impact", "Mitigation", "Owner", "Category", "Review Date"]])

        def score_level(value):
            return {"Low": 1, "Medium": 2, "High": 3}.get(value, 1)

        df["Score"] = df.apply(lambda row: score_level(row["Probability"]) * score_level(row["Impact"]), axis=1)
        color_map = {3: "🟡 Low", 4: "🟡 Low", 6: "🟠 Medium", 9: "🔴 High"}
        df["Severity"] = df["Score"].apply(lambda x: color_map.get(x, "🟢 Minimal"))

        selected_categories = st.multiselect("Filter by Risk Category", options=df["Category"].unique())
        if selected_categories:
            df = df[df["Category"].isin(selected_categories)]

        severity_order = ["🔴 High", "🟠 Medium", "🟡 Low", "🟢 Minimal"]
        df = df.sort_values(by="Severity", key=lambda x: x.map(lambda s: severity_order.index(s)), ascending=True)

        grouped = df.groupby("Severity")
        for severity, group in grouped:
            st.subheader(f"{severity} Risks")
            st.dataframe(group[["Risk", "Probability", "Impact", "Score", "Severity", "Mitigation", "Owner", "Category", "Review Date"]])

        st.subheader("🔥 Prioritized Risks")
        st.dataframe(df[["Risk", "Probability", "Impact", "Score", "Severity", "Mitigation", "Owner", "Category", "Review Date"]])

        fig = px.scatter(df, x="Probability", y="Impact", size="Score", color="Score", hover_name="Risk",
                         category_orders={"Probability": ["Low", "Medium", "High"], "Impact": ["Low", "Medium", "High"]},
                         title="Risk Matrix Visualization")
        fig.update_layout(xaxis_title="Probability", yaxis_title="Impact")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📤 Export Risk Register")
        export_excel(df, filename="risk_register.xlsx")





# === Module Dispatcher ===
if selected_module == "CAPEX/OPEX/NPV Calculator":
    capex_opex()
elif selected_module == "Tornado Chart":
    tornado_chart()
elif selected_module == "IRR Analysis":
    irr_analysis()







# === Helper: Export to Excel ===
def export_excel(df, filename="export.xlsx"):
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False)
    towrite.seek(0)
    st.download_button(
        label="📥 Download Excel",
        data=towrite,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# === Helper: Export to PDF ===
def export_pdf(text, filename="export.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split('\n'):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf_output = io.BytesIO(pdf.output(dest='S').encode('latin-1'))
    st.download_button(
        label="📥 Download PDF",
        data=pdf_output,
        file_name=filename,
        mime="application/pdf"
    )


# === Life & Career Decision Scoring ===
def life_and_career():
    st.session_state.personal_options = []
    st.session_state.personal_criteria = []
    st.session_state.personal_weights = []

    st.header("🎯 Life & Career Decision Scoring")
    options = st.text_area("Enter your options (one per line)").splitlines()
    st.session_state.personal_options = options
    criteria = st.text_area("Enter criteria (one per line)").splitlines()
    st.session_state.personal_criteria = criteria

    weights_enabled = st.checkbox("Enable weighted criteria scoring", key="weighting_enable")
    pairwise_enabled = st.checkbox("Enable AHP pairwise comparisons", key="ahp_enable")
    weights = []
    if criteria and pairwise_enabled:
        st.subheader("AHP Pairwise Comparison Matrix")
        matrix = np.ones((len(criteria), len(criteria)))
        for i in range(len(criteria)):
            for j in range(i + 1, len(criteria)):
                val = st.slider(f"How much more important is '{criteria[i]}' than '{criteria[j]}'?", 1, 9, 1, key=f"ahp_{i}_{j}")
                matrix[i, j] = val
                matrix[j, i] = 1 / val
        eigvals, eigvecs = np.linalg.eig(matrix)
        max_index = np.argmax(eigvals.real)
        weights = eigvecs[:, max_index].real
        weights = weights / weights.sum()
        st.session_state.personal_weights = weights.tolist()

        st.subheader("AHP Matrix Heatmap")
        fig_heat = px.imshow(matrix, text_auto=True, x=criteria, y=criteria, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_heat, use_container_width=True, key="ahp_heatmap")

        lambda_max = np.mean(np.dot(matrix, weights) / weights)
        ci = (lambda_max - len(criteria)) / (len(criteria) - 1)
        ri_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_dict.get(len(criteria), 1.49)
        cr = ci / ri if ri else 0

        st.caption(f"Consistency Ratio (CR): {cr:.3f} ({'Acceptable' if cr < 0.1 else 'Too high – revise pairwise ratings'})")

        st.subheader("📋 Ranking Order")
        ranking_df = pd.DataFrame({"Criterion": criteria, "Weight": weights})
        ranking_df = ranking_df.sort_values(by="Weight", ascending=False).reset_index(drop=True)
        st.dataframe(ranking_df.style.format({"Weight": "{:.3f}"}))

    elif criteria:
        if weights_enabled:
            weights = []
            for criterion in criteria:
                weight = st.slider(f"Weight for {criterion}", 1, 10, 5, key=f"weight_{criterion}")
                weights.append(weight)
            weights = np.array(weights)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
            st.session_state.personal_weights = weights.tolist()
        else:
            weights = np.ones(len(criteria)) / len(criteria)
            st.session_state.personal_weights = weights.tolist()

    if options and criteria:
        scores = {}
        for option in options:
            scores[option] = []
            for criterion in criteria:
                score = st.slider(f"{option} - {criterion}", 0, 10, 5, key=f"{option}_{criterion}")
                scores[option].append(score)

        df = pd.DataFrame(scores, index=criteria)
        st.subheader("Scoring Table")
        st.dataframe(df)

        st.subheader("Weighted Scores")
        weights = np.array(st.session_state.personal_weights) if "personal_weights" in st.session_state else weights
        weighted_scores = df.mul(weights, axis=0)
        averages = weighted_scores.sum(axis=0).sort_values(ascending=False)
        st.bar_chart(averages)

        st.success(f"Best option: {averages.idxmax()} with weighted score {averages.max():.2f}")

        st.subheader("Radar Chart")
        radar_df = weighted_scores.T.reset_index().rename(columns={"index": "Option"})
        radar_df = pd.melt(radar_df, id_vars=["Option"], var_name="Criterion", value_name="Score")
        fig = px.line_polar(radar_df, r="Score", theta="Criterion", color="Option", line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, key="radar_chart")



# === Pros & Cons Evaluator ===
def pros_cons():
    # Reuse personal_options if available
    prefill = st.session_state.personal_options if "personal_options" in st.session_state else []
    st.header("⚖️ Pros & Cons Evaluator")
    option = st.selectbox("Decision Topic", prefill) if prefill else st.text_input("Decision Topic")
    pros = st.text_area("Pros (one per line):").splitlines()
    cons = st.text_area("Cons (one per line):").splitlines()

    st.subheader("Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("✅ Pros")
        for p in pros:
            st.write(f"- {p}")
    with col2:
        st.write("❌ Cons")
        for c in cons:
            st.write(f"- {c}")

    summary = f"Pros:\n" + "\n".join(pros) + "\n\nCons:\n" + "\n".join(cons)
    export_pdf(summary, "pros_cons_summary.pdf")

# === SWOT Analysis ===
def swot_analysis():
    # Optionally show criteria to help guide SWOT
    if "personal_criteria" in st.session_state:
        st.info("Your decision criteria: " + ", ".join(st.session_state.personal_criteria))
    st.header("📈 SWOT Analysis")
    strengths = st.text_area("Strengths")
    weaknesses = st.text_area("Weaknesses")
    opportunities = st.text_area("Opportunities")
    threats = st.text_area("Threats")
    
    st.markdown(f"""
    ### 📊 SWOT Summary
    - **Strengths**: {strengths.replace(chr(10), '<br>')}
    - **Weaknesses**: {weaknesses.replace(chr(10), '<br>')}
    - **Opportunities**: {opportunities.replace(chr(10), '<br>')}
    - **Threats**: {threats.replace(chr(10), '<br>')}
    """, unsafe_allow_html=True)

    text = f"SWOT Analysis\n\nStrengths:\n{strengths}\n\nWeaknesses:\n{weaknesses}\n\nOpportunities:\n{opportunities}\n\nThreats:\n{threats}"
    export_pdf(text, "swot_summary.pdf")

# === Rent vs Buy ===
def rent_vs_buy():
    st.header("🏠 Rent vs Buy Calculator")
    rent = st.number_input("Monthly Rent ($)", min_value=0.0)
    home_price = st.number_input("Home Purchase Price ($)", min_value=0.0)
    interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0)
    years = st.slider("Loan Term (years)", 1, 40, 30)

    if rent and home_price and interest_rate:
        monthly_rate = interest_rate / 100 / 12
        months = years * 12
        mortgage = npf.pmt(monthly_rate, months, -home_price)

        st.write(f"🏡 Estimated Mortgage Payment: ${mortgage:,.2f}/month")
        st.write(f"📉 Rent over {years} years: ${rent*12*years:,.2f}")
        result = f"Mortgage: ${mortgage:,.2f}/month\nTotal Rent: ${rent*12*years:,.2f}"
        export_pdf(result, "rent_vs_buy.pdf")



# === S-Curve & Schedule Planner ===
def s_curve_schedule():
    st.header("📈 S-Curve & Schedule Planner")
    st.caption("📍 Tasks and milestones are now integrated with critical path logic")
    st.markdown("""
Use this tool to plan and visualize your project schedule over time.
Enter key milestones or task data to generate an S-curve.
""")

    schedule_data = st.text_area("Enter schedule data (Format: Task, Start Date [YYYY-MM-DD], End Date [YYYY-MM-DD])",
    """Design,2025-01-01,2025-02-01
Procurement,2025-02-02,2025-03-15
Construction,2025-03-16,2025-06-30""")

    if schedule_data:
        df = pd.read_csv(io.StringIO(schedule_data), header=None, names=["Task", "Start", "End"])
        df["Start"] = pd.to_datetime(df["Start"])
        df["End"] = pd.to_datetime(df["End"])
        df["Duration"] = (df["End"] - df["Start"]).dt.days
        df["Cumulative"] = df["Duration"].cumsum()

        # Add milestone flag (e.g., tasks < 5 days are milestones)
        df["Milestone"] = df["Duration"] <= 5
        df["Midpoint"] = df["Start"] + (df["End"] - df["Start"]) / 2

        st.subheader("Schedule Table")
        st.dataframe(df)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["End"],
            y=df["Cumulative"],
            text=df["Task"],
            mode='lines+markers',
            name='Cumulative Progress',
            marker=dict(symbol="circle", size=8, color='blue')
        ))

        milestone_df = df[df["Milestone"] == True]
        fig.add_trace(go.Scatter(
            x=milestone_df["Midpoint"],
            y=milestone_df["Cumulative"],
            mode='markers+text',
            text=milestone_df["Task"],
            name='Milestones',
            marker=dict(symbol="star", size=12, color='orange'),
            textposition='top center'
        ))
        fig.update_layout(title="S-Curve & Milestone Chart", xaxis_title="Date", yaxis_title="Cumulative Days", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        export_excel(df, "schedule_scurve.xlsx")



# === Critical Path Planner ===
def critical_path():
    st.caption("🔁 Integrated with schedule tasks for consistent planning")
    st.header("📅 Critical Path Method")
    st.write("Enter tasks with durations and dependencies.")
    data = st.text_area("Format: Task,Duration,Dependencies\nExample:\nA,3,\nB,2,A\nC,4,A\nD,2,B,C")

    if data:
        df = pd.read_csv(io.StringIO(data), header=None)
        df.columns = ["Task", "Duration", "Dependencies"]

        G = nx.DiGraph()

        for _, row in df.iterrows():
            G.add_node(row["Task"], duration=int(row["Duration"]))
            if pd.notna(row["Dependencies"]):
                for dep in row["Dependencies"].split(','):
                    G.add_edge(dep.strip(), row["Task"])

        critical_path = nx.dag_longest_path(G, weight='duration')
        st.write("📌 Critical Path:", " ➝ ".join(critical_path))
        export_pdf(" ➝ ".join(critical_path), "critical_path.pdf")


# === Module Dispatcher ===
if selected_module == "Life & Career Decisions":
    life_and_career()
elif selected_module == "Pros & Cons Evaluator":
    pros_cons()
elif selected_module == "SWOT Analysis":
    swot_analysis()
elif selected_module == "Rent vs Buy":
    rent_vs_buy()
elif selected_module == "CAPEX/OPEX/NPV Calculator":
    capex_opex()
elif selected_module == "IRR Analysis":
    irr_analysis()
elif selected_module == "Tornado Chart":
    tornado_chart()
elif selected_module == "Critical Path Planner":
    critical_path()
elif selected_module == "S-Curve & Schedule Planner":
    s_curve_schedule()
    critical_path()
elif selected_module == "Risk Management":
    risk_management()
