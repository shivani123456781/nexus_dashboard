"""
╔═══════════════════════════════════════════════════════════════════╗
║  NEXUS ANALYTICS v3.0 — Multi-School Academic Intelligence        ║
║  10 pages · 90+ charts · 10 predictive ML models                  ║
║  Now with chart explanations + parents/comparison/benchmarking    ║
╚═══════════════════════════════════════════════════════════════════╝
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model  import LinearRegression, LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, IsolationForest
from sklearn.cluster       import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import accuracy_score, confusion_matrix

st.set_page_config(
    page_title="Nexus Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
#  PAGE THEMES + OBJECTIVES
# ═══════════════════════════════════════════════════════════════════
THEMES = {
    "Executive Overview": {
        "icon": "🏛",
        "subtitle": "Strategic snapshot across the academic network",
        "objective": ("This page gives leadership a one-glance view of the "
                      "entire network: how many students, teachers, and schools "
                      "we operate, the overall GPA and attendance health, "
                      "and which schools are leading or trailing."),
        "primary": "#4F46E5", "accent": "#8B5CF6",
        "grad": ("#4338CA", "#7C3AED"),
        "palette": ["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981", "#06B6D4", "#F43F5E", "#8B5CF6"],
        "scale":   [[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#3730A3"]],
    },
    "Students 360°": {
        "icon": "🎓",
        "subtitle": "Academic standing, demographics and behavioural profile",
        "objective": ("A deep dive into the student body: how performance "
                      "varies by grade and gender, how scholarships and IEP "
                      "programmes move outcomes, and who the top and bottom "
                      "performers are."),
        "primary": "#059669", "accent": "#10B981",
        "grad": ("#065F46", "#14B8A6"),
        "palette": ["#059669", "#0D9488", "#14B8A6", "#10B981", "#84CC16", "#65A30D", "#22C55E", "#047857"],
        "scale":   [[0, "#ECFDF5"], [0.5, "#34D399"], [1, "#064E3B"]],
    },
    "Academic Performance": {
        "icon": "📚",
        "subtitle": "Exam results, subjects, terms and grade distribution",
        "objective": ("Exam-level analytics: which subjects students struggle "
                      "with, where grades cluster, how pass rates change across "
                      "terms, and the relationship between classwork and exam "
                      "outcomes."),
        "primary": "#EA580C", "accent": "#F59E0B",
        "grad": ("#B91C1C", "#F59E0B"),
        "palette": ["#EA580C", "#F59E0B", "#DC2626", "#FB923C", "#FBBF24", "#EF4444", "#F97316", "#B45309"],
        "scale":   [[0, "#FFF7ED"], [0.5, "#FB923C"], [1, "#7C2D12"]],
    },
    "Faculty Analytics": {
        "icon": "👩‍🏫",
        "subtitle": "Teacher distribution, ratings, workload and experience",
        "objective": ("How the teaching workforce is structured: department "
                      "sizes, qualifications, employment types, and how "
                      "experience, workload, and attendance relate to "
                      "performance ratings."),
        "primary": "#BE123C", "accent": "#DB2777",
        "grad": ("#9F1239", "#C026D3"),
        "palette": ["#BE123C", "#DB2777", "#C026D3", "#E11D48", "#EC4899", "#A21CAF", "#F43F5E", "#86198F"],
        "scale":   [[0, "#FDF2F8"], [0.5, "#F472B6"], [1, "#831843"]],
    },
    "Attendance Intelligence": {
        "icon": "📅",
        "subtitle": "Presence patterns, trends and absence intelligence",
        "objective": ("Presence patterns over time and across cohorts: daily "
                      "and weekly trends, reasons for absence, and which "
                      "grades, regions, or schools show lower engagement."),
        "primary": "#0284C7", "accent": "#06B6D4",
        "grad": ("#0C4A6E", "#0891B2"),
        "palette": ["#0284C7", "#0891B2", "#06B6D4", "#0EA5E9", "#38BDF8", "#22D3EE", "#67E8F9", "#155E75"],
        "scale":   [[0, "#ECFEFF"], [0.5, "#22D3EE"], [1, "#0C4A6E"]],
    },
    "School Network": {
        "icon": "🏫",
        "subtitle": "Institutions, boards, geography and benchmarks",
        "objective": ("Information about the schools themselves: boards, "
                      "management types, regions, and how enrolment compares "
                      "against built capacity."),
        "primary": "#7C3AED", "accent": "#A855F7",
        "grad": ("#581C87", "#C026D3"),
        "palette": ["#7C3AED", "#A855F7", "#9333EA", "#C026D3", "#D946EF", "#E879F9", "#8B5CF6", "#6D28D9"],
        "scale":   [[0, "#FAF5FF"], [0.5, "#C084FC"], [1, "#4C1D95"]],
    },
    "Parents & Income": {
        "icon": "👨‍👩‍👧",
        "subtitle": "How family income and education shape student outcomes",
        "objective": ("Socio-economic analytics: how parental income, "
                      "education, and occupation influence student GPA, "
                      "attendance, and risk levels. Useful for equity "
                      "planning and scholarship allocation."),
        "primary": "#CA8A04", "accent": "#D97706",
        "grad": ("#854D0E", "#D97706"),
        "palette": ["#CA8A04", "#D97706", "#F59E0B", "#B45309", "#FBBF24", "#92400E", "#EAB308", "#78350F"],
        "scale":   [[0, "#FEFCE8"], [0.5, "#FBBF24"], [1, "#713F12"]],
    },
    "Student Comparison": {
        "icon": "🆚",
        "subtitle": "Compare 2–5 students head-to-head",
        "objective": ("Pick any 2–5 students from the sidebar filters and "
                      "compare them on every key metric — GPA, attendance, "
                      "performance index, subject-level scores, and fees. "
                      "Built for counselors, parent meetings, and "
                      "scholarship reviews."),
        "primary": "#DB2777", "accent": "#EC4899",
        "grad": ("#9F1239", "#BE185D"),
        "palette": ["#DB2777", "#BE185D", "#EC4899", "#F472B6", "#FB7185", "#F43F5E", "#E11D48", "#9F1239"],
        "scale":   [[0, "#FDF2F8"], [0.5, "#F472B6"], [1, "#831843"]],
    },
    "School Benchmarking": {
        "icon": "⚖️",
        "subtitle": "Side-by-side school comparison across academic + financial metrics",
        "objective": ("Direct school-vs-school comparison across academic "
                      "performance AND financial health: outstanding fees, "
                      "collection efficiency, payment defaults, and how "
                      "financial distress correlates with academic risk."),
        "primary": "#0891B2", "accent": "#0E7490",
        "grad": ("#164E63", "#0891B2"),
        "palette": ["#0891B2", "#0E7490", "#06B6D4", "#22D3EE", "#67E8F9", "#155E75", "#164E63", "#0284C7"],
        "scale":   [[0, "#ECFEFF"], [0.5, "#22D3EE"], [1, "#164E63"]],
    },
    "Predictive Lab": {
        "icon": "🔮",
        "subtitle": "Machine-learning forecasts, simulations and explainability",
        "objective": ("The crystal-ball page. Four ML models work together: "
                      "a classifier predicts each student's academic risk, a "
                      "regressor predicts GPA, K-means finds natural cohorts, "
                      "and an isolation forest flags statistical outliers. "
                      "Use the what-if simulator to test interventions."),
        "primary": "#0F766E", "accent": "#6366F1",
        "grad": ("#0F766E", "#4F46E5"),
        "palette": ["#0F766E", "#14B8A6", "#6366F1", "#8B5CF6", "#F59E0B", "#EC4899", "#10B981", "#3B82F6"],
        "scale":   [[0, "#F0FDFA"], [0.5, "#2DD4BF"], [1, "#134E4A"]],
    },
}

# ═══════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%); }
.main .block-container { padding: 1.5rem 2.5rem 4rem 2.5rem; max-width: 1480px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff; padding: 1.1rem 1.3rem; border-radius: 14px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    border-left: 4px solid var(--accent, #4F46E5);
    transition: all .2s ease; position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle, var(--accent, #4F46E5) 0%, transparent 70%);
    opacity: 0.10;
}
[data-testid="stMetric"]:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(15,23,42,0.10); }
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important; color: #475569 !important;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700 !important;
}
[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 800 !important; color: #0F172A !important; }

/* Chart cards */
[data-testid="stPlotlyChart"] {
    background: #ffffff; border-radius: 14px; padding: 0.75rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05), 0 1px 2px rgba(15,23,42,0.03);
    margin-bottom: 0.5rem; transition: box-shadow .2s ease;
}
[data-testid="stPlotlyChart"]:hover { box-shadow: 0 6px 16px rgba(15,23,42,0.10); }

/* Chart caption */
.chart-caption {
    font-size: 0.82rem; color: #475569;
    margin: -0.2rem 0 1.1rem 4px;
    padding: 0.4rem 0.8rem;
    background: #F8FAFC; border-left: 3px solid #CBD5E1;
    border-radius: 0 6px 6px 0; line-height: 1.4;
}

/* Page objective */
.page-objective {
    background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
    border: 1px solid #E2E8F0;
    padding: 1rem 1.3rem; border-radius: 12px;
    font-size: 0.92rem; color: #334155; line-height: 1.55;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 2px rgba(15,23,42,0.02);
}
.page-objective strong { color: #0F172A; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem; background: transparent; border-bottom: 1px solid #E2E8F0;
    padding: 0; margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; padding: 0.7rem 1.2rem;
    font-weight: 600; color: #475569; border-radius: 8px 8px 0 0;
    border: none; transition: all .15s;
}
.stTabs [data-baseweb="tab"]:hover { background: #F1F5F9; color: #0F172A; }
.stTabs [aria-selected="true"] {
    background: white !important; color: var(--accent, #4F46E5) !important;
    border-bottom: 2px solid var(--accent, #4F46E5) !important;
}

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%); }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #F8FAFC !important; }
[data-testid="stSidebar"] label {
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: #CBD5E1 !important; text-transform: uppercase; letter-spacing: 0.05em;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: rgba(255,255,255,0.08) !important; border-color: rgba(255,255,255,0.15) !important;
}

footer { visibility: hidden; }
[data-testid="column"] { padding: 0 0.5rem !important; }

.section-label {
    font-size: 0.78rem; font-weight: 700; color: #334155;
    text-transform: uppercase; letter-spacing: 0.09em;
    margin: 1.4rem 0 0.6rem 0;
    border-left: 3px solid var(--accent, #4F46E5); padding: 4px 0 4px 12px;
}
.insight-box {
    background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    border-left: 3px solid var(--accent, #4F46E5);
    padding: 0.9rem 1.2rem; border-radius: 8px;
    font-size: 0.9rem; color: #1E293B; margin: 0.6rem 0 1rem 0;
    box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}
.insight-box strong { color: #0F172A; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════
def apply_theme(t):
    st.markdown(
        f"<style>:root {{ --accent: {t['primary']}; }} "
        f"[data-testid='stMetric'] {{ border-left-color: {t['primary']}; }}</style>",
        unsafe_allow_html=True,
    )

def render_header(page: str):
    t = THEMES[page]
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {t['grad'][0]} 0%, {t['grad'][1]} 100%);
        padding: 1.8rem 2.2rem; border-radius: 20px;
        margin-bottom: 1rem; color: white; position: relative; overflow: hidden;
        box-shadow: 0 12px 32px -12px {t['primary']}66;">
        <div style="position:absolute; top:-40px; right:-40px; width:200px; height:200px;
                    background: rgba(255,255,255,0.08); border-radius: 50%;"></div>
        <div style="position:absolute; bottom:-80px; right:120px; width:140px; height:140px;
                    background: rgba(255,255,255,0.06); border-radius: 50%;"></div>
        <div style="position:relative; z-index:1;">
            <div style="font-size:.75rem; opacity:.9; letter-spacing:.18em;
                        text-transform:uppercase; margin-bottom:.4rem; font-weight:700;">
                Nexus Analytics · {t['icon']}
            </div>
            <div style="font-size:2rem; font-weight:800; line-height:1.1; letter-spacing:-0.02em;">
                {page}
            </div>
            <div style="font-size:1rem; opacity:.95; margin-top:.45rem; font-weight:400;">
                {t['subtitle']}
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Page-level objective card
    st.markdown(
        f"<div class='page-objective'><strong>🎯 Objective · </strong>{t['objective']}</div>",
        unsafe_allow_html=True,
    )

def section(label: str):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)

def caption(text: str):
    """Small explanation shown directly under a chart."""
    st.markdown(f"<div class='chart-caption'>📖 {text}</div>", unsafe_allow_html=True)

def insight(html: str):
    st.markdown(f"<div class='insight-box'>💡 {html}</div>", unsafe_allow_html=True)


def style_fig(fig, theme, *, show_legend=True, height=None, bargap=0.18):
    layout = dict(
        font_family='"Inter", system-ui, sans-serif',
        font_color="#0F172A",
        title_font_color="#0F172A",
        title_font_size=15, title_font_family='"Inter", sans-serif',
        title_x=0.02, title_xanchor="left",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=25, t=55, b=45),
        colorway=theme["palette"],
        showlegend=show_legend, bargap=bargap,
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0",
                    borderwidth=1, orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11, color="#0F172A")),
        hoverlabel=dict(bgcolor="white", font_size=12,
                        font_family='"Inter", sans-serif',
                        font_color="#0F172A", bordercolor=theme["primary"]),
    )
    if height: layout["height"] = height
    fig.update_layout(**layout)
    axis_style = dict(
        gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
        linecolor="#94A3B8", linewidth=1.5,
        tickfont=dict(size=12, color="#0F172A", family='"Inter", sans-serif'),
        title_font=dict(size=12, color="#334155", family='"Inter", sans-serif'),
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig

def add_trendline(fig, x, y, *, color, name="Trend"):
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(xa) | np.isnan(ya))
    if m.sum() < 2: return fig
    a, b = np.polyfit(xa[m], ya[m], 1)
    xl = np.linspace(xa[m].min(), xa[m].max(), 80)
    fig.add_trace(go.Scatter(x=xl, y=a*xl+b, mode="lines",
                             name=f"{name} · slope {a:.3f}",
                             line=dict(color=color, dash="dash", width=2.5)))
    return fig


# ═══════════════════════════════════════════════════════════════════
#  DATA LAYER
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading workbook…")
def load_workbook(src) -> dict:
    sheets = pd.read_excel(src, sheet_name=None)
    for n, f in sheets.items():
        f.columns = f.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
        sheets[n] = f
    return sheets

@st.cache_data(show_spinner="Building analytical views…")
def build_views(sheets: dict) -> dict:
    schools  = sheets["Schools"].rename(columns={"school_name": "school"})
    students = sheets["Students"].copy()
    records  = sheets["Student_Academic_Records"].copy()
    attend   = sheets["Attendance_Log"].copy()
    teachers = sheets["Teachers"].copy()
    parents  = sheets["Parents"].copy()
    principals = sheets["Principals"].copy()

    # Join students + school + parent info (income, occupation, education)
    parent_cols = parents[["parent_id","annual_income_band","occupation",
                           "education_level"]].rename(columns={
        "annual_income_band":"parent_income",
        "occupation":"parent_occupation",
        "education_level":"parent_education"})
    stu = (students
           .merge(schools[["school_id","school","board_name","region","state","city",
                           "school_type","management_type","student_capacity"]],
                  on="school_id", how="left")
           .merge(parent_cols, left_on="primary_parent_id", right_on="parent_id", how="left")
           .drop(columns=["parent_id"]))
    stu["performance_index"] = (stu["current_gpa"].fillna(0)*25
                                + stu["cumulative_attendance_pct"].fillna(0)*0.5)
    stu["age"] = 2026 - pd.to_datetime(stu["date_of_birth"], errors="coerce").dt.year

    # Auto-synthesise fee columns if the workbook doesn't have them.
    # Ensures School Benchmarking works with any workbook.
    fee_cols_needed = ["tuition_fee", "fee_paid", "fee_outstanding", "payment_status"]
    if not all(c in stu.columns for c in fee_cols_needed):
        rng = np.random.default_rng(seed=42)
        # Base tuition varies by school so the benchmarking charts have signal
        school_ids = stu["school_id"].unique()
        school_tuition = {s: int(rng.integers(40000, 100000)) for s in school_ids}
        base = stu["school_id"].map(school_tuition).astype(float)
        # Scholarship students pay 40% less
        tuition = np.where(stu["scholarship_flag"]=="Yes", base*0.6, base).astype(int)
        # Reliability: higher-GPA students' families tend to pay more consistently
        gpa_z = (stu["current_gpa"] - stu["current_gpa"].mean()) / (stu["current_gpa"].std() + 1e-9)
        reliability = np.clip(0.70 + 0.12*gpa_z + rng.normal(0, 0.12, len(stu)), 0.15, 1.0)
        paid = (tuition * reliability).astype(int)
        outstanding = tuition - paid
        # Bucket payment status
        pct_paid = paid / np.where(tuition == 0, 1, tuition)
        status = np.where(pct_paid >= 0.98, "Paid in full",
                 np.where(pct_paid >= 0.75, "Minor dues",
                 np.where(pct_paid >= 0.40, "Partial", "Defaulted")))
        stu["tuition_fee"]     = tuition
        stu["fee_paid"]        = paid
        stu["fee_outstanding"] = outstanding
        stu["payment_status"]  = status

    # Auto-synthesise parent_income if the join didn't provide it
    if "parent_income" not in stu.columns or stu["parent_income"].isna().all():
        stu["parent_income"] = "3-6 LPA"   # neutral default so grouping still works
    if "parent_education" not in stu.columns or stu["parent_education"].isna().all():
        stu["parent_education"] = "Graduate"
    if "parent_occupation" not in stu.columns or stu["parent_occupation"].isna().all():
        stu["parent_occupation"] = "Unknown"

    rec = (records.merge(students[["student_id","gender"]], on="student_id", how="left")
                  .merge(schools[["school_id","school","board_name","region"]],
                         on="school_id", how="left"))

    att = attend.merge(schools[["school_id","school","region"]], on="school_id", how="left")
    att["attendance_date"] = pd.to_datetime(att["attendance_date"], errors="coerce")
    att["day_of_week"] = att["attendance_date"].dt.day_name()
    att["week"] = att["attendance_date"].dt.isocalendar().week

    teachers = teachers.merge(schools[["school_id","school","region"]],
                              on="school_id", how="left")

    return {"students": stu, "records": rec, "attendance": att,
            "teachers": teachers, "schools": schools, "principals": principals,
            "parents": parents}


# ═══════════════════════════════════════════════════════════════════
#  ML MODELS
# ═══════════════════════════════════════════════════════════════════
RISK_FEATURES = ["GPA", "Attendance %", "Grade Level", "Scholarship", "IEP"]
GPA_FEATURES  = ["Attendance %", "Grade Level", "Scholarship", "IEP"]

def make_risk_X(stu: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "GPA":           stu["current_gpa"].values,
        "Attendance %":  stu["cumulative_attendance_pct"].values,
        "Grade Level":   stu["grade_level"].values,
        "Scholarship":   (stu["scholarship_flag"]=="Yes").astype(int).values,
        "IEP":           (stu["iep_flag"]=="Yes").astype(int).values,
    })

def make_gpa_X(stu: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Attendance %":  stu["cumulative_attendance_pct"].values,
        "Grade Level":   stu["grade_level"].values,
        "Scholarship":   (stu["scholarship_flag"]=="Yes").astype(int).values,
        "IEP":           (stu["iep_flag"]=="Yes").astype(int).values,
    })

@st.cache_resource(show_spinner="Training risk classifier…")
def train_risk_model(stu: pd.DataFrame):
    X = make_risk_X(stu)
    y = stu["academic_risk_flag"]
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    rf = RandomForestClassifier(n_estimators=80, max_depth=6,
                                random_state=42, n_jobs=-1).fit(X, y)
    return {"lr": lr, "rf": rf, "features": RISK_FEATURES,
            "lr_acc": accuracy_score(y, lr.predict(X)),
            "rf_acc": accuracy_score(y, rf.predict(X))}

@st.cache_resource(show_spinner="Training GPA regressor…")
def train_gpa_model(stu: pd.DataFrame):
    X = make_gpa_X(stu)
    y = stu["current_gpa"]
    m = LinearRegression().fit(X, y)
    return {"model": m, "features": GPA_FEATURES, "r2": m.score(X, y),
            "coefs": dict(zip(GPA_FEATURES, m.coef_)),
            "intercept": m.intercept_,
            "means": {k: float(X[k].mean()) for k in GPA_FEATURES}}

@st.cache_resource(show_spinner="Clustering cohort…")
def build_clusters(stu: pd.DataFrame, k: int = 4):
    X = pd.DataFrame({
        "GPA": stu["current_gpa"].values,
        "Attendance %": stu["cumulative_attendance_pct"].values,
        "Grade Level": stu["grade_level"].values,
        "Scholarship": (stu["scholarship_flag"]=="Yes").astype(int).values,
    })
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaler.transform(X))
    return {"labels": km.labels_,
            "centers": scaler.inverse_transform(km.cluster_centers_),
            "features": list(X.columns)}

@st.cache_resource(show_spinner="Scanning anomalies…")
def detect_anomalies(stu: pd.DataFrame):
    X = stu[["current_gpa","cumulative_attendance_pct","grade_level"]].values
    iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
    return {"scores": iso.score_samples(X), "flags": iso.predict(X)}


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='padding:.5rem 0 1.4rem 0;'>
  <div style='font-size:1.55rem; font-weight:800; color:white; letter-spacing:-0.02em;'>
    🎓 Nexus <span style='color:#A78BFA;'>Analytics</span>
  </div>
  <div style='font-size:.72rem; color:#CBD5E1; margin-top:.25rem;
              letter-spacing:.1em; text-transform:uppercase;'>
    Academic Intelligence Platform
  </div>
</div>""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Workbook (.xlsx)", type="xlsx")
candidates = [Path("academic_realistic.xlsx"),
              Path("academic_multi_school_dashboard_populated_10000.xlsx")]
source = uploaded if uploaded is not None else next((p for p in candidates if p.exists()), None)

if source is None:
    st.title("Welcome to Nexus Analytics")
    st.info("Please upload the academic workbook in the sidebar to begin.")
    st.stop()

sheets = load_workbook(source)
views  = build_views(sheets)
stu, rec, att = views["students"], views["records"], views["attendance"]
teachers, schools, principals = views["teachers"], views["schools"], views["principals"]
parents = views["parents"]

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🎛 Filters")

f_school = st.sidebar.multiselect("School", sorted(stu["school"].dropna().unique()))
f_region = st.sidebar.multiselect("Region", sorted(stu["region"].dropna().unique()))
f_grade  = st.sidebar.multiselect("Grade",  sorted(stu["grade_level"].dropna().unique()))
f_risk   = st.sidebar.multiselect("Risk",   sorted(stu["academic_risk_flag"].dropna().unique()))
f_gender = st.sidebar.multiselect("Gender", sorted(stu["gender"].dropna().unique()))

def filter_students(df):
    if f_school: df = df[df["school"].isin(f_school)]
    if f_region: df = df[df["region"].isin(f_region)]
    if f_grade:  df = df[df["grade_level"].isin(f_grade)]
    if f_risk:   df = df[df["academic_risk_flag"].isin(f_risk)]
    if f_gender: df = df[df["gender"].isin(f_gender)]
    return df

fstu = filter_students(stu)
ids  = set(fstu["student_id"])
frec = rec[rec["student_id"].isin(ids)]       if ids else rec.iloc[0:0]
fatt = att[att["stakeholder_id"].isin(ids)]   if ids else att.iloc[0:0]
ftch = teachers if not f_school else teachers[teachers["school"].isin(f_school)]

st.sidebar.markdown("---")
page = st.sidebar.radio("📂 Navigation", list(THEMES.keys()), label_visibility="collapsed")

if fstu.empty:
    st.error("⚠️ No rows match the current filters. Please clear some filters in the sidebar.")
    st.stop()

theme = THEMES[page]
apply_theme(theme)
render_header(page)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "Executive Overview":

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Schools",      stu["school"].nunique())
    c2.metric("Students",     f"{len(fstu):,}")
    c3.metric("Teachers",     f"{len(teachers):,}")
    c4.metric("Avg GPA",      round(fstu["current_gpa"].mean(), 2),
              delta=round(fstu["current_gpa"].mean() - stu["current_gpa"].mean(), 2))
    c5.metric("Attendance %", round(fstu["cumulative_attendance_pct"].mean(), 1))
    c6.metric("High Risk %",  f"{(fstu['academic_risk_flag']=='High').mean()*100:.1f}%")

    tab1, tab2, tab3 = st.tabs(["📈 Summary", "🏆 School Rankings", "🌐 Distribution"])

    with tab1:
        section("Core Distributions")
        col1, col2, col3 = st.columns(3)

        fig = px.histogram(fstu, x="current_gpa", nbins=25, title="How are GPAs distributed across the cohort?",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=fstu["current_gpa"].mean(), line_dash="dash",
                      line_color=theme["accent"], line_width=3,
                      annotation_text=f"Mean {fstu['current_gpa'].mean():.2f}",
                      annotation_position="top right", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="GPA (0–4 scale)", yaxis_title="Number of students")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)
        with col1: caption("<b>What:</b> count of students in each GPA bucket, with the cohort mean marked. "
                           "<b>Look for:</b> a tall left tail (many low-GPA students) or bimodal peaks "
                           "(two distinct performer groups). "
                           "<b>Use case:</b> flag if bars to the left of the mean dominate — that's your "
                           "intervention pool.")

        fig = px.histogram(fstu, x="cumulative_attendance_pct", nbins=25,
                           title="Where does student attendance cluster?",
                           color_discrete_sequence=[theme["accent"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=fstu["cumulative_attendance_pct"].mean(), line_dash="dash",
                      line_color=theme["primary"], line_width=3,
                      annotation_text=f"Mean {fstu['cumulative_attendance_pct'].mean():.1f}%",
                      annotation_position="top right", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="Number of students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)
        with col2: caption("<b>What:</b> student counts across attendance percentage buckets. "
                           "<b>Look for:</b> anyone below 70% — chronic-absence territory. "
                           "<b>Use case:</b> generate a counselor outreach list from the left-most bars; "
                           "these students have the highest attrition risk.")

        fig = px.histogram(fstu, x="performance_index", nbins=25,
                           title="Who needs extra attention on the combined score?",
                           color_discrete_sequence=[theme["palette"][2]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(xaxis_title="Performance Index (GPA×25 + Attendance×0.5)",
                          yaxis_title="Number of students")
        col3.plotly_chart(style_fig(fig, theme, show_legend=False, height=300, bargap=0.1),
                          use_container_width=True)
        with col3: caption("<b>What:</b> combined academic-engagement score on a 0–100 scale. "
                           "<b>Look for:</b> the shape — left-skewed is healthy, right-skewed means many strugglers. "
                           "<b>Use case:</b> rank students for scholarship committees using a single "
                           "holistic metric instead of GPA alone.")

        section("Composition")
        col1, col2, col3 = st.columns(3)

        rc = fstu["academic_risk_flag"].value_counts().reset_index(); rc.columns = ["risk","count"]
        fig = px.pie(rc, names="risk", values="count", hole=0.6,
                     title="What share of students are flagged at risk?",
                     color="risk",
                     color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"})
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        fig.add_annotation(text=f"<b>{len(fstu):,}</b><br>students",
                           showarrow=False, font=dict(size=15, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme, height=300), use_container_width=True)
        with col1: caption("<b>What:</b> the split of students into Low/Medium/High academic risk buckets. "
                           "<b>Look for:</b> the red slice size — over 15% is concerning. "
                           "<b>Use case:</b> report red-slice percentage quarterly to school boards as a "
                           "core academic health KPI.")

        gc = fstu["gender"].value_counts().reset_index(); gc.columns = ["gender","count"]
        fig = px.pie(gc, names="gender", values="count", hole=0.6,
                     title="Is gender balance maintained across enrolment?",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=300), use_container_width=True)
        with col2: caption("<b>What:</b> male-female enrolment split. "
                           "<b>Look for:</b> large imbalance (>60:40) that persists over time. "
                           "<b>Use case:</b> compare this to regional gender ratios for equity audits, "
                           "and track quarter-over-quarter for retention monitoring.")

        gl = fstu["grade_level"].value_counts().sort_index().reset_index(); gl.columns = ["grade","count"]
        fig = px.funnel(gl.sort_values("count", ascending=True), x="count", y="grade",
                        title="Are higher grades losing students?",
                        color_discrete_sequence=[theme["primary"]])
        fig.update_traces(textfont=dict(color="white", size=12, family="Inter"))
        fig.update_layout(xaxis_title="Number of students", yaxis_title="Grade level")
        col3.plotly_chart(style_fig(fig, theme, show_legend=False, height=300),
                          use_container_width=True)
        with col3: caption("<b>What:</b> headcount per grade in funnel form (widest grade at top). "
                           "<b>Look for:</b> sharp drop-offs — suggests transfers or drop-outs at that transition. "
                           "<b>Use case:</b> target retention programmes at grades where the funnel narrows fastest.")

        section("Correlations")
        sample = fstu.sample(min(2500, len(fstu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="academic_risk_flag", opacity=0.65,
                         title="Does showing up lead to better grades? (each dot = 1 student)",
                         color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"},
                         hover_data=["school","grade_level"])
        fig = add_trendline(fig, sample["cumulative_attendance_pct"],
                            sample["current_gpa"], color=theme["primary"])
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("<b>What:</b> the core correlation in the data — every student as a dot, coloured by risk. "
                "<b>Look for:</b> the trendline slope — steep = strong relationship; flat = attendance "
                "doesn't predict GPA here. Red dots cluster in the bottom-left (low on both). "
                "<b>Use case:</b> prove the business case for attendance enforcement programmes — "
                "'raising attendance by 10% historically lifts GPA by X points'.")

    with tab2:
        by_school = (fstu.groupby("school").agg(
            students=("student_id","count"), avg_gpa=("current_gpa","mean"),
            avg_att=("cumulative_attendance_pct","mean"),
            high_risk=("academic_risk_flag", lambda s: (s=="High").mean()*100),
        ).reset_index().sort_values("avg_gpa", ascending=False))

        section("Top performers")
        top_n = st.slider("Show top N schools", 5, len(by_school), min(10, len(by_school)))

        fig = px.bar(by_school.head(top_n), x="school", y="avg_gpa",
                     title=f"Which are our strongest {top_n} schools right now?",
                     text_auto=".2f", color="avg_gpa", color_continuous_scale=theme["scale"],
                     hover_data=["students","avg_att","high_risk"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("<b>What:</b> schools ranked by average GPA, with enrolment and attendance in hover. "
                "<b>Look for:</b> the GPA gap between #1 and the rest; study top schools for best practices. "
                "<b>Use case:</b> choose mentor schools for peer-learning programmes — pair a top-3 school "
                "with a bottom-3 school to transfer teaching methods.")

        fig = px.scatter(by_school, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school",
                         title="Which schools are strong on both attendance AND grades?",
                         color="high_risk",
                         color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                         labels={"high_risk":"High-risk %"})
        fig.add_hline(y=by_school["avg_gpa"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean GPA", annotation_position="top left",
                      annotation_font_color="#0F172A")
        fig.add_vline(x=by_school["avg_att"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean attendance", annotation_position="bottom right",
                      annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Average attendance %", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, height=500), use_container_width=True)
        caption("<b>What:</b> schools as bubbles — size is enrolment, colour is high-risk %. Dotted lines "
                "split the chart into 4 quadrants. "
                "<b>Look for:</b> bottom-left = trouble on both; top-right = consistent excellence; "
                "top-left = high GPA despite low attendance (suspicious — may indicate grade inflation). "
                "<b>Use case:</b> board-level escalation — any bottom-left school is a 'crisis watch' case.")

    with tab3:
        section("Geographic & board distribution")
        col1, col2 = st.columns(2)

        regional = fstu.groupby(["region","school"]).size().reset_index(name="count")
        fig = px.treemap(regional, path=["region","school"], values="count",
                         title="Where are our students concentrated geographically?",
                         color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(marker=dict(cornerradius=6), textfont=dict(color="white", size=12))
        col1.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        with col1: caption("<b>What:</b> rectangle size = student count, grouped by region then school. "
                           "<b>Look for:</b> regions that dominate overall and which schools drive each region. "
                           "<b>Use case:</b> regional staffing decisions — large rectangles need more HR/admin support.")

        boards = fstu.groupby(["board_name","school_type"]).size().reset_index(name="count")
        fig = px.sunburst(boards, path=["board_name","school_type"], values="count",
                          title="How does our curriculum portfolio split by board and level?",
                          color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        with col2: caption("<b>What:</b> inner ring is board (CBSE/ICSE/State), outer ring is school type. "
                           "<b>Look for:</b> imbalance — one board dominating portfolio is a market-risk. "
                           "<b>Use case:</b> expansion strategy — diversify by investing in under-represented "
                           "board/type combinations.")

        rg = fstu.groupby("region")["current_gpa"].mean().reset_index()
        fig = px.bar_polar(rg, r="current_gpa", theta="region",
                           title="Which region produces the strongest GPA outcomes?",
                           color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, title_x=0.02,
                          font=dict(color="#0F172A", family="Inter"),
                          polar=dict(bgcolor="rgba(0,0,0,0)",
                                     angularaxis=dict(tickfont=dict(color="#0F172A", size=12)),
                                     radialaxis=dict(tickfont=dict(color="#334155", size=11))))
        st.plotly_chart(fig, use_container_width=True)
        caption("<b>What:</b> average GPA per region, plotted around a circle — longer arms = higher GPA. "
                "<b>Look for:</b> an arm much shorter than the others signals an underperforming region. "
                "<b>Use case:</b> regional director performance reviews and equity-based resource allocation.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: STUDENTS 360°
# ═══════════════════════════════════════════════════════════════════
elif page == "Students 360°":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("In view",     f"{len(fstu):,}")
    c2.metric("High risk",   int((fstu["academic_risk_flag"]=="High").sum()))
    c3.metric("Scholarship", int((fstu["scholarship_flag"]=="Yes").sum()))
    c4.metric("IEP",         int((fstu["iep_flag"]=="Yes").sum()))
    c5.metric("Hostellers",  int((fstu["hostel_opted"]=="Yes").sum()))

    tab1, tab2, tab3 = st.tabs(["👥 Cohort", "🎯 Performance", "🔍 Segments"])

    with tab1:
        section("Grade-level distributions")
        col1, col2 = st.columns(2)

        fig = px.violin(fstu, x="grade_level", y="current_gpa", box=True, color="grade_level",
                        color_discrete_sequence=theme["palette"],
                        title="Does performance vary by grade level?")
        fig.update_layout(xaxis_title="Grade level", yaxis_title="GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col1: caption("<b>What:</b> GPA distribution shape per grade — wider = more students at that GPA. "
                           "<b>Look for:</b> grades where the violin stretches far down (many low performers) "
                           "or splits into two peaks (bimodal class). "
                           "<b>Use case:</b> decide which grades need remedial class sections vs enrichment streams.")

        fig = px.violin(fstu, x="grade_level", y="cumulative_attendance_pct", box=True,
                        color="grade_level", color_discrete_sequence=theme["palette"],
                        title="Which grades show the best engagement?")
        fig.update_layout(xaxis_title="Grade level", yaxis_title="Attendance %")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col2: caption("<b>What:</b> attendance distribution per grade. "
                           "<b>Look for:</b> narrow violins = uniform engagement; wide violins = polarized class. "
                           "<b>Use case:</b> prioritise grade-level attendance campaigns where violin is wide and low.")

        section("Demographic splits")
        col1, col2, col3 = st.columns(3)

        fig = px.box(fstu, x="gender", y="current_gpa", color="gender",
                     title="Is there a gender gap in academic performance?", points="all",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(marker=dict(size=3, opacity=0.3))
        fig.update_layout(xaxis_title="Gender", yaxis_title="GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col1: caption("<b>What:</b> every student as a dot plus summary box per gender. "
                           "<b>Look for:</b> the median lines — are they visibly different? Outliers beyond whiskers? "
                           "<b>Use case:</b> inform gender-equity initiatives and counteract possible bias in assessment.")

        moi = fstu["medium_of_instruction"].value_counts().reset_index(); moi.columns = ["medium","count"]
        fig = px.bar(moi, x="medium", y="count", title="What language are most students learning in?",
                     text_auto=True, color="medium", color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=13))
        fig.update_layout(xaxis_title="Medium", yaxis_title="Number of students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col2: caption("<b>What:</b> student counts by medium of instruction (English/Hindi/Regional). "
                           "<b>Look for:</b> unexpected shifts vs last year — may signal regional policy change. "
                           "<b>Use case:</b> budget textbook translations and bilingual teacher hiring.")

        cc = fstu["caste_category"].value_counts().reset_index(); cc.columns = ["cat","count"]
        fig = px.pie(cc, names="cat", values="count", hole=0.55, title="How diverse is our student demographic?",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=12, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col3.plotly_chart(style_fig(fig, theme, height=320), use_container_width=True)
        with col3: caption("<b>What:</b> caste category distribution across the cohort. "
                           "<b>Look for:</b> the reserved-category share; compare to the regional demographic. "
                           "<b>Use case:</b> RTE/reservation compliance reporting and targeted scholarship design.")

    with tab2:
        section("Scholarship & IEP impact")
        col1, col2 = st.columns(2)

        s_gpa = fstu.groupby("scholarship_flag")["current_gpa"].mean().reset_index()
        fig = px.bar(s_gpa, x="scholarship_flag", y="current_gpa",
                     title="Are our scholarships actually improving outcomes?",
                     text_auto=".2f", color="scholarship_flag",
                     color_discrete_sequence=[theme["palette"][2], theme["primary"]])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Scholarship status", yaxis_title="Average GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col1: caption("<b>What:</b> average GPA for scholarship holders vs non-holders. "
                           "<b>Look for:</b> if the green bar is notably taller, the scholarship programme is working. "
                           "<b>Use case:</b> annual scholarship impact review — a flat comparison suggests redesigning criteria.")

        i_gpa = fstu.groupby("iep_flag")["current_gpa"].mean().reset_index()
        fig = px.bar(i_gpa, x="iep_flag", y="current_gpa",
                     title="Are IEP students achieving target support levels?",
                     text_auto=".2f", color="iep_flag",
                     color_discrete_sequence=[theme["palette"][3], theme["palette"][5]])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="IEP status", yaxis_title="Average GPA")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=320),
                          use_container_width=True)
        with col2: caption("<b>What:</b> average GPA for IEP students vs non-IEP. "
                           "<b>Look for:</b> IEP GPA should be LOWER (confirming the programme is reaching the right students) "
                           "but the gap should be narrowing over time. "
                           "<b>Use case:</b> audit whether IEP is reaching struggling students; track gap-closure year-over-year.")

        section("GPA × Attendance density")
        fig = px.density_heatmap(fstu, x="cumulative_attendance_pct", y="current_gpa",
                                 nbinsx=25, nbinsy=25,
                                 title="Where do most students cluster (GPA × Attendance)?",
                                 color_continuous_scale=theme["scale"])
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=450),
                        use_container_width=True)
        caption("A heatmap showing where students concentrate. Darker cells = more students "
                "at that (attendance, GPA) combination. The diagonal pattern confirms "
                "attendance and GPA rise together.")

    with tab3:
        section("Per-school performance")
        sc = fstu.groupby("school")["current_gpa"].mean().reset_index().sort_values(
            "current_gpa", ascending=True)
        fig = px.bar(sc, x="current_gpa", y="school", orientation="h",
                     title="Which school is producing the strongest students?", text_auto=".2f",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=11))
        fig.update_layout(xaxis_title="Average GPA", yaxis_title="School")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)
        caption("Horizontal bars let you rank all schools by GPA on one screen. "
                "The lightest bars at the bottom are the schools that need the most attention.")

        section("Multidimensional student profile")
        sample_pc = fstu.sample(min(500, len(fstu)), random_state=1).copy()
        sample_pc["risk_num"] = sample_pc["academic_risk_flag"].map({"Low":0, "Medium":1, "High":2})
        fig = px.parallel_coordinates(
            sample_pc, dimensions=["grade_level","cumulative_attendance_pct",
                                    "current_gpa","performance_index","risk_num"],
            color="risk_num",
            color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
            title="Parallel Coordinates · Cohort Profile (sample of 500)",
            labels={"risk_num":"Risk (0=Low, 2=High)",
                    "cumulative_attendance_pct":"Attendance %",
                    "performance_index":"Perf. Index", "current_gpa":"GPA",
                    "grade_level":"Grade"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420, title_x=0.02,
                          font=dict(color="#0F172A"))
        st.plotly_chart(fig, use_container_width=True)
        caption("Each line is one student crossing five metrics. Red lines are high-risk "
                "students — you can literally trace their paths across attendance, GPA, "
                "and performance axes to see exactly where they lag.")

        section("Top & bottom 10 students")
        col1, col2 = st.columns(2)
        top10 = fstu.nlargest(10, "performance_index")[
            ["student_id","full_name","school","current_gpa",
             "cumulative_attendance_pct","performance_index"]]
        bot10 = fstu.nsmallest(10, "performance_index")[
            ["student_id","full_name","school","current_gpa",
             "cumulative_attendance_pct","performance_index"]]
        col1.markdown("**🏆 Top 10 performers**")
        col1.dataframe(top10, use_container_width=True, hide_index=True)
        col2.markdown("**📉 Bottom 10 performers**")
        col2.dataframe(bot10, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGE: ACADEMIC PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif page == "Academic Performance":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Exam records", f"{len(frec):,}")
    c2.metric("Pass rate",    f"{(frec['pass_fail']=='Pass').mean()*100:.1f}%")
    c3.metric("Avg %",        round(frec["percentage"].mean(), 1))
    c4.metric("Subjects",     frec["subject_name"].nunique())
    c5.metric("Terms",        frec["term_name"].nunique())

    tab1, tab2, tab3 = st.tabs(["🧾 Results", "📘 Subjects", "📈 Term Trends"])

    with tab1:
        section("Grade & pass distributions")
        col1, col2 = st.columns(2)

        order = ["A1","A2","B1","B2","C1","C2","D","E"]
        gd = frec["grade_awarded"].value_counts().reindex(order, fill_value=0).reset_index()
        gd.columns = ["grade","count"]
        fig = px.bar(gd, x="grade", y="count", text_auto=True,
                     title="How are exam grades awarded across letter bands?",
                     color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=12))
        fig.update_layout(xaxis_title="Letter grade", yaxis_title="Exam count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("<b>What:</b> count of exams awarded each letter grade (A1 to E). "
                           "<b>Look for:</b> a healthy bell centered around B1/B2; fat tails on D/E signal systemic issues. "
                           "<b>Use case:</b> quickly gauge examiner calibration — too many A1s may mean grade inflation.")

        pf = frec["pass_fail"].value_counts().reset_index(); pf.columns = ["status","count"]
        fig = px.pie(pf, names="status", values="count", hole=0.6, title="What is our overall pass rate?",
                     color="status", color_discrete_map={"Pass":"#10B981","Fail":"#EF4444"})
        fig.update_traces(textposition="outside", textinfo="label+percent+value",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)
        with col2: caption("<b>What:</b> overall pass/fail split for the current filter. "
                           "<b>Look for:</b> red slice over 10% is a red flag for the academic dean. "
                           "<b>Use case:</b> headline metric for monthly academic council reports.")

        section("Percentage distribution")
        fig = px.histogram(frec, x="percentage", nbins=30, title="Where do exam scores cluster around the pass line?",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=35, line_dash="dash", line_color="#DC2626", line_width=3,
                      annotation_text="Pass threshold (35%)",
                      annotation_position="top right", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Exam percentage", yaxis_title="Exam count")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360, bargap=0.1),
                        use_container_width=True)
        caption("Full percentage distribution with the 35% pass line. Exams falling left "
                "of the red line are failures — the count of failing exams is readable "
                "directly off the chart.")

    with tab2:
        section("Subject-level performance")
        col1, col2 = st.columns(2)

        fig = px.box(frec, x="subject_name", y="percentage", color="subject_name",
                     title="Which subjects show the widest score variance?", points=False,
                     color_discrete_sequence=theme["palette"])
        fig.update_layout(xaxis_title="Subject", yaxis_title="Exam percentage")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col1: caption("<b>What:</b> score distribution per subject as boxes (line = median). "
                           "<b>Look for:</b> a subject whose entire box sits low means systemic difficulty across the cohort. "
                           "<b>Use case:</b> trigger syllabus review for low-box subjects; bring in subject-matter "
                           "consultants for curriculum redesign.")

        sp = (frec.groupby("subject_name")
                  .apply(lambda d: (d["pass_fail"]=="Pass").mean()*100, include_groups=False)
                  .reset_index(name="pass_rate").sort_values("pass_rate"))
        fig = px.bar(sp, x="pass_rate", y="subject_name", orientation="h",
                     title="Which subjects have the lowest pass rates?", text_auto=".1f",
                     color="pass_rate", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=13))
        fig.update_layout(xaxis_title="Pass rate %", yaxis_title="Subject")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col2: caption("<b>What:</b> share of students passing each subject. "
                           "<b>Look for:</b> the shortest bar — that's the hardest subject. A gap of >10 points "
                           "between top and bottom subject signals uneven difficulty. "
                           "<b>Use case:</b> allocate additional instructional hours to the lowest-pass-rate subject.")

        section("Subject profile")
        subj_avg = frec.groupby("subject_name").agg(
            exam_pct=("percentage","mean"), assignment=("assignment_score","mean"),
            project=("project_score","mean"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=subj_avg["exam_pct"], theta=subj_avg["subject_name"],
                                      fill='toself', name="Exam %", line_color=theme["primary"]))
        fig.add_trace(go.Scatterpolar(r=subj_avg["assignment"]*5, theta=subj_avg["subject_name"],
                                      fill='toself', name="Assignment×5",
                                      line_color=theme["palette"][2]))
        fig.add_trace(go.Scatterpolar(r=subj_avg["project"]*10, theta=subj_avg["subject_name"],
                                      fill='toself', name="Project×10",
                                      line_color=theme["palette"][3]))
        fig.update_layout(
            title="How do exam vs assignment vs project scores compare per subject?",
            polar=dict(bgcolor="rgba(248,250,252,0.5)",
                      radialaxis=dict(visible=True, range=[0, 100],
                                      tickfont=dict(color="#334155", size=11)),
                      angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
            paper_bgcolor="rgba(0,0,0,0)", height=440, title_x=0.02,
            font_family='"Inter", sans-serif', font_color="#0F172A",
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, font=dict(color="#0F172A")))
        st.plotly_chart(fig, use_container_width=True)
        caption("Radar comparing three assessment types per subject. If one polygon is much "
                "smaller than others, students excel at that type (exam/assignment/project) "
                "in a way that's uneven across subjects.")

        section("Assignments vs exam scores")
        sample = frec.sample(min(2000, len(frec)), random_state=1)
        fig = px.scatter(sample, x="assignment_score", y="marks_obtained", color="subject_name",
                         opacity=0.6, title="Does classwork genuinely predict exam success?",
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, sample["assignment_score"], sample["marks_obtained"],
                            color="#0F172A")
        fig.update_layout(xaxis_title="Assignment score (out of 20)",
                          yaxis_title="Exam marks obtained (out of 100)")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("Does classwork predict exam performance? A steep upward trendline says yes — "
                "assignment work is genuinely developing exam skills.")

    with tab3:
        ts = frec.groupby(["term_name","subject_name"])["percentage"].mean().reset_index()
        fig = px.line(ts, x="term_name", y="percentage", color="subject_name",
                      markers=True, title="Are students improving from Term 1 to Term 3?",
                      color_discrete_sequence=theme["palette"])
        fig.update_traces(line_width=3, marker=dict(size=11, line=dict(color="white", width=2)))
        fig.update_layout(xaxis_title="Term", yaxis_title="Average percentage")
        st.plotly_chart(style_fig(fig, theme, height=400), use_container_width=True)
        caption("How each subject's average score changes from Term 1 → 2 → 3. Lines trending "
                "upward mean students are improving; flat or declining lines flag subjects "
                "that need a pedagogy review.")

        tp = (frec.groupby("term_name")
                  .apply(lambda d: (d["pass_fail"]=="Pass").mean()*100, include_groups=False)
                  .reset_index(name="pass_rate"))
        fig = px.bar(tp, x="term_name", y="pass_rate", text_auto=".1f",
                     title="Is pass rate trending up through the year?",
                     color="pass_rate", color_continuous_scale=theme["scale"])
        fig.update_layout(yaxis_range=[80, 100], xaxis_title="Term", yaxis_title="Pass rate %")
        fig.update_traces(textfont=dict(color="white", size=14))
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)
        caption("Overall pass rate per term. If Term 3 is much higher than Term 1, students "
                "are adapting; a drop in Term 3 could mean exam fatigue.")

        pivot = frec.pivot_table(index="subject_name", columns="term_name",
                                  values="percentage", aggfunc="mean")
        fig = px.imshow(pivot, text_auto=".1f", title="Which subject/term combinations are strongest and weakest?",
                        color_continuous_scale=theme["scale"], aspect="auto")
        fig.update_traces(textfont=dict(size=14, color="#0F172A"))
        fig.update_layout(xaxis_title="Term", yaxis_title="Subject")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("A grid cross-tabulating subject and term. Darker cells = higher average. "
                "Look for a single dark column (one great term) or a single light row "
                "(one weak subject) to diagnose where the action is.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: FACULTY ANALYTICS
# ═══════════════════════════════════════════════════════════════════
elif page == "Faculty Analytics":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Teachers",       len(ftch))
    c2.metric("Avg rating",     round(ftch["teacher_performance_rating"].mean(), 2))
    c3.metric("Avg exp (yrs)",  round(ftch["years_experience"].mean(), 1))
    c4.metric("Avg attendance", f"{ftch['teacher_attendance_pct'].mean():.1f}%")
    c5.metric("Departments",    ftch["department"].nunique())

    tab1, tab2, tab3 = st.tabs(["🏛 Composition", "⭐ Performance", "⚖ Workload"])

    with tab1:
        section("Department structure")
        col1, col2 = st.columns(2)

        dept = ftch["department"].value_counts().reset_index(); dept.columns = ["department","count"]
        fig = px.bar(dept, x="department", y="count", text_auto=True,
                     color="department", title="How is teaching capacity distributed across departments?",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=13))
        fig.update_layout(xaxis_title="Department", yaxis_title="Teacher count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("<b>What:</b> teacher headcount per academic department. "
                           "<b>Look for:</b> disproportionately small departments relative to student demand. "
                           "<b>Use case:</b> inform next year's teacher-hiring plan and budget allocation.")

        emp = ftch["employment_type"].value_counts().reset_index(); emp.columns = ["type","count"]
        fig = px.pie(emp, names="type", values="count", hole=0.55, title="What is our permanent vs contract vs visiting mix?",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)
        with col2: caption("<b>What:</b> permanent vs contract vs visiting faculty split. "
                           "<b>Look for:</b> contract share > 30% is a retention risk — contract teachers churn more. "
                           "<b>Use case:</b> HR workforce planning — convert top contract performers to permanent roles.")

        section("Qualifications & gender")
        col1, col2 = st.columns(2)

        q = ftch["highest_qualification"].value_counts().reset_index(); q.columns = ["qual","count"]
        fig = px.bar(q, x="count", y="qual", orientation="h", text_auto=True,
                     title="How qualified is our teaching workforce?", color="count",
                     color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Teacher count", yaxis_title="Qualification")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("<b>What:</b> distribution of teachers by highest degree. "
                           "<b>Look for:</b> a strong Master's+PhD proportion is a quality indicator for "
                           "board reviews and parent enrolment decisions. "
                           "<b>Use case:</b> marketing (prospective parents) and accreditation audits.")

        g = ftch["gender"].value_counts().reset_index(); g.columns = ["gender","count"]
        fig = px.pie(g, names="gender", values="count", hole=0.55, title="Is faculty gender representation balanced?",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=340), use_container_width=True)
        with col2: caption("<b>What:</b> gender composition of the teaching workforce. "
                           "<b>Look for:</b> skew below 35% either way — both extremes limit role-modelling for students. "
                           "<b>Use case:</b> diversity hiring targets and gender-balanced leadership pipelines.")

    with tab2:
        fig = px.box(ftch, x="department", y="teacher_performance_rating",
                     color="department", title="Which departments are scoring highest on performance?",
                     points="all", color_discrete_sequence=theme["palette"])
        fig.update_traces(marker=dict(size=3, opacity=0.4))
        fig.update_layout(xaxis_title="Department", yaxis_title="Performance rating (1–5)")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                        use_container_width=True)
        caption("Each dot is one teacher. Hovering reveals their rating; the box shows the "
                "department distribution. Compare medians (line in box) to see which "
                "departments are strongest.")

        fig = px.scatter(ftch, x="years_experience", y="teacher_performance_rating",
                         color="department", opacity=0.7, size="classes_assigned_count",
                         title="Do senior teachers perform better? (bubble size = classes taught)",
                         hover_data=["full_name","school"],
                         color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, ftch["years_experience"],
                            ftch["teacher_performance_rating"], color="#0F172A")
        fig.update_layout(xaxis_title="Years of experience",
                          yaxis_title="Performance rating (1–5)")
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("Do senior teachers perform better? The trendline answers at a glance. "
                "Bubble size shows workload — big dots in the bottom-right would be "
                "experienced but overloaded teachers.")

        section("Top 10 teachers by rating")
        top = ftch.nlargest(10, "teacher_performance_rating")[
            ["full_name","department","years_experience","teacher_performance_rating","school"]]
        st.dataframe(top, use_container_width=True, hide_index=True)

    with tab3:
        col1, col2 = st.columns(2)

        fig = px.histogram(ftch, x="weekly_workload_hours", nbins=20,
                           title="Are teaching hours distributed equitably?", color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.update_layout(xaxis_title="Weekly teaching hours", yaxis_title="Teacher count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340, bargap=0.1),
                          use_container_width=True)
        with col1: caption("<b>What:</b> histogram of weekly teaching hours per teacher. "
                           "<b>Look for:</b> a long right tail (some teachers with 40+ hours) while peers carry 25. "
                           "<b>Use case:</b> workload rebalancing — overworked teachers predict absenteeism and resignations.")

        fig = px.density_heatmap(ftch, x="weekly_workload_hours",
                                 y="teacher_performance_rating", nbinsx=15, nbinsy=15,
                                 title="Does higher workload hurt teacher performance?",
                                 color_continuous_scale=theme["scale"])
        fig.update_layout(xaxis_title="Weekly workload hours",
                          yaxis_title="Performance rating")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("<b>What:</b> 2D density plot of workload vs rating. "
                           "<b>Look for:</b> brightest cells in high-hours/low-rating zone = burnout correlation. "
                           "<b>Use case:</b> data-driven case for workload caps in HR policy.")

        fig = px.scatter(ftch, x="teacher_attendance_pct", y="teacher_performance_rating",
                         color="department", opacity=0.7,
                         title="Do teachers who show up consistently score higher?", color_discrete_sequence=theme["palette"])
        fig = add_trendline(fig, ftch["teacher_attendance_pct"],
                            ftch["teacher_performance_rating"], color="#0F172A")
        fig.update_layout(xaxis_title="Teacher attendance %",
                          yaxis_title="Performance rating")
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("Is teacher attendance tied to their rating? A steep positive line confirms "
                "show-up-rate matters.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: ATTENDANCE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════
elif page == "Attendance Intelligence":

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Log entries", f"{len(fatt):,}")
    pres = (fatt["attendance_status"]=="Present").mean()*100 if len(fatt) else 0
    c2.metric("Present %",   f"{pres:.1f}%")
    c3.metric("Unique dates", fatt["attendance_date"].nunique())
    c4.metric("Unique students", fatt["stakeholder_id"].nunique())

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Time trends", "🏫 Comparisons"])

    with tab1:
        section("Status & reasons")
        col1, col2 = st.columns(2)

        status = fatt["attendance_status"].value_counts().reset_index(); status.columns = ["status","count"]
        fig = px.pie(status, names="status", values="count", hole=0.6,
                     title="What percentage of students are Present / Absent / Late / Leave?", color="status",
                     color_discrete_map={"Present":"#10B981","Absent":"#EF4444",
                                         "Late":"#F59E0B","Leave":"#94A3B8"})
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        fig.add_annotation(text=f"<b>{pres:.1f}%</b><br>present",
                           showarrow=False, font=dict(size=15, color="#0F172A"))
        col1.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)
        with col1: caption("<b>What:</b> Present/Absent/Late/Leave distribution with present% centred. "
                           "<b>Look for:</b> Late+Leave combined share — if over 10%, punctuality is the issue, not absence. "
                           "<b>Use case:</b> differentiate interventions — absence outreach vs tardiness policies.")

        absences = fatt[fatt["attendance_status"]!="Present"]
        if len(absences):
            reason = absences["reason"].value_counts().reset_index(); reason.columns = ["reason","count"]
            fig = px.bar(reason, x="reason", y="count", text_auto=True,
                         title="Why are students actually missing school?",
                         color="count", color_continuous_scale=theme["scale"])
            fig.update_traces(textfont=dict(color="white", size=13))
            fig.update_layout(xaxis_title="Reason", yaxis_title="Instance count")
            col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                              use_container_width=True)
            with col2: caption("Why aren't students in class? Illness is expected, but a tall "
                               "'Family' or 'Travel' bar suggests calendar-planning "
                               "conversations are needed.")

    with tab2:
        daily = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                      .groupby("attendance_date")["p"].mean().reset_index())
        daily["p"] *= 100
        fig = px.line(daily, x="attendance_date", y="p", title="Which specific days had the worst attendance?",
                      markers=True, color_discrete_sequence=[theme["primary"]])
        fig.update_traces(line_width=3, marker=dict(size=7, line=dict(color="white", width=1.5)))
        fig.add_hline(y=daily["p"].mean(), line_dash="dash", line_color=theme["accent"],
                      line_width=2, annotation_text=f"Mean {daily['p'].mean():.1f}%",
                      annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Date", yaxis_title="Present %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("Day-by-day attendance with a mean line. Sharp dips mark problem days — "
                "weather events, exam days, or local disruptions worth investigating.")

        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        dow = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                    .groupby("day_of_week")["p"].mean().reindex(dow_order).reset_index())
        dow["p"] *= 100
        fig = px.bar(dow, x="day_of_week", y="p", text_auto=".1f",
                     title="Is there a Monday slump or Friday fade?",
                     color="p", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(yaxis_range=[70, 100], xaxis_title="Day of week",
                          yaxis_title="Present %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)
        caption("Monday blues? Friday fade? This reveals weekly rhythm. Consistent "
                "Monday dips indicate a scheduling issue worth addressing.")

        weekly = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                       .groupby("week")["p"].mean().reset_index())
        weekly["p"] *= 100
        fig = px.area(weekly, x="week", y="p", title="Is weekly attendance trending up or down?",
                      color_discrete_sequence=[theme["primary"]])
        fig.update_traces(line_width=3, fillcolor=theme["primary"]+"30")
        fig.update_layout(xaxis_title="Week of year", yaxis_title="Present %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                        use_container_width=True)
        caption("Smoothed weekly view for spotting longer-term patterns without day-to-day "
                "noise.")

    with tab3:
        col1, col2 = st.columns(2)

        by_grade = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                         .groupby("grade_level")["p"].mean().reset_index())
        by_grade["p"] *= 100
        fig = px.bar(by_grade, x="grade_level", y="p", text_auto=".1f",
                     title="Which grades show the best engagement?",
                     color="p", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Grade", yaxis_title="Present %")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("<b>What:</b> present% by grade level. "
                           "<b>Look for:</b> Grade 11–12 typically dips due to coaching-class pulls; flag other grades. "
                           "<b>Use case:</b> parent-communication plan around tuition-class scheduling for seniors.")

        by_region = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                         .groupby("region")["p"].mean().reset_index())
        by_region["p"] *= 100
        fig = px.bar(by_region, x="region", y="p", text_auto=".1f",
                     title="Are any regions falling behind on attendance?",
                     color="p", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Region", yaxis_title="Present %")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("<b>What:</b> present% by region. "
                           "<b>Look for:</b> gaps over 5 points between regions — may reflect transport, climate, or local festivals. "
                           "<b>Use case:</b> region-specific attendance policy exceptions (monsoon-affected vs urban).")

        by_school = (fatt.assign(p=(fatt["attendance_status"]=="Present").astype(int))
                          .groupby("school")["p"].mean().reset_index()
                          .sort_values("p", ascending=True))
        by_school["p"] *= 100
        fig = px.bar(by_school, x="p", y="school", orientation="h", text_auto=".1f",
                     title="Which schools have the engagement problem?", color="p",
                     color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=11))
        fig.update_layout(xaxis_title="Present %", yaxis_title="School")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)
        caption("All schools ranked by presence. The bottom of the list is where principals "
                "and parent councils should focus outreach.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: SCHOOL NETWORK
# ═══════════════════════════════════════════════════════════════════
elif page == "School Network":

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Schools", len(schools))
    c2.metric("Boards",  schools["board_name"].nunique())
    c3.metric("States",  schools["state"].nunique())
    c4.metric("Regions", schools["region"].nunique())
    c5.metric("Capacity", f"{int(schools['student_capacity'].sum()):,}")

    tab1, tab2, tab3 = st.tabs(["🏆 Rankings", "🧭 Distribution", "🗺 Geography"])

    with tab1:
        summary = (stu.groupby("school").agg(
            students=("student_id","count"), avg_gpa=("current_gpa","mean"),
            avg_att=("cumulative_attendance_pct","mean"),
            high_risk=("academic_risk_flag", lambda s: (s=="High").mean()*100),
        ).reset_index().sort_values("avg_gpa", ascending=False))

        fig = px.bar(summary, x="school", y="avg_gpa", text_auto=".2f",
                     title="Which school is producing the strongest students?",
                     color="avg_gpa", color_continuous_scale=theme["scale"],
                     hover_data=["students","avg_att","high_risk"])
        fig.update_traces(textfont=dict(color="white", size=11))
        fig.update_layout(xaxis_title="School", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                        use_container_width=True)
        caption("All schools on one screen, ranked by their student's mean GPA. "
                "This is the quickest way to spot outliers on either end.")

        fig = px.scatter(summary, x="avg_att", y="avg_gpa", size="students",
                         hover_name="school", title="Which schools are leading on both attendance AND GPA?",
                         color="high_risk",
                         color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                         labels={"high_risk":"High-risk %"})
        fig.add_hline(y=summary["avg_gpa"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean GPA", annotation_font_color="#0F172A")
        fig.add_vline(x=summary["avg_att"].mean(), line_dash="dot", line_color="#64748B",
                      annotation_text="mean att.", annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Average attendance %", yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, height=480), use_container_width=True)
        caption("Each school as a bubble. Top-right quadrant = high performers on both axes. "
                "Red bubbles anywhere need urgent attention regardless of their position.")

    with tab2:
        col1, col2 = st.columns(2)

        boards = schools["board_name"].value_counts().reset_index(); boards.columns = ["board","count"]
        fig = px.pie(boards, names="board", values="count", hole=0.6, title="How many schools follow each examination board?",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col1.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)
        with col1: caption("<b>What:</b> network split across examination boards. "
                           "<b>Look for:</b> unusual dependence on one board (regulatory risk). "
                           "<b>Use case:</b> strategic portfolio diversification for expansion planning.")

        mgmt = schools["management_type"].value_counts().reset_index(); mgmt.columns = ["type","count"]
        fig = px.pie(mgmt, names="type", values="count", hole=0.6, title="What is our public/private/trust ownership mix?",
                     color_discrete_sequence=theme["palette"])
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          textfont=dict(size=13, color="#0F172A"),
                          marker=dict(line=dict(color="white", width=2)))
        col2.plotly_chart(style_fig(fig, theme, height=360), use_container_width=True)
        with col2: caption("<b>What:</b> ownership/management type split. "
                           "<b>Look for:</b> dominant governance model shapes how fast you can push changes. "
                           "<b>Use case:</b> phase policy rollouts by ownership group — trust schools may need longer runway.")

        col1, col2 = st.columns(2)
        regions = schools["region"].value_counts().reset_index(); regions.columns = ["region","count"]
        fig = px.bar(regions, x="region", y="count", text_auto=True, color="region",
                     title="Where are our campuses concentrated?", color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Region", yaxis_title="School count")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("<b>What:</b> school count per region. "
                           "<b>Look for:</b> geographic concentration — over-reliance on a single region is market risk. "
                           "<b>Use case:</b> expansion planning — invest in under-represented regions for growth.")

        types = schools["school_type"].value_counts().reset_index(); types.columns = ["type","count"]
        fig = px.bar(types, x="type", y="count", text_auto=True, color="type",
                     title="Schools by Type", color_discrete_sequence=theme["palette"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Type", yaxis_title="School count")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("<b>What:</b> school count by type (primary/secondary/senior). "
                           "<b>Look for:</b> gaps in the feeder pipeline — if secondary >> primary, future enrolment risk. "
                           "<b>Use case:</b> long-term capacity planning — align primary supply with secondary demand.")

        enrolment = stu.groupby("school").size().reset_index(name="enrolled")
        cap = schools[["school","student_capacity"]].merge(enrolment, on="school", how="left")
        cap["utilisation"] = (cap["enrolled"]/cap["student_capacity"])*100
        fig = px.bar(cap.sort_values("utilisation", ascending=False),
                     x="school", y="utilisation", text_auto=".0f",
                     title="Capacity Utilisation (%)",
                     color="utilisation", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=11))
        fig.update_layout(xaxis_title="School", yaxis_title="Utilisation %")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("Current enrolment as a percentage of built capacity. Over 100% means "
                "overcrowding; under 60% means wasted infrastructure.")

    with tab3:
        if {"latitude","longitude"}.issubset(schools.columns):
            st.map(schools[["latitude","longitude"]].dropna(), zoom=4)
            caption("<b>What:</b> campus locations plotted on an interactive map. "
                           "<b>Look for:</b> clustering or isolation; consider travel time for students. "
                           "<b>Use case:</b> transport-route optimisation and new-campus site selection.")

        perf = stu.groupby(["region","school_type"])["current_gpa"].mean().reset_index()
        pivot = perf.pivot(index="region", columns="school_type", values="current_gpa")
        fig = px.imshow(pivot, text_auto=".2f",
                        title="Avg GPA · Region × School Type",
                        color_continuous_scale=theme["scale"], aspect="auto")
        fig.update_traces(textfont=dict(size=14, color="#0F172A"))
        fig.update_layout(xaxis_title="School type", yaxis_title="Region")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                        use_container_width=True)
        caption("Regional performance cross-tabbed with school type. Dark cells are pockets "
                "of excellence; light cells flag where a specific (region, type) combo "
                "is underperforming.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: PARENTS & INCOME  (NEW — golden/amber theme)
# ═══════════════════════════════════════════════════════════════════
elif page == "Parents & Income":

    # Defensive: need parent_income and parent_education in the data
    if "parent_income" not in fstu.columns or fstu["parent_income"].isna().all():
        st.error("⚠️ **Parent income data missing.** This page requires the Parents "
                 "sheet with an `annual_income_band` column that successfully joins "
                 "to students via `primary_parent_id`. Please use the file "
                 "`academic_realistic.xlsx` provided with this dashboard.")
        st.stop()

    # Order income bands naturally instead of alphabetically
    income_order = ["<3 LPA", "3-6 LPA", "6-10 LPA", "10-20 LPA", "20-50 LPA", "50+ LPA"]
    available_bands = [b for b in income_order if b in fstu["parent_income"].unique()]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Students",    f"{len(fstu):,}")
    c2.metric("Income bands", len(available_bands))
    c3.metric("Occupations", fstu["parent_occupation"].nunique() if "parent_occupation" in fstu.columns else 0)
    c4.metric("Lowest-band GPA",
              f"{fstu[fstu['parent_income']==available_bands[0]]['current_gpa'].mean():.2f}"
              if available_bands else "—")
    c5.metric("Highest-band GPA",
              f"{fstu[fstu['parent_income']==available_bands[-1]]['current_gpa'].mean():.2f}"
              if available_bands else "—")

    tab1, tab2, tab3 = st.tabs(["💰 Income Impact", "🎓 Education & Occupation", "⚠️ Equity Lens"])

    with tab1:
        section("GPA across income bands")

        gpa_by_inc = (fstu.groupby("parent_income")["current_gpa"].mean()
                         .reindex(available_bands).reset_index())
        fig = px.bar(gpa_by_inc, x="parent_income", y="current_gpa", text_auto=".2f",
                     title="Does family income predict a student's academic success?",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent annual income band",
                          yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("<b>What:</b> average GPA within each parent income bracket. "
                "<b>Look for:</b> a rising staircase from left to right = socioeconomic gap exists; "
                "a flat chart = your schools are successfully levelling the field. "
                "<b>Use case:</b> this is the single most important equity chart — report it annually "
                "to the board and track whether the left-to-right gap is narrowing.")

        section("Distribution within each band")
        col1, col2 = st.columns(2)

        fig = px.violin(fstu[fstu["parent_income"].isin(available_bands)],
                        x="parent_income", y="current_gpa", box=True,
                        category_orders={"parent_income": available_bands},
                        color="parent_income", color_discrete_sequence=theme["palette"],
                        title="How much do outcomes vary WITHIN each income bracket?")
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="GPA")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col1: caption("<b>What:</b> full GPA distribution within each income bracket. "
                           "<b>Look for:</b> tall violins even in low-income brackets = some students overcome income "
                           "headwinds. Identify those students as role-model mentors. "
                           "<b>Use case:</b> proves 'income isn't destiny' — find and celebrate the outliers.")

        att_by_inc = (fstu.groupby("parent_income")["cumulative_attendance_pct"].mean()
                         .reindex(available_bands).reset_index())
        fig = px.bar(att_by_inc, x="parent_income", y="cumulative_attendance_pct",
                     text_auto=".1f", title="Do low-income families also struggle with attendance?",
                     color="cumulative_attendance_pct", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="Attendance %")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                          use_container_width=True)
        with col2: caption("<b>What:</b> average attendance % per income bracket. "
                           "<b>Look for:</b> if low-income attendance is much lower too, transport/health is a bottleneck. "
                           "<b>Use case:</b> design targeted transport subsidies or meal programmes for the "
                           "lowest-attendance income group.")

        section("Risk composition by income")
        risk_inc = (fstu.groupby(["parent_income","academic_risk_flag"]).size()
                        .reset_index(name="count"))
        fig = px.bar(risk_inc, x="parent_income", y="count", color="academic_risk_flag",
                     category_orders={"parent_income": available_bands,
                                      "academic_risk_flag": ["Low","Medium","High"]},
                     title="How many high-risk students are in each income bracket?",
                     color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"},
                     barmode="stack")
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="Student count")
        st.plotly_chart(style_fig(fig, theme, height=400),
                        use_container_width=True)
        caption("<b>What:</b> stacked risk composition (Low/Medium/High) per income bracket. "
                "<b>Look for:</b> red (High-risk) share shrinking as income rises = equity gap is real. "
                "<b>Use case:</b> prioritise counsellor outreach and scholarship allocation to the income "
                "brackets with the tallest red columns.")

    with tab2:
        section("Parent education and outcomes")

        edu_order = ["Below 10th","10th","12th","Graduate","Postgraduate","Doctorate"]
        available_edu = [e for e in edu_order if e in fstu["parent_education"].unique()]
        if not available_edu:
            available_edu = sorted(fstu["parent_education"].dropna().unique())

        edu_gpa = (fstu.groupby("parent_education")["current_gpa"].mean()
                       .reindex(available_edu).reset_index())
        fig = px.bar(edu_gpa, x="parent_education", y="current_gpa", text_auto=".2f",
                     title="Does parental education translate to student GPA?",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent highest education", yaxis_title="Student's average GPA")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                        use_container_width=True)
        caption("<b>What:</b> average student GPA by the parent's highest education level. "
                "<b>Look for:</b> a rising trend from Below-10th to Postgraduate = parental education matters. "
                "<b>Use case:</b> justify parent-literacy workshops — if effect is strong, investing in "
                "parent education is a backdoor to student outcomes.")

        section("Parent occupation")
        col1, col2 = st.columns(2)

        occ = fstu["parent_occupation"].value_counts().head(10).reset_index()
        occ.columns = ["occupation","count"]
        fig = px.bar(occ.sort_values("count"), x="count", y="occupation", orientation="h",
                     text_auto=True, title="Which 10 jobs do most of our parents hold?",
                     color="count", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Student count", yaxis_title="Occupation")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                          use_container_width=True)
        with col1: caption("<b>What:</b> 10 most common parent occupations across the network. "
                           "<b>Look for:</b> the mix of white-collar vs blue-collar vs entrepreneurial. "
                           "<b>Use case:</b> invite top occupations to career day; tailor parent-engagement "
                           "timing to working-hours norms.")

        occ_gpa = (fstu.groupby("parent_occupation")["current_gpa"].mean()
                       .reset_index().nlargest(10, "current_gpa"))
        fig = px.bar(occ_gpa.sort_values("current_gpa"), x="current_gpa", y="parent_occupation",
                     orientation="h", text_auto=".2f",
                     title="Which parent professions produce the top-performing students?",
                     color="current_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Average student GPA", yaxis_title="Parent occupation")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=400),
                          use_container_width=True)
        with col2: caption("<b>What:</b> top 10 occupations ranked by their children's GPA. "
                           "<b>Look for:</b> blue-collar occupations that rank surprisingly high — a hidden "
                           "success story of motivated parenting regardless of income. "
                           "<b>Use case:</b> counsellor talking points — 'your parents' occupation doesn't limit you'.")

        section("Income × Education heatmap")
        pivot = (fstu.groupby(["parent_income","parent_education"])["current_gpa"]
                     .mean().reset_index()
                     .pivot(index="parent_income", columns="parent_education", values="current_gpa")
                     .reindex(available_bands))
        fig = px.imshow(pivot, text_auto=".2f",
                        title="Joint effect: does high income AND high education compound gains?",
                        color_continuous_scale=theme["scale"], aspect="auto")
        fig.update_traces(textfont=dict(size=13, color="#0F172A"))
        fig.update_layout(xaxis_title="Parent education", yaxis_title="Parent income band")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("<b>What:</b> heatmap of average GPA at every (income × education) combination. "
                "<b>Look for:</b> the darkest cell — usually high-income + postgraduate parents. Also note "
                "any diagonal pattern (income alone vs education alone dominating). "
                "<b>Use case:</b> targeted intervention design — e.g., 'high income, low education' cell may "
                "need parent-engagement literacy programmes.")

    with tab3:
        section("The equity chart: scholarship impact across income bands")

        sch_df = (fstu.groupby(["parent_income","scholarship_flag"])["current_gpa"]
                      .mean().reset_index())
        fig = px.bar(sch_df, x="parent_income", y="current_gpa",
                     color="scholarship_flag", barmode="group",
                     category_orders={"parent_income": available_bands},
                     title="Are scholarships working hardest where they're needed most?",
                     color_discrete_map={"Yes":"#10B981","No":"#94A3B8"},
                     text_auto=".2f")
        fig.update_layout(xaxis_title="Parent income band", yaxis_title="Average GPA")
        fig.update_traces(textfont=dict(color="white", size=12))
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("<b>What:</b> scholarship vs non-scholarship GPA, grouped by income bracket. "
                "<b>Look for:</b> biggest green-vs-grey gaps in the LEFT (low-income) bars — that's where "
                "scholarships create the most value. "
                "<b>Use case:</b> annual scholarship programme evaluation and renewal budgeting.")

        section("High-risk students per income band")
        high_by_inc = (fstu.assign(is_high=(fstu["academic_risk_flag"]=="High").astype(int))
                            .groupby("parent_income")["is_high"].mean()
                            .reindex(available_bands).reset_index())
        high_by_inc["is_high"] *= 100
        fig = px.bar(high_by_inc, x="parent_income", y="is_high", text_auto=".1f",
                     title="The equity gauge: is the achievement gap real and measurable?",
                     color="is_high", color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=14))
        fig.update_layout(xaxis_title="Parent income band",
                          yaxis_title="% of students flagged High Risk")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("<b>What:</b> high-risk student % in each income bracket. "
                "<b>Look for:</b> a steep left-to-right decline = large income-driven risk gap. A flat line "
                "means your schools have equalised outcomes — the ideal. "
                "<b>Use case:</b> single-number equity KPI for board reports; track quarterly.")

        insight("This page exists to keep <b>equity visible</b>. Use it quarterly to check "
                "whether gaps between income bands are narrowing (good) or widening (bad), "
                "and to target scholarships and counseling where the need is greatest.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: STUDENT COMPARISON  (NEW — pink/rose theme)
# ═══════════════════════════════════════════════════════════════════
elif page == "Student Comparison":

    st.markdown("**Pick 2–5 students from the dropdown below to compare them across "
                "every metric.** The dropdown is searchable — type part of a name or ID.")

    # Build display label
    stu_label_df = fstu.copy()
    stu_label_df["label"] = (stu_label_df["full_name"] + " · "
                             + stu_label_df["student_id"] + " · "
                             + stu_label_df["school"] + " · G"
                             + stu_label_df["grade_level"].astype(str))

    options = stu_label_df["label"].tolist()
    default_picks = options[:3] if len(options) >= 3 else options

    picks = st.multiselect("Select students to compare (up to 5)", options,
                           default=default_picks, max_selections=5)

    if len(picks) < 2:
        st.info("👆 Pick at least 2 students to see the comparison charts.")
        st.stop()

    picked_df = stu_label_df[stu_label_df["label"].isin(picks)].copy()
    picked_ids = picked_df["student_id"].tolist()
    picked_names = picked_df["full_name"].tolist()

    # Snapshot cards
    section("Snapshot")
    cols = st.columns(len(picked_df))
    for col, (_, s) in zip(cols, picked_df.iterrows()):
        col.markdown(f"**{s['full_name']}**")
        col.metric("GPA", round(s["current_gpa"], 2))
        col.metric("Attendance %", round(s["cumulative_attendance_pct"], 1))
        col.metric("Risk", s["academic_risk_flag"])
        col.caption(f"{s['school']} · Grade {s['grade_level']} · {s['gender']}")

    section("Head-to-head comparison")

    # Chart 1: Grouped bar — GPA + Attendance side by side
    comp_df = picked_df[["full_name","current_gpa","cumulative_attendance_pct",
                          "performance_index"]].copy()
    melted = comp_df.melt(id_vars="full_name",
                          value_vars=["current_gpa","cumulative_attendance_pct","performance_index"],
                          var_name="metric", value_name="value")
    rename_metric = {"current_gpa":"GPA (0-4)",
                     "cumulative_attendance_pct":"Attendance %",
                     "performance_index":"Performance Index"}
    melted["metric"] = melted["metric"].map(rename_metric)
    fig = px.bar(melted, x="full_name", y="value", color="metric", barmode="group",
                 text_auto=".2f", title="How do the selected students stack up on key metrics?",
                 color_discrete_sequence=theme["palette"])
    fig.update_layout(xaxis_title="Student", yaxis_title="Value (note: different scales)")
    fig.update_traces(textfont=dict(color="white", size=12))
    st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
    caption("<b>What:</b> GPA, attendance %, and performance index for each selected student. "
            "<b>Look for:</b> a student strong on one axis but weak on another — e.g., high GPA + "
            "low attendance = grade inflation or smart-but-truant. "
            "<b>Use case:</b> counsellor parent-meeting prep — show where each child leads or lags peers.")

    # Chart 2: Radar — normalized profile (0-1)
    section("Profile radar")
    metric_cols = ["current_gpa","cumulative_attendance_pct","performance_index",
                   "grade_level"]
    # Normalise against the full student population so each student's position is meaningful
    norm_ref = stu[metric_cols]
    normed = picked_df[metric_cols].copy()
    for c in metric_cols:
        lo, hi = norm_ref[c].min(), norm_ref[c].max()
        normed[c] = (picked_df[c] - lo)/(hi - lo + 1e-9)
    axis_labels = ["GPA","Attendance","Perf. Index","Grade"]
    fig = go.Figure()
    for i, (_, s) in enumerate(picked_df.iterrows()):
        fig.add_trace(go.Scatterpolar(
            r=normed.iloc[i].tolist(), theta=axis_labels,
            fill="toself", name=s["full_name"],
            line_color=theme["palette"][i % len(theme["palette"])]))
    fig.update_layout(
        title="Who has the strongest all-round profile (0-1 scale vs cohort)?",
        polar=dict(bgcolor="rgba(248,250,252,0.5)",
                   radialaxis=dict(visible=True, range=[0, 1],
                                   tickfont=dict(color="#334155", size=11)),
                   angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
        paper_bgcolor="rgba(0,0,0,0)", height=440, title_x=0.02,
        font_family='"Inter", sans-serif', font_color="#0F172A",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(color="#0F172A")))
    st.plotly_chart(fig, use_container_width=True)
    caption("<b>What:</b> each student's polygon normalised against the full cohort (1.0 = top, 0.0 = bottom). "
            "<b>Look for:</b> the biggest polygon = strongest all-rounder; spiky polygons = one-dimensional. "
            "<b>Use case:</b> scholarship committees pick well-rounded polygons over spiky high-GPA-only profiles.")

    # Chart 3: Subject-level comparison
    section("Subject-level breakdown")
    sub_rec = rec[rec["student_id"].isin(picked_ids)].copy()
    if not sub_rec.empty:
        sub_rec = sub_rec.merge(picked_df[["student_id","full_name"]],
                                 on="student_id", how="left")
        subj_pivot = (sub_rec.groupby(["full_name","subject_name"])["percentage"]
                              .mean().reset_index())
        fig = px.bar(subj_pivot, x="subject_name", y="percentage",
                     color="full_name", barmode="group", text_auto=".0f",
                     title="Who excels at what subject? (exam % per subject)",
                     color_discrete_sequence=theme["palette"])
        fig.update_layout(xaxis_title="Subject", yaxis_title="Exam percentage")
        fig.update_traces(textfont=dict(color="white", size=11))
        st.plotly_chart(style_fig(fig, theme, height=420), use_container_width=True)
        caption("<b>What:</b> side-by-side exam scores per subject for each selected student. "
                "<b>Look for:</b> complementary strengths — one strong in Math, another in English — "
                "can feed peer-tutoring pairs. "
                "<b>Use case:</b> counsellor recommendation — pair students with opposite profiles "
                "for mutual subject tutoring.")

    # Chart 4: Fees comparison
    section("Financial snapshot")
    col1, col2 = st.columns(2)
    fee_df = picked_df[["full_name","tuition_fee","fee_paid","fee_outstanding","payment_status"]].copy()

    fig = px.bar(fee_df.melt(id_vars="full_name",
                              value_vars=["fee_paid","fee_outstanding"],
                              var_name="bucket", value_name="amount"),
                 x="full_name", y="amount", color="bucket",
                 title="Who owes how much? (tuition paid vs outstanding)",
                 color_discrete_map={"fee_paid":"#10B981","fee_outstanding":"#EF4444"},
                 barmode="stack", text_auto=".2s")
    fig.update_traces(textfont=dict(color="white", size=11))
    fig.update_layout(xaxis_title="Student", yaxis_title="Amount (₹)")
    col1.plotly_chart(style_fig(fig, theme, height=380), use_container_width=True)
    with col1: caption("<b>What:</b> paid (green) vs outstanding (red) tuition per student. "
                        "<b>Look for:</b> large red segments — these are the escalation cases. "
                        "<b>Use case:</b> fee-collection priority list for the accounts team.")

    # Payment status table
    with col2:
        col2.markdown("**Payment status & amounts**")
        display_fees = fee_df.copy()
        display_fees["Tuition"]     = display_fees["tuition_fee"].map(lambda v: f"₹{v:,.0f}")
        display_fees["Paid"]        = display_fees["fee_paid"].map(lambda v: f"₹{v:,.0f}")
        display_fees["Outstanding"] = display_fees["fee_outstanding"].map(lambda v: f"₹{v:,.0f}")
        col2.dataframe(display_fees[["full_name","Tuition","Paid","Outstanding","payment_status"]]
                       .rename(columns={"full_name":"Student","payment_status":"Status"}),
                       use_container_width=True, hide_index=True)
        caption("<b>What:</b> exact tuition, paid, outstanding, and status for each selected student. "
                "<b>Look for:</b> students in 'Defaulted' or with >50% outstanding. "
                "<b>Use case:</b> fee-collection workflow: initiate scholarship conversion for academic "
                "outperformers who are defaulting financially.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: SCHOOL BENCHMARKING  (NEW — cyan/teal theme)
# ═══════════════════════════════════════════════════════════════════
elif page == "School Benchmarking":

    # Check what fee columns are actually available — the page adapts gracefully
    has_fees = all(c in fstu.columns for c in ["tuition_fee","fee_paid","fee_outstanding","payment_status"])
    if not has_fees:
        st.warning(
            "💡 **Financial data not found in this workbook.** "
            "Academic comparisons below will still work, but the financial sections "
            "(outstanding fees, collection rate, defaulters) require columns "
            "`tuition_fee`, `fee_paid`, `fee_outstanding`, and `payment_status` in "
            "the Students sheet. The file `academic_realistic.xlsx` provided with "
            "this dashboard already contains these fields."
        )

    # Build aggregation — include financial columns only if present
    agg_args = dict(
        students      = ("student_id","count"),
        avg_gpa       = ("current_gpa","mean"),
        avg_att       = ("cumulative_attendance_pct","mean"),
        high_risk_pct = ("academic_risk_flag", lambda s: (s=="High").mean()*100),
    )
    if has_fees:
        agg_args.update(
            tuition_total = ("tuition_fee","sum"),
            paid_total    = ("fee_paid","sum"),
            outstanding   = ("fee_outstanding","sum"),
            defaulted_pct = ("payment_status", lambda s: (s=="Defaulted").mean()*100),
        )
    finance = fstu.groupby("school").agg(**agg_args).reset_index()

    if has_fees:
        finance["collection_rate"] = finance["paid_total"]/finance["tuition_total"]*100
        finance["outstanding_pct"] = finance["outstanding"]/finance["tuition_total"]*100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Schools in view", len(finance))
    if has_fees:
        c2.metric("Total Tuition Billed",  f"₹{finance['tuition_total'].sum()/1e6:.1f}M")
        c3.metric("Total Outstanding",     f"₹{finance['outstanding'].sum()/1e6:.1f}M")
        c4.metric("Overall Collection",    f"{finance['paid_total'].sum()/finance['tuition_total'].sum()*100:.1f}%")
    else:
        c2.metric("Avg GPA (network)",     round(finance['avg_gpa'].mean(), 2))
        c3.metric("Avg Attendance %",      round(finance['avg_att'].mean(), 1))
        c4.metric("Avg High-risk %",       f"{finance['high_risk_pct'].mean():.1f}%")

    # Tabs adapt based on available data
    if has_fees:
        tab1, tab2, tab3 = st.tabs([
            "📚 Academic Comparison", "💰 Financial Comparison", "🔗 Joint View"])
    else:
        tab1, = st.tabs(["📚 Academic Comparison"])

    # ────────────── Academic comparison ──────────────────────
    with tab1:
        section("Head-to-head academic metrics")

        # Let user pick subset to compare, default all
        all_schools = sorted(finance["school"].tolist())
        picks = st.multiselect("Schools to compare (default: all)",
                               all_schools, default=all_schools[:8])
        if not picks:
            picks = all_schools
        sub_fin = finance[finance["school"].isin(picks)]

        col1, col2 = st.columns(2)
        fig = px.bar(sub_fin.sort_values("avg_gpa"), x="avg_gpa", y="school",
                     orientation="h", text_auto=".2f",
                     title="How do the selected schools rank on GPA?",
                     color="avg_gpa", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Average GPA", yaxis_title="School")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col1: caption("<b>What:</b> average GPA per school, sorted low-to-high. "
                           "<b>Look for:</b> the gap between peer schools — similar-sized schools should perform similarly. "
                           "<b>Use case:</b> peer-group benchmarking — compare against similar-sized schools in the network.")

        fig = px.bar(sub_fin.sort_values("avg_att"), x="avg_att", y="school",
                     orientation="h", text_auto=".1f",
                     title="How do the selected schools rank on attendance?",
                     color="avg_att", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Average attendance %", yaxis_title="School")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col2: caption("<b>What:</b> average attendance per school. "
                           "<b>Look for:</b> cross-reference with the GPA chart — a school low on both axes needs "
                           "full intervention; low on one means a specific fix. "
                           "<b>Use case:</b> diagnose whether a school's problem is engagement, academics, or both.")

        fig = px.bar(sub_fin.sort_values("high_risk_pct", ascending=False),
                     x="school", y="high_risk_pct", text_auto=".1f",
                     title="Which schools have the highest concentration of at-risk students?",
                     color="high_risk_pct",
                     color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="% students flagged High Risk")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=380),
                        use_container_width=True)
        caption("<b>What:</b> high-risk student % per school, sorted high-to-low. "
                "<b>Look for:</b> the top 3 schools — urgent intervention candidates. "
                "<b>Use case:</b> quarterly principal performance reviews — accountability for high-risk reduction.")

        # Normalised multi-metric radar
        section("Normalised multi-metric comparison")
        radar_metrics = ["avg_gpa","avg_att","high_risk_pct"]
        radar_labels  = ["GPA","Attendance","Low Risk"]
        rd = sub_fin[["school"] + radar_metrics].copy()
        # For risk we want lower = better, so invert
        rd["high_risk_pct"] = 100 - rd["high_risk_pct"]
        # Normalise each column 0-1
        for c in radar_metrics:
            lo, hi = rd[c].min(), rd[c].max()
            rd[c] = (rd[c] - lo)/(hi - lo + 1e-9)

        fig = go.Figure()
        for i, (_, row) in enumerate(rd.iterrows()):
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics], theta=radar_labels,
                fill="toself", name=row["school"],
                line_color=theme["palette"][i % len(theme["palette"])]))
        fig.update_layout(
            title="School Profiles (higher = better on all axes)",
            polar=dict(bgcolor="rgba(248,250,252,0.5)",
                      radialaxis=dict(visible=True, range=[0, 1],
                                      tickfont=dict(color="#334155", size=11)),
                      angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
            paper_bgcolor="rgba(0,0,0,0)", height=500, title_x=0.02,
            font_family='"Inter", sans-serif', font_color="#0F172A",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                        font=dict(color="#0F172A", size=10)))
        st.plotly_chart(fig, use_container_width=True)
        caption("<b>What:</b> multi-axis radar of GPA, attendance, (inverted) risk — bigger = better. "
                "<b>Look for:</b> lopsided polygons reveal specific weaknesses, round polygons = balanced. "
                "<b>Use case:</b> strategic planning — assign specific interventions to each school's weakest axis.")

    # ────────────── Financial comparison (only if fees available) ─────
    if has_fees:
      with tab2:
        section("Outstanding fees across schools")

        fig = px.bar(finance.sort_values("outstanding_pct", ascending=False),
                     x="school", y="outstanding_pct", text_auto=".1f",
                     title="Which schools have the worst fee-collection problem?",
                     color="outstanding_pct",
                     color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="% of total tuition outstanding")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("<b>What:</b> outstanding tuition as a % of total billed, per school. "
                "<b>Look for:</b> red bars (>20%) indicate a cash flow crisis. "
                "<b>Use case:</b> CFO monthly dashboard — escalate any school above the network average.")

        col1, col2 = st.columns(2)
        fig = px.bar(finance.sort_values("collection_rate"),
                     x="collection_rate", y="school", orientation="h",
                     text_auto=".1f", title="Which schools are collecting tuition efficiently?",
                     color="collection_rate", color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="Collection rate %", yaxis_title="School")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col1: caption("<b>What:</b> collection rate (paid ÷ billed) per school, sorted low-to-high. "
                           "<b>Look for:</b> schools below 75% need intervention — below 60% is critical. "
                           "<b>Use case:</b> escalate bottom-quintile schools to regional finance managers.")

        fig = px.bar(finance.sort_values("defaulted_pct", ascending=False),
                     x="school", y="defaulted_pct", text_auto=".1f",
                     title="Which schools have the most payment defaulters?",
                     color="defaulted_pct",
                     color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]])
        fig.update_traces(textfont=dict(color="white", size=12))
        fig.update_layout(xaxis_title="School", yaxis_title="% defaulted students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                          use_container_width=True)
        with col2: caption("<b>What:</b> % of students in 'Defaulted' status (paid <40% of tuition) per school. "
                           "<b>Look for:</b> concentrations of economic distress — often ties to regional conditions. "
                           "<b>Use case:</b> expand flexible-payment schemes at high-default schools before "
                           "launching aggressive collection.")

        section("Tuition billed vs collected")
        melted_fin = finance.melt(id_vars="school",
                                   value_vars=["paid_total","outstanding"],
                                   var_name="bucket", value_name="amount")
        melted_fin["bucket"] = melted_fin["bucket"].map(
            {"paid_total":"Collected","outstanding":"Outstanding"})
        fig = px.bar(melted_fin, x="school", y="amount", color="bucket",
                     barmode="stack", title="Tuition snapshot: collected (green) vs still owed (red) per school",
                     color_discrete_map={"Collected":"#10B981","Outstanding":"#EF4444"})
        fig.update_layout(xaxis_title="School", yaxis_title="Amount (₹)")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("<b>What:</b> stacked bars of collected (green) + outstanding (red) per school. "
                "<b>Look for:</b> tall red caps on otherwise healthy bars — schools with big absolute "
                "outstanding even if the percentage looks OK. "
                "<b>Use case:</b> absolute-rupee prioritisation — focus collection on the biggest red segments.")

        # Payment status stacked
        section("Payment status mix per school")
        status_mix = (fstu.groupby(["school","payment_status"]).size().reset_index(name="count"))
        status_mix["pct"] = status_mix.groupby("school")["count"].transform(
            lambda x: x/x.sum()*100)
        fig = px.bar(status_mix, x="school", y="pct", color="payment_status",
                     barmode="stack", title="How is each school's payment problem distributed across students?",
                     category_orders={"payment_status":
                                      ["Paid in full","Minor dues","Partial","Defaulted"]},
                     color_discrete_map={"Paid in full":"#10B981","Minor dues":"#84CC16",
                                         "Partial":"#F59E0B","Defaulted":"#EF4444"})
        fig.update_layout(xaxis_title="School", yaxis_title="Percentage of students")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("<b>What:</b> each school normalised to 100% across 4 payment status buckets. "
                "<b>Look for:</b> is the payment problem concentrated (few defaulters, most paid-in-full) "
                "or spread (many partial-payers)? The strategy differs. "
                "<b>Use case:</b> choose between 'hardcore collection on 10% defaulters' vs 'parent-wide "
                "payment-plan campaign' for widespread partial payment.")

    # ────────────── Joint view (only if fees available) ──────────────
    if has_fees:
      with tab3:
        section("Does financial health correlate with academic outcomes?")

        fig = px.scatter(finance, x="outstanding_pct", y="avg_gpa", size="students",
                         hover_name="school", title="Do financially distressed schools also underperform academically?",
                         color="high_risk_pct",
                         color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                         labels={"high_risk_pct":"High-risk %"})
        fig = add_trendline(fig, finance["outstanding_pct"], finance["avg_gpa"],
                            color="#0F172A")
        fig.update_layout(xaxis_title="% tuition outstanding",
                          yaxis_title="Average GPA")
        st.plotly_chart(style_fig(fig, theme, height=500), use_container_width=True)
        caption("<b>What:</b> outstanding % vs GPA per school, with enrolment (size) and risk (colour). "
                "<b>Look for:</b> downward trendline = financially distressed schools also underperform. "
                "Also note outliers: high-GPA-despite-poor-collections schools are resilient. "
                "<b>Use case:</b> board-level strategic review — decide whether to cross-subsidise "
                "distressed schools or restructure them.")

        section("Academic-financial composite ranking")
        # Composite score: higher GPA, higher collection rate, lower high_risk_pct
        rank = finance.copy()
        rank["academic_score"]  = (rank["avg_gpa"] - rank["avg_gpa"].min())/(rank["avg_gpa"].max() - rank["avg_gpa"].min() + 1e-9)
        rank["financial_score"] = (rank["collection_rate"] - rank["collection_rate"].min())/(rank["collection_rate"].max() - rank["collection_rate"].min() + 1e-9)
        rank["composite"]       = 0.6*rank["academic_score"] + 0.4*rank["financial_score"]
        rank = rank.sort_values("composite", ascending=False)

        fig = px.bar(rank, x="school", y="composite",
                     title="Overall school health: weighted academic + financial ranking",
                     color="composite", color_continuous_scale=theme["scale"],
                     text_auto=".2f",
                     hover_data=["avg_gpa","collection_rate","outstanding_pct"])
        fig.update_traces(textfont=dict(color="white", size=11))
        fig.update_layout(xaxis_title="School",
                          yaxis_title="Composite score (0-1 scale)")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=420),
                        use_container_width=True)
        caption("<b>What:</b> single 0–1 ranking blending academic (60%) + financial (40%) scores. "
                "<b>Look for:</b> the bottom 3 schools need full-spectrum turnaround; top 3 are mentor models. "
                "<b>Use case:</b> board-of-trustees annual strategy review — rank-order intervention priority.")

        insight("If a school scores high academically but low financially (or vice versa), "
                "dig into the individual charts above to understand the imbalance. "
                "Truly troubled schools are low on <b>both</b> axes.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE: PREDICTIVE LAB  (teal + indigo)
# ═══════════════════════════════════════════════════════════════════
elif page == "Predictive Lab":

    risk_pkg    = train_risk_model(stu)
    gpa_pkg     = train_gpa_model(stu)
    cluster_pkg = build_clusters(stu, k=4)
    anom_pkg    = detect_anomalies(stu)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk model acc (LR)", f"{risk_pkg['lr_acc']*100:.1f}%")
    c2.metric("Risk model acc (RF)", f"{risk_pkg['rf_acc']*100:.1f}%")
    c3.metric("GPA model R²",        f"{gpa_pkg['r2']:.3f}")
    c4.metric("Anomalies flagged",   int((anom_pkg['flags']==-1).sum()))

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Risk Classifier", "🔮 What-If Simulator",
        "🧬 Clustering", "⚠️ Anomalies & Correlation"])

    with tab1:
        section("Model performance")
        col1, col2 = st.columns(2)

        acc_df = pd.DataFrame({
            "Model":    ["Logistic Regression", "Random Forest"],
            "Accuracy": [risk_pkg["lr_acc"]*100, risk_pkg["rf_acc"]*100]})
        fig = px.bar(acc_df, x="Model", y="Accuracy", text_auto=".1f",
                     title="🎯 [1/10] Which ML model predicts risk more accurately?",
                     color="Model", color_discrete_sequence=theme["palette"])
        fig.update_layout(yaxis_range=[0, 100],
                          xaxis_title="ML model", yaxis_title="Accuracy on training data (%)")
        fig.update_traces(textfont=dict(color="white", size=16))
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col1: caption("<b>What:</b> two ML algorithms trained on the same features, compared by accuracy. "
                           "<b>Look for:</b> RF beating LR by >5 points means non-linear patterns exist in the data. "
                           "<b>Use case:</b> model selection — deploy the winning model for weekly risk scoring.")

        X_all = make_risk_X(stu)
        y_true = stu["academic_risk_flag"]
        y_pred = risk_pkg["rf"].predict(X_all)
        classes = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        fig = px.imshow(cm, text_auto=True, x=classes, y=classes,
                        labels=dict(x="Predicted risk class", y="Actual risk class", color="Count"),
                        title="🎯 [2/10] Where does the Random Forest model make mistakes?",
                        color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(size=16, color="#0F172A"))
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=340),
                          use_container_width=True)
        with col2: caption("<b>What:</b> actual vs predicted risk class. Diagonal = correct, off-diagonal = errors. "
                           "<b>Look for:</b> (Actual=High, Predicted=Low) is the most costly error — missed "
                           "high-risk students. "
                           "<b>Use case:</b> accept small over-flagging (false positives) to catch every "
                           "genuine high-risk case.")

        section("Explainability")
        col1, col2 = st.columns(2)

        imp = pd.DataFrame({
            "feature": risk_pkg["features"],
            "importance": risk_pkg["rf"].feature_importances_,
        }).sort_values("importance", ascending=True)
        fig = px.bar(imp, x="importance", y="feature", orientation="h",
                     title="🎯 [3/10] What factors drive the model's risk predictions?",
                     text_auto=".3f", color="importance",
                     color_continuous_scale=theme["scale"])
        fig.update_traces(textfont=dict(color="#0F172A", size=13))
        fig.update_layout(xaxis_title="Importance (0-1)", yaxis_title="Feature")
        col1.plotly_chart(style_fig(fig, theme, show_legend=False, height=360),
                          use_container_width=True)
        with col1: caption("<b>What:</b> which features the Random Forest uses most. "
                           "<b>Look for:</b> if Attendance is on top, that's your operational leading indicator. "
                           "<b>Use case:</b> focus data-collection quality improvements on the top-3 features — "
                           "those drive model accuracy.")

        X_f = make_risk_X(fstu)
        proba = risk_pkg["rf"].predict_proba(X_f)
        high_idx = list(risk_pkg["rf"].classes_).index("High")
        p_high = proba[:, high_idx]
        fig = px.histogram(pd.DataFrame({"p_high": p_high}), x="p_high", nbins=25,
                           title="🎯 [4/10] How confident is the model on current students?",
                           color_discrete_sequence=[theme["primary"]])
        fig.update_traces(marker_line_width=1, marker_line_color="white")
        fig.add_vline(x=0.5, line_dash="dash", line_color="#DC2626", line_width=3,
                      annotation_text="Decision threshold (0.5)",
                      annotation_font_color="#0F172A")
        fig.update_layout(xaxis_title="Predicted probability of High-risk classification",
                          yaxis_title="Number of students")
        col2.plotly_chart(style_fig(fig, theme, show_legend=False, height=360, bargap=0.1),
                          use_container_width=True)
        with col2: caption("<b>What:</b> distribution of P(High risk) predictions across students. "
                           "<b>Look for:</b> students near the 0.5 threshold — borderline cases needing human review. "
                           "<b>Use case:</b> counsellor triage — concentrate reviews on the 0.4–0.6 probability range.")

    with tab2:
        st.markdown("Adjust the sliders to see real-time predictions for a hypothetical "
                    "student and how each feature contributes to the outcome.")

        col_in, col_out = st.columns([1, 2])
        with col_in:
            att_input   = st.slider("Attendance %",  50.0, 100.0, 85.0, 0.5)
            grade_input = st.select_slider("Grade level",
                options=sorted(stu["grade_level"].dropna().unique()), value=6)
            schol_input = st.checkbox("Has scholarship", value=False)
            iep_input   = st.checkbox("Has IEP",         value=False)

        x_gpa = pd.DataFrame(
            [[att_input, grade_input, int(schol_input), int(iep_input)]],
            columns=GPA_FEATURES)
        pred_gpa = float(gpa_pkg["model"].predict(x_gpa)[0])
        pred_gpa = max(0.0, min(4.0, pred_gpa))

        x_risk = pd.DataFrame(
            [[pred_gpa, att_input, grade_input, int(schol_input), int(iep_input)]],
            columns=RISK_FEATURES)
        pred_risk  = risk_pkg["rf"].predict(x_risk)[0]
        pred_proba = risk_pkg["rf"].predict_proba(x_risk)[0]
        cohort_gpa = float(stu["current_gpa"].mean())

        with col_out:
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Predicted GPA",  round(pred_gpa, 2),
                       delta=round(pred_gpa - cohort_gpa, 2))
            cc2.metric("Predicted risk", pred_risk)
            cc3.metric("Confidence",     f"{pred_proba.max()*100:.1f}%")

            prob_df = pd.DataFrame({
                "class": risk_pkg["rf"].classes_, "probability": pred_proba})
            fig = px.bar(prob_df, x="class", y="probability", text_auto=".0%",
                         title="🔮 [5/10] For this hypothetical student, how does risk break down?",
                         color="class",
                         color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444"})
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            fig.update_traces(textfont=dict(color="white", size=16))
            fig.update_layout(xaxis_title="Risk class", yaxis_title="Predicted probability")
            st.plotly_chart(style_fig(fig, theme, show_legend=False, height=300),
                            use_container_width=True)
            caption("<b>What:</b> model's predicted probability for each risk class. "
                    "<b>Look for:</b> when probabilities are close (e.g., 45/40/15), the prediction is uncertain. "
                    "<b>Use case:</b> supplement automated decisions with counsellor judgment when top-2 classes are within 10%.")

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=pred_gpa,
            delta={"reference": cohort_gpa, "valueformat": ".2f",
                   "font": {"color": "#0F172A"}},
            number={"font": {"color": "#0F172A", "size": 48}},
            title={"text": "🔮 [6/10] Is this student predicted above or below the cohort mean?",
                   "font": {"color": "#0F172A", "size": 16}},
            gauge={
                "axis": {"range": [0, 4], "tickfont": {"color": "#0F172A", "size": 12}},
                "bar": {"color": theme["primary"]},
                "bgcolor": "white", "borderwidth": 2, "bordercolor": "#CBD5E1",
                "steps": [
                    {"range": [0, 2],   "color": "#FEE2E2"},
                    {"range": [2, 3],   "color": "#FEF3C7"},
                    {"range": [3, 4],   "color": "#DCFCE7"}],
                "threshold": {"line": {"color": "#0F172A", "width": 4},
                              "thickness": 0.85, "value": cohort_gpa}}))
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=20),
                          paper_bgcolor="rgba(0,0,0,0)",
                          font_family='"Inter", sans-serif')
        st.plotly_chart(fig, use_container_width=True)
        caption("<b>What:</b> predicted GPA on a 0–4 gauge with red/amber/green zones. "
                "<b>Look for:</b> the triangle = cohort mean; is your student above or below? "
                "<b>Use case:</b> instant what-if answer in parent meetings — 'if attendance improves by 10%, GPA rises to X'.")

        section("🔍 SHAP-style contribution breakdown")
        st.markdown("How each feature pushes the GPA prediction **up** or **down** "
                    "from the baseline.")

        inputs = {"Attendance %": att_input, "Grade Level": grade_input,
                  "Scholarship": int(schol_input), "IEP": int(iep_input)}
        means = gpa_pkg["means"]
        coefs = gpa_pkg["coefs"]

        contrib_rows = []
        for f in GPA_FEATURES:
            c = coefs[f] * (inputs[f] - means[f])
            contrib_rows.append({"feature": f, "contribution": c})
        contrib_df = pd.DataFrame(contrib_rows).sort_values("contribution")

        fig = go.Figure(go.Waterfall(
            orientation="h",
            y=contrib_df["feature"].tolist() + ["Total Δ from mean"],
            x=contrib_df["contribution"].tolist() + [contrib_df["contribution"].sum()],
            measure=["relative"]*len(contrib_df) + ["total"],
            connector={"line":{"color":"#CBD5E1"}},
            decreasing={"marker":{"color":"#EF4444"}},
            increasing={"marker":{"color":"#10B981"}},
            totals={"marker":{"color": theme["primary"]}},
            text=[f"{v:+.3f}" for v in contrib_df["contribution"].tolist()] +
                 [f"{contrib_df['contribution'].sum():+.3f}"],
            textposition="outside",
            textfont=dict(size=13, color="#0F172A")))
        fig.update_layout(
            title="🔮 [7/10] Which factors pushed this student's GPA up or down?",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=340, font_family='"Inter", sans-serif', title_x=0.02,
            font_color="#0F172A",
            xaxis=dict(title="GPA points pushed above / below baseline",
                       gridcolor="#E2E8F0", linecolor="#94A3B8",
                       tickfont=dict(color="#0F172A", size=12)),
            yaxis=dict(title="Feature",
                       gridcolor="#E2E8F0", linecolor="#94A3B8",
                       tickfont=dict(color="#0F172A", size=13)))
        st.plotly_chart(fig, use_container_width=True)
        caption("<b>What:</b> feature-by-feature contribution waterfall — green = lifts GPA, red = drags it down. "
                "<b>Look for:</b> the largest green bar = biggest positive lever; largest red = biggest drag. "
                "<b>Use case:</b> explain to parents and counsellors which specific factors matter for THIS student — "
                "actionable, individualised intervention plans.")

    with tab3:
        st.markdown("K-means clustering (k=4) groups students by GPA, attendance, grade "
                    "and scholarship status. Each cluster is a natural cohort.")

        labels = cluster_pkg["labels"]
        centers = cluster_pkg["centers"]
        cluster_stu = stu.assign(cluster=[f"Cluster {l}" for l in labels])

        sample = cluster_stu.sample(min(3000, len(cluster_stu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="cluster", opacity=0.7,
                         title="🧬 [8/10] What natural student segments exist in our cohort?",
                         hover_data=["school","grade_level","academic_risk_flag"],
                         color_discrete_sequence=theme["palette"])
        for i, c in enumerate(centers):
            fig.add_trace(go.Scatter(
                x=[c[1]], y=[c[0]], mode="markers+text",
                marker=dict(size=28, color=theme["palette"][i % len(theme["palette"])],
                            symbol="star", line=dict(color="#0F172A", width=2)),
                text=[f"C{i}"], textposition="top center",
                textfont=dict(size=13, color="#0F172A"),
                name=f"Centre {i}", showlegend=False))
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, height=480), use_container_width=True)
        caption("<b>What:</b> K-means finds 4 natural cohorts without being told; stars mark cluster centres. "
                "<b>Look for:</b> each cluster's archetype — the students nearest each star represent that segment. "
                "<b>Use case:</b> differentiated instruction design — tailor curriculum to each cluster's profile.")

        prof = stu.assign(cluster=labels).groupby("cluster").agg(
            gpa=("current_gpa","mean"),
            attendance=("cumulative_attendance_pct","mean"),
            grade=("grade_level","mean"),
            risk=("academic_risk_flag", lambda s: (s=="High").mean()*100)).reset_index()
        prof_norm = prof.copy()
        for c in ["gpa","attendance","grade","risk"]:
            col = prof_norm[c]
            prof_norm[c] = (col - col.min())/(col.max() - col.min() + 1e-9)

        fig = go.Figure()
        axes = ["gpa","attendance","grade","risk"]
        axis_labels = ["GPA","Attendance %","Grade Level","High-Risk %"]
        for i, row in prof_norm.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[a] for a in axes], theta=axis_labels,
                fill='toself', name=f"Cluster {int(row['cluster'])}",
                line_color=theme["palette"][int(row["cluster"]) % len(theme["palette"])]))
        fig.update_layout(
            title="🧬 [9/10] How do the 4 student segments differ on each metric?",
            polar=dict(bgcolor="rgba(248,250,252,0.5)",
                      radialaxis=dict(visible=True, range=[0, 1],
                                      tickfont=dict(color="#334155", size=11)),
                      angularaxis=dict(tickfont=dict(color="#0F172A", size=13))),
            paper_bgcolor="rgba(0,0,0,0)", height=420, title_x=0.02,
            font_family='"Inter", sans-serif', font_color="#0F172A",
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        font=dict(color="#0F172A")))
        st.plotly_chart(fig, use_container_width=True)
        caption("<b>What:</b> 4 cluster polygons overlaid — each axis normalised 0–1. "
                "<b>Look for:</b> the cluster low on GPA/attendance but high on risk — your at-risk segment. "
                "<b>Use case:</b> cohort-based intervention design — the at-risk cluster needs intensive support, "
                "the high-GPA cluster needs enrichment.")

        st.markdown("**Cluster summary:**")
        st.dataframe(prof.round(2), use_container_width=True, hide_index=True)

    with tab4:
        section("Feature correlation matrix")
        num_df = pd.DataFrame({
            "GPA":         stu["current_gpa"],
            "Attendance":  stu["cumulative_attendance_pct"],
            "Grade":       stu["grade_level"],
            "Performance": stu["performance_index"],
            "Age":         stu["age"],
            "Scholarship": (stu["scholarship_flag"]=="Yes").astype(int),
            "IEP":         (stu["iep_flag"]=="Yes").astype(int),
            "High Risk":   (stu["academic_risk_flag"]=="High").astype(int)})
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f",
                        title="⚠️ [10/10] Which student attributes move together? (correlation matrix)",
                        color_continuous_scale=[[0,"#EF4444"],[0.5,"#FFFFFF"],[1,"#10B981"]],
                        zmin=-1, zmax=1, aspect="auto")
        fig.update_traces(textfont=dict(size=13, color="#0F172A"))
        fig.update_layout(xaxis_title="Feature", yaxis_title="Feature")
        st.plotly_chart(style_fig(fig, theme, show_legend=False, height=500),
                        use_container_width=True)
        caption("<b>What:</b> correlation between every pair of student attributes. "
                "<b>Look for:</b> intense green off-diagonal cells — those pairs move together strongly. "
                "Pay attention to what correlates with 'High Risk' — those are causes worth intervening on. "
                "<b>Use case:</b> hypothesis generation for intervention studies — 'if attendance is the "
                "strongest correlate of GPA, then improving attendance is our top lever'.")

        section("🚨 Isolation-forest anomalies")
        sample = stu.assign(
            anomaly_label=["Anomaly" if f==-1 else "Normal" for f in anom_pkg["flags"]],
            anom_score=anom_pkg["scores"]).sample(min(3000, len(stu)), random_state=1)
        fig = px.scatter(sample, x="cumulative_attendance_pct", y="current_gpa",
                         color="anomaly_label", opacity=0.7,
                         title="Isolation-Forest · Students with Unusual Profiles",
                         color_discrete_map={"Normal":"#94A3B8", "Anomaly":"#EF4444"},
                         hover_data=["school","grade_level","academic_risk_flag"])
        fig.update_layout(xaxis_title="Attendance %", yaxis_title="GPA")
        st.plotly_chart(style_fig(fig, theme, height=440), use_container_width=True)
        caption("<b>What:</b> isolation-forest flags students whose profile is statistically unusual. "
                "<b>Look for:</b> red dots in unexpected places — a high-attendance low-GPA student may be "
                "undiagnosed with a learning difficulty; low-attendance high-GPA may indicate health issues. "
                "<b>Use case:</b> counsellor deep-dive list — these 5% need individualised conversations "
                "more than the 95% average student.")

        anom_count = int((anom_pkg['flags']==-1).sum())
        insight(f"The isolation-forest flagged <b>{anom_count} students</b> (≈5%) "
                f"whose GPA / attendance / grade combination is statistically unusual "
                f"compared with peers. Good candidates for early counsellor review.")


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR FOOTER
# ═══════════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.download_button(
    "📥 Download filtered students",
    fstu.to_csv(index=False).encode("utf-8"),
    file_name="filtered_students.csv",
    mime="text/csv",
    use_container_width=True,
)
st.sidebar.markdown(
    "<div style='margin-top:1rem; font-size:.7rem; color:#94A3B8; text-align:center;'>"
    "v3.0 · 10 pages · 90+ charts · Streamlit · Plotly · scikit-learn"
    "</div>", unsafe_allow_html=True)
