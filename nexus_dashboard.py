"""
Nexus Analytics – Multi-School Academic Dashboard
Each page has its own colour identity; shared data layer reads every sheet.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Nexus Analytics", page_icon="📊", layout="wide",initial_sidebar_state="expanded")


# ============================================================
# PER-PAGE COLOUR THEMES
# ============================================================
THEMES = {
    "Overview": {
        "icon": "📊",
        "subtitle": "A snapshot across schools, students and performance",
        "primary": "#4F46E5",
        "grad": ("#6366F1", "#8B5CF6"),
        "palette": ["#4F46E5", "#7C3AED", "#EC4899", "#F59E0B", "#10B981", "#06B6D4"],
    },
    "Students": {
        "icon": "🎓",
        "subtitle": "Academic standing, attendance and risk profile",
        "primary": "#059669",
        "grad": ("#10B981", "#0D9488"),
        "palette": ["#059669", "#0D9488", "#10B981", "#34D399", "#6EE7B7", "#065F46"],
    },
    "Academic Records": {
        "icon": "📚",
        "subtitle": "Exam outcomes, subjects and grade distribution",
        "primary": "#D97706",
        "grad": ("#F59E0B", "#EA580C"),
        "palette": ["#D97706", "#EA580C", "#F59E0B", "#FB923C", "#FBBF24", "#92400E"],
    },
    "Teachers": {
        "icon": "👩‍🏫",
        "subtitle": "Faculty distribution, ratings and experience",
        "primary": "#BE123C",
        "grad": ("#E11D48", "#DB2777"),
        "palette": ["#BE123C", "#DB2777", "#E11D48", "#F43F5E", "#FB7185", "#881337"],
    },
    "Attendance": {
        "icon": "📅",
        "subtitle": "Presence patterns by school and grade",
        "primary": "#0284C7",
        "grad": ("#0EA5E9", "#0891B2"),
        "palette": ["#0284C7", "#0891B2", "#0EA5E9", "#06B6D4", "#38BDF8", "#075985"],
    },
    "Schools": {
        "icon": "🏫",
        "subtitle": "Institutions, boards and geography",
        "primary": "#7C3AED",
        "grad": ("#8B5CF6", "#A855F7"),
        "palette": ["#7C3AED", "#A855F7", "#8B5CF6", "#A78BFA", "#C4B5FD", "#4C1D95"],
    },
}


# ============================================================
# GLOBAL CSS
# ============================================================
BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1400px;
}

[data-testid="stMetric"] {
    background: #ffffff;
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06),
                0 1px 2px rgba(15, 23, 42, 0.04);
    border-left: 4px solid var(--accent, #4F46E5);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

[data-testid="stPlotlyChart"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 0.75rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    margin-bottom: 1rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span 
}
/* ===== RADIO BUTTON LABELS (ALL OPTIONS) ===== */
[data-testid="stSidebar"] .stRadio label {
    color: #CBD5F5 !important;   /* light blue-gray */
    font-weight: 500 !important;
    opacity: 0.9;
}

/* ===== SELECTED OPTION (HIGHLIGHT) ===== */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    background: rgba(59, 130, 246, 0.15);  /* soft blue highlight */
    border-radius: 8px;
    padding: 6px 10px;
}

/* ===== RADIO DOT (SELECTED) ===== */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"]::before {
    background-color: #3B82F6 !important;  /* blue dot */
    border-color: #3B82F6 !important;
}
/* FIX RADIO BUTTON TEXT (Navigation) */
[data-testid="stSidebar"] .stRadio label {
    color: #E2E8F0 !important;
    font-weight: 500 !important;
}
/* ===== FILE UPLOADER BUTTON ===== */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #3B82F6, #6366F1) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
}

/* ===== DOWNLOAD BUTTON ===== */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #3B82F6, #6366F1) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.12) !important;
}

footer, header {visibility: hidden;}

[data-testid="column"] {
    padding: 0 0.5rem !important;
}

.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.6rem 0;
}
</style>
"""
st.markdown(BASE_CSS, unsafe_allow_html=True)


def apply_theme(theme):
    st.markdown(
        f"<style>:root {{ --accent: {theme['primary']}; }} "
        f"[data-testid='stMetric'] {{ border-left-color: {theme['primary']}; }}</style>",
        unsafe_allow_html=True,
    )


def render_header(page: str):
    t = THEMES[page]
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {t['grad'][0]} 0%, {t['grad'][1]} 100%);
            padding: 2rem 2.4rem;
            border-radius: 18px;
            margin-bottom: 1.8rem;
            color: white;
            box-shadow: 0 10px 30px -10px {t['primary']}55;
        ">
            <div style="font-size:0.78rem; opacity:0.85; letter-spacing:0.14em;
                        text-transform:uppercase; margin-bottom:0.4rem;">
                Nexus Analytics
            </div>
            <div style="font-size:2rem; font-weight:700; line-height:1.1;">
                {t['icon']}  {page}
            </div>
            <div style="font-size:1rem; opacity:0.92; margin-top:0.4rem;">
                {t['subtitle']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(label: str):
    st.markdown(f"<div class='section-label'>{label}</div>", unsafe_allow_html=True)


def style_fig(fig, theme, *, show_legend=True):
    fig.update_layout(
        font_family='"Inter", system-ui, sans-serif',
        font_color="#1E293B",
        title_font_color="#0F172A",
        title_font_size=14,
        title_x=0.02,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=20, t=50, b=30),
        colorway=theme["palette"],
        showlegend=show_legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        hoverlabel=dict(
            bgcolor="white", font_size=12,
            font_family='"Inter", system-ui, sans-serif',
            bordercolor=theme["primary"],
        ),
    )
    fig.update_xaxes(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", linecolor="#E2E8F0")
    return fig


# ============================================================
# DATA LAYER
# ============================================================
@st.cache_data(show_spinner="Loading workbook…")
def load_workbook(src) -> dict:
    sheets = pd.read_excel(src, sheet_name=None)
    for name, frame in sheets.items():
        frame.columns = (frame.columns.astype(str)
                         .str.strip().str.lower().str.replace(" ", "_"))
        sheets[name] = frame
    return sheets


@st.cache_data(show_spinner="Building views…")
def build_views(sheets: dict) -> dict:
    schools  = sheets["Schools"].rename(columns={"school_name": "school"})
    students = sheets["Students"].copy()
    records  = sheets["Student_Academic_Records"].copy()
    attend   = sheets["Attendance_Log"].copy()
    teachers = sheets["Teachers"].copy()

    stu = students.merge(
        schools[["school_id","school","board_name","region","state","city","school_type"]],
        on="school_id", how="left"
    )
    stu["performance_index"] = (
        stu["current_gpa"].fillna(0) * 25
        + stu["cumulative_attendance_pct"].fillna(0) * 0.5
    )

    rec = (records
           .merge(students[["student_id","gender"]], on="student_id", how="left")
           .merge(schools[["school_id","school","board_name","region"]],
                  on="school_id", how="left"))

    att = attend.merge(schools[["school_id","school","region"]],
                       on="school_id", how="left")
    att["attendance_date"] = pd.to_datetime(att["attendance_date"], errors="coerce")

    return {"students": stu, "records": rec, "attendance": att,
            "teachers": teachers, "schools": schools}


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    """
    <div style='padding: 0.5rem 0 1.5rem 0;'>
        <div style='font-size:1.5rem; font-weight:700; color:white;'>
            📊 Nexus Analytics
        </div>
        <div style='font-size:0.75rem; color:#94A3B8; margin-top:0.25rem;'>
            Multi-School Academic Dashboard
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.sidebar.file_uploader("Workbook (.xlsx)", type="xlsx")
default_path = Path("academic_multi_school_dashboard_populated_10000.xlsx")
source = uploaded if uploaded is not None else (default_path if default_path.exists() else None)

if source is None:
    st.title("Nexus Analytics")
    st.info("Upload the academic workbook in the sidebar to begin.")
    st.stop()

sheets = load_workbook(source)
views  = build_views(sheets)
stu, rec, att = views["students"], views["records"], views["attendance"]
teachers, schools = views["teachers"], views["schools"]

st.sidebar.markdown("---")

f_school = st.sidebar.multiselect("School", sorted(stu["school"].dropna().unique()))
f_grade  = st.sidebar.multiselect("Grade",  sorted(stu["grade_level"].dropna().unique()))
f_risk   = st.sidebar.multiselect("Academic risk",
                                  sorted(stu["academic_risk_flag"].dropna().unique()))

def filter_students(df):
    if f_school: df = df[df["school"].isin(f_school)]
    if f_grade:  df = df[df["grade_level"].isin(f_grade)]
    if f_risk:   df = df[df["academic_risk_flag"].isin(f_risk)]
    return df

fstu = filter_students(stu)
ids = set(fstu["student_id"])
frec = rec[rec["student_id"].isin(ids)] if ids else rec.iloc[0:0]
fatt = att[att["stakeholder_id"].isin(ids)] if ids else att.iloc[0:0]

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", list(THEMES.keys()))

if fstu.empty:
    st.error("No rows match the current filters. Clear some filters in the sidebar.")
    st.stop()

theme = THEMES[page]
apply_theme(theme)
render_header(page)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    section("Key figures")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Schools",  stu["school"].nunique())
    c2.metric("Students", f"{len(fstu):,}")
    c3.metric("Teachers", f"{len(teachers):,}")
    c4.metric("Avg GPA",  round(fstu["current_gpa"].mean(), 2))
    c5.metric("Attendance %", round(fstu["cumulative_attendance_pct"].mean(), 1))

    section("Distributions")
    col1, col2 = st.columns(2)
    fig = px.histogram(fstu, x="current_gpa", nbins=30, title="GPA distribution",
                       color_discrete_sequence=[theme["primary"]])
    col1.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    fig = px.histogram(fstu, x="cumulative_attendance_pct", nbins=30,
                       title="Attendance distribution",
                       color_discrete_sequence=[theme["palette"][1]])
    col2.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    section("Composition")
    col1, col2 = st.columns(2)
    fig = px.pie(fstu, names="academic_risk_flag", title="Risk composition", hole=0.55)
    fig.update_traces(textposition="outside", textinfo="label+percent")
    col1.plotly_chart(style_fig(fig, theme), use_container_width=True)

    g = fstu["gender"].value_counts().reset_index()
    g.columns = ["gender", "count"]
    fig = px.bar(g, x="gender", y="count", title="Gender distribution",
                 color="gender", color_discrete_sequence=theme["palette"])
    col2.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    section("Schools at a glance")
    by_school = (fstu.groupby("school")
                     .agg(students=("student_id","count"),
                          avg_gpa=("current_gpa","mean"),
                          avg_attendance=("cumulative_attendance_pct","mean"))
                     .reset_index().sort_values("students", ascending=False))
    fig = px.bar(by_school, x="school", y="students",
                 hover_data=["avg_gpa","avg_attendance"],
                 title="Students per school",
                 color_discrete_sequence=[theme["primary"]])
    st.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)


# ============================================================
# STUDENTS
# ============================================================
elif page == "Students":
    section("Cohort")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("In view",      f"{len(fstu):,}")
    c2.metric("High risk",    int((fstu["academic_risk_flag"] == "High").sum()))
    c3.metric("Scholarship",  int((fstu["scholarship_flag"] == "Yes").sum()))
    c4.metric("Avg GPA",      round(fstu["current_gpa"].mean(), 2))

    section("By grade level")
    col1, col2 = st.columns(2)
    fig = px.box(fstu, x="grade_level", y="current_gpa", title="GPA by grade",
                 color_discrete_sequence=[theme["primary"]])
    col1.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    fig = px.box(fstu, x="grade_level", y="cumulative_attendance_pct",
                 title="Attendance by grade",
                 color_discrete_sequence=[theme["palette"][1]])
    col2.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    section("Relationship between attendance and GPA")
    fig = px.scatter(fstu, x="cumulative_attendance_pct", y="current_gpa",
                     color="academic_risk_flag",
                     hover_data=["school","grade_level"],
                     title="GPA vs attendance — coloured by risk",
                     color_discrete_sequence=theme["palette"], opacity=0.7)
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)

    fig = px.density_heatmap(fstu, x="cumulative_attendance_pct", y="current_gpa",
                             title="Density — attendance vs GPA",
                             color_continuous_scale=[[0, "#F8FAFC"], [1, theme["primary"]]])
    st.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)


# ============================================================
# ACADEMIC RECORDS
# ============================================================
elif page == "Academic Records":
    section("Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Exam records",  f"{len(frec):,}")
    c2.metric("Pass rate",     f"{(frec['pass_fail']=='Pass').mean()*100:.1f}%")
    c3.metric("Avg %",         round(frec["percentage"].mean(), 1))
    c4.metric("Subjects",      frec["subject_name"].nunique())

    section("By subject and grade")
    col1, col2 = st.columns(2)
    fig = px.box(frec, x="subject_name", y="percentage",
                 color="subject_name", title="Score by subject",
                 color_discrete_sequence=theme["palette"])
    col1.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    order = sorted(frec["grade_awarded"].dropna().unique())
    fig = px.histogram(frec, x="grade_awarded",
                       category_orders={"grade_awarded": order},
                       title="Grade distribution",
                       color_discrete_sequence=[theme["primary"]])
    col2.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    section("Subject performance by term")
    term_subject = (frec.groupby(["term_name","subject_name"])["percentage"]
                        .mean().reset_index())
    fig = px.bar(term_subject, x="subject_name", y="percentage",
                 color="term_name", barmode="group",
                 title="Average % by subject and term",
                 color_discrete_sequence=theme["palette"])
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)

    section("Assignments vs exam marks")
    fig = px.scatter(frec, x="assignment_score", y="marks_obtained",
                     color="subject_name",
                     title="Assignment score vs exam marks",
                     color_discrete_sequence=theme["palette"], opacity=0.6)
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)


# ============================================================
# TEACHERS
# ============================================================
elif page == "Teachers":
    section("Faculty")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teachers",       len(teachers))
    c2.metric("Avg rating",     round(teachers["teacher_performance_rating"].mean(), 2))
    c3.metric("Avg attendance", f"{teachers['teacher_attendance_pct'].mean():.1f}%")
    c4.metric("Departments",    teachers["department"].nunique())

    section("By department")
    col1, col2 = st.columns(2)
    dept = teachers["department"].value_counts().reset_index()
    dept.columns = ["department", "count"]
    fig = px.bar(dept, x="department", y="count",
                 color="department", title="Teachers by department",
                 color_discrete_sequence=theme["palette"])
    col1.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    fig = px.box(teachers, x="department", y="teacher_performance_rating",
                 color="department", title="Rating by department",
                 color_discrete_sequence=theme["palette"])
    col2.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    section("Experience and rating")
    fig = px.scatter(teachers, x="years_experience",
                     y="teacher_performance_rating",
                     color="department",
                     hover_data=["subject_specialization","teacher_attendance_pct"],
                     title="Experience vs performance rating",
                     color_discrete_sequence=theme["palette"], opacity=0.75)
    st.plotly_chart(style_fig(fig, theme), use_container_width=True)


# ============================================================
# ATTENDANCE
# ============================================================
elif page == "Attendance":
    section("Log summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Log entries", f"{len(fatt):,}")
    pres = (fatt["attendance_status"] == "Present").mean()*100 if len(fatt) else 0
    c2.metric("Present %",   f"{pres:.1f}%")
    c3.metric("Dates",       fatt["attendance_date"].nunique())

    section("Breakdown")
    col1, col2 = st.columns(2)
    status = fatt["attendance_status"].value_counts().reset_index()
    status.columns = ["status", "count"]
    fig = px.pie(status, names="status", values="count",
                 title="Attendance status split", hole=0.55,
                 color_discrete_sequence=theme["palette"])
    fig.update_traces(textposition="outside", textinfo="label+percent")
    col1.plotly_chart(style_fig(fig, theme), use_container_width=True)

    by_grade = (fatt.assign(present=(fatt["attendance_status"] == "Present").astype(int))
                     .groupby("grade_level")["present"].mean().reset_index())
    by_grade["present"] *= 100
    fig = px.bar(by_grade, x="grade_level", y="present",
                 title="Attendance % by grade",
                 labels={"present":"present %"},
                 color_discrete_sequence=[theme["primary"]])
    col2.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    section("School-level comparison")
    by_school = (fatt.assign(present=(fatt["attendance_status"] == "Present").astype(int))
                      .groupby("school")["present"].mean().reset_index()
                      .sort_values("present", ascending=False))
    by_school["present"] *= 100
    fig = px.bar(by_school, x="school", y="present",
                 title="Attendance % by school",
                 labels={"present":"present %"},
                 color_discrete_sequence=[theme["palette"][1]])
    st.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)


# ============================================================
# SCHOOLS
# ============================================================
elif page == "Schools":
    section("Network")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Schools",  len(schools))
    c2.metric("Boards",   schools["board_name"].nunique())
    c3.metric("States",   schools["state"].nunique())
    c4.metric("Regions",  schools["region"].nunique())

    section("Average GPA by school")
    summary = (stu.groupby("school")
                  .agg(students=("student_id","count"),
                       avg_gpa=("current_gpa","mean"),
                       avg_att=("cumulative_attendance_pct","mean"))
                  .reset_index().sort_values("avg_gpa", ascending=False))
    fig = px.bar(summary, x="school", y="avg_gpa",
                 hover_data=["students","avg_att"],
                 title="Average GPA by school",
                 color_discrete_sequence=[theme["primary"]])
    st.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    section("Distribution")
    col1, col2 = st.columns(2)
    boards = schools["board_name"].value_counts().reset_index()
    boards.columns = ["board", "count"]
    fig = px.pie(boards, names="board", values="count",
                 title="Schools by board", hole=0.55,
                 color_discrete_sequence=theme["palette"])
    fig.update_traces(textposition="outside", textinfo="label+percent")
    col1.plotly_chart(style_fig(fig, theme), use_container_width=True)

    regions = schools["region"].value_counts().reset_index()
    regions.columns = ["region", "count"]
    fig = px.bar(regions, x="region", y="count",
                 color="region", title="Schools by region",
                 color_discrete_sequence=theme["palette"])
    col2.plotly_chart(style_fig(fig, theme, show_legend=False), use_container_width=True)

    if {"latitude","longitude"}.issubset(schools.columns):
        section("Geography")
        st.map(schools[["latitude","longitude"]].dropna(), zoom=4)


# ============================================================
# DOWNLOAD
# ============================================================
st.sidebar.markdown("---")
st.sidebar.download_button(
    "📥 Download filtered students",
    fstu.to_csv(index=False).encode("utf-8"),
    file_name="filtered_students.csv",
    mime="text/csv",
    use_container_width=True,
)
