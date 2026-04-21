"""
Nexus Analytics – Multi-School Academic Dashboard
Uses every relevant sheet from the workbook:
Schools, Students, Teachers, Principals, Student_Academic_Records, Attendance_Log.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Nexus Analytics", layout="wide")


# ============================================================
# LOAD
# ============================================================
@st.cache_data(show_spinner="Loading workbook…")
def load_workbook(src) -> dict:
    """Read every sheet, normalise column names, return as a dict."""
    sheets = pd.read_excel(src, sheet_name=None)
    for name, frame in sheets.items():
        frame.columns = (frame.columns.astype(str)
                         .str.strip().str.lower().str.replace(" ", "_"))
        sheets[name] = frame
    return sheets


@st.cache_data(show_spinner="Building views…")
def build_views(sheets: dict) -> dict:
    """Join related sheets into ready-to-plot views."""
    schools  = sheets["Schools"].rename(columns={"school_name": "school"})
    students = sheets["Students"].copy()
    records  = sheets["Student_Academic_Records"].copy()
    attend   = sheets["Attendance_Log"].copy()
    teachers = sheets["Teachers"].copy()

    # Students + school metadata
    stu = students.merge(
        schools[["school_id", "school", "board_name", "region",
                 "state", "city", "school_type"]],
        on="school_id", how="left"
    )
    stu["performance_index"] = (
        stu["current_gpa"].fillna(0) * 25
        + stu["cumulative_attendance_pct"].fillna(0) * 0.5
    )

    # Academic records + student demographics + school
    rec = (records
           .merge(students[["student_id", "gender"]], on="student_id", how="left")
           .merge(schools[["school_id", "school", "board_name", "region"]],
                  on="school_id", how="left"))

    # Attendance log + school
    att = attend.merge(schools[["school_id", "school", "region"]],
                       on="school_id", how="left")
    att["attendance_date"] = pd.to_datetime(att["attendance_date"], errors="coerce")

    return {"students": stu, "records": rec, "attendance": att,
            "teachers": teachers, "schools": schools}


# ============================================================
# DATA SOURCE  (uploader, with fallback to a local file)
# ============================================================
st.sidebar.title("🎛 Filters")

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


# ============================================================
# FILTERS
# ============================================================
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

page = st.sidebar.radio("📂 Navigation",
    ["Overview", "Students", "Academic Records",
     "Teachers", "Attendance", "Schools"])

if fstu.empty:
    st.error("No rows match the current filters. Clear some filters in the sidebar.")
    st.stop()


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.title("📊 Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Schools",  stu["school"].nunique())
    c2.metric("Students", f"{len(fstu):,}")
    c3.metric("Teachers", f"{len(teachers):,}")
    c4.metric("Avg GPA",  round(fstu["current_gpa"].mean(), 2))
    c5.metric("Avg Attendance %", round(fstu["cumulative_attendance_pct"].mean(), 1))

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.histogram(fstu, x="current_gpa", nbins=30,
                                   title="GPA distribution"),
                      use_container_width=True)
    col2.plotly_chart(px.histogram(fstu, x="cumulative_attendance_pct", nbins=30,
                                   title="Attendance distribution"),
                      use_container_width=True)

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.pie(fstu, names="academic_risk_flag",
                             title="Risk composition"),
                      use_container_width=True)
    g = fstu["gender"].value_counts().reset_index()
    g.columns = ["gender", "count"]
    col2.plotly_chart(px.bar(g, x="gender", y="count",
                             title="Gender distribution"),
                      use_container_width=True)

    by_school = (fstu.groupby("school")
                     .agg(students=("student_id", "count"),
                          avg_gpa=("current_gpa", "mean"),
                          avg_attendance=("cumulative_attendance_pct", "mean"))
                     .reset_index()
                     .sort_values("students", ascending=False))
    st.plotly_chart(px.bar(by_school, x="school", y="students",
                           hover_data=["avg_gpa", "avg_attendance"],
                           title="Students per school"),
                    use_container_width=True)


# ============================================================
# STUDENTS
# ============================================================
elif page == "Students":
    st.title("🎓 Students")

    c1, c2, c3 = st.columns(3)
    c1.metric("Students in view", f"{len(fstu):,}")
    c2.metric("High risk",        int((fstu["academic_risk_flag"] == "High").sum()))
    c3.metric("Scholarship",      int((fstu["scholarship_flag"] == "Yes").sum()))

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.box(fstu, x="grade_level", y="current_gpa",
                             title="GPA by grade"),
                      use_container_width=True)
    col2.plotly_chart(px.box(fstu, x="grade_level", y="cumulative_attendance_pct",
                             title="Attendance by grade"),
                      use_container_width=True)

    st.plotly_chart(px.scatter(fstu, x="cumulative_attendance_pct",
                               y="current_gpa",
                               color="academic_risk_flag",
                               hover_data=["school", "grade_level"],
                               title="GPA vs attendance (coloured by risk)"),
                    use_container_width=True)

    st.plotly_chart(px.density_heatmap(fstu,
                                       x="cumulative_attendance_pct",
                                       y="current_gpa",
                                       title="Density — attendance vs GPA"),
                    use_container_width=True)


# ============================================================
# ACADEMIC RECORDS
# ============================================================
elif page == "Academic Records":
    st.title("📚 Academic Records")

    c1, c2, c3 = st.columns(3)
    c1.metric("Exam records",   f"{len(frec):,}")
    c2.metric("Pass rate",      f"{(frec['pass_fail']=='Pass').mean()*100:.1f}%")
    c3.metric("Avg percentage", round(frec["percentage"].mean(), 1))

    col1, col2 = st.columns(2)
    col1.plotly_chart(px.box(frec, x="subject_name", y="percentage",
                             title="Score by subject"),
                      use_container_width=True)
    grades_order = sorted(frec["grade_awarded"].dropna().unique())
    col2.plotly_chart(px.histogram(frec, x="grade_awarded",
                                   category_orders={"grade_awarded": grades_order},
                                   title="Grade distribution"),
                      use_container_width=True)

    term_subject = (frec.groupby(["term_name", "subject_name"])["percentage"]
                        .mean().reset_index())
    st.plotly_chart(px.bar(term_subject, x="subject_name", y="percentage",
                           color="term_name", barmode="group",
                           title="Average % by subject and term"),
                    use_container_width=True)

    st.plotly_chart(px.scatter(frec, x="assignment_score", y="marks_obtained",
                               color="subject_name",
                               title="Assignment score vs exam marks"),
                    use_container_width=True)


# ============================================================
# TEACHERS
# ============================================================
elif page == "Teachers":
    st.title("👩‍🏫 Teachers")

    c1, c2, c3 = st.columns(3)
    c1.metric("Teachers",        len(teachers))
    c2.metric("Avg rating",      round(teachers["teacher_performance_rating"].mean(), 2))
    c3.metric("Avg attendance %", round(teachers["teacher_attendance_pct"].mean(), 1))

    col1, col2 = st.columns(2)
    dept = teachers["department"].value_counts().reset_index()
    dept.columns = ["department", "count"]
    col1.plotly_chart(px.bar(dept, x="department", y="count",
                             title="Teachers by department"),
                      use_container_width=True)
    col2.plotly_chart(px.box(teachers, x="department",
                             y="teacher_performance_rating",
                             title="Rating by department"),
                      use_container_width=True)

    st.plotly_chart(px.scatter(teachers, x="years_experience",
                               y="teacher_performance_rating",
                               color="department",
                               hover_data=["subject_specialization",
                                           "teacher_attendance_pct"],
                               title="Experience vs performance rating"),
                    use_container_width=True)


# ============================================================
# ATTENDANCE
# ============================================================
elif page == "Attendance":
    st.title("📅 Attendance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Log entries", f"{len(fatt):,}")
    present_pct = (fatt["attendance_status"] == "Present").mean() * 100 if len(fatt) else 0
    c2.metric("Present %",   f"{present_pct:.1f}%")
    c3.metric("Dates",       fatt["attendance_date"].nunique())

    status = fatt["attendance_status"].value_counts().reset_index()
    status.columns = ["status", "count"]
    col1, col2 = st.columns(2)
    col1.plotly_chart(px.pie(status, names="status", values="count",
                             title="Attendance status split"),
                      use_container_width=True)

    by_grade = (fatt.assign(present=(fatt["attendance_status"] == "Present").astype(int))
                     .groupby("grade_level")["present"].mean().reset_index())
    by_grade["present"] *= 100
    col2.plotly_chart(px.bar(by_grade, x="grade_level", y="present",
                             title="Attendance % by grade",
                             labels={"present": "present %"}),
                      use_container_width=True)

    by_school = (fatt.assign(present=(fatt["attendance_status"] == "Present").astype(int))
                      .groupby("school")["present"].mean().reset_index()
                      .sort_values("present", ascending=False))
    by_school["present"] *= 100
    st.plotly_chart(px.bar(by_school, x="school", y="present",
                           title="Attendance % by school",
                           labels={"present": "present %"}),
                    use_container_width=True)


# ============================================================
# SCHOOLS
# ============================================================
elif page == "Schools":
    st.title("🏫 Schools")

    c1, c2, c3 = st.columns(3)
    c1.metric("Schools", len(schools))
    c2.metric("Boards",  schools["board_name"].nunique())
    c3.metric("States",  schools["state"].nunique())

    summary = (stu.groupby("school")
                  .agg(students=("student_id", "count"),
                       avg_gpa=("current_gpa", "mean"),
                       avg_att=("cumulative_attendance_pct", "mean"))
                  .reset_index()
                  .sort_values("avg_gpa", ascending=False))
    st.plotly_chart(px.bar(summary, x="school", y="avg_gpa",
                           hover_data=["students", "avg_att"],
                           title="Average GPA by school"),
                    use_container_width=True)

    col1, col2 = st.columns(2)
    boards = schools["board_name"].value_counts().reset_index()
    boards.columns = ["board", "count"]
    col1.plotly_chart(px.pie(boards, names="board", values="count",
                             title="Schools by board"),
                      use_container_width=True)

    regions = schools["region"].value_counts().reset_index()
    regions.columns = ["region", "count"]
    col2.plotly_chart(px.bar(regions, x="region", y="count",
                             title="Schools by region"),
                      use_container_width=True)

    if {"latitude", "longitude"}.issubset(schools.columns):
        st.subheader("School locations")
        st.map(schools[["latitude", "longitude"]].dropna())


# ============================================================
# DOWNLOAD
# ============================================================
st.sidebar.download_button(
    "📥 Download filtered students",
    fstu.to_csv(index=False).encode("utf-8"),
    file_name="filtered_students.csv",
    mime="text/csv",
)
