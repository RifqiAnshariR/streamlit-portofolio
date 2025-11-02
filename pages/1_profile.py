import base64
import json
import streamlit as st
from config.config import Config
from utils.styling import load_css


st.set_page_config(page_title="Profile", 
                   page_icon=Config.PAGE_ICON, 
                   layout=Config.LAYOUT, 
                   menu_items=Config.MENU_ITEMS)


load_css()


@st.cache_data(ttl=Config.CACHE_TTL)
def get_personal_info(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_data(ttl=Config.CACHE_TTL)
def get_file_bytes(file_path):
    with open(file_path, "rb") as file:
        return file.read()


try:
    personal_info = get_personal_info(Config.PERSONAL_INFO_PATH)
    image_bytes = get_file_bytes(Config.PROFILE_PHOTO_PATH)
    base64_image = base64.b64encode(image_bytes).decode()
    cv_bytes = get_file_bytes(Config.CV_PATH)
except FileNotFoundError:
    st.error("File not found.")
    st.stop()
except json.JSONDecodeError:
    st.error(f"Error: JSON file '{Config.PERSONAL_INFO_PATH}' is corrupt or invalid.")
    st.stop()


NAME = personal_info.get("name", "N/A")
CONTACT = personal_info.get("contact", {})
SOCIALS = personal_info.get("socials", [])
CV_FILENAME = personal_info.get("cv_filename", "N/A")
EDUCATIONS = personal_info.get("educations", [])
EXPERIENCES = personal_info.get("experiences", [])
PROJECTS = personal_info.get("projects", [])
SKILLS = personal_info.get("skills", [])
CERTIFICATIONS = personal_info.get("certifications", [])
LANGUAGES = personal_info.get("languages", [])


st.title("Profile Dashboard")
st.html(
        f'''
        <img class="profile_img"
             src="data:image/jpeg;base64,{base64_image}"
             alt="Profile Photo">
        '''
    )
st.header(NAME)
st.divider()


col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Education(s)")
    clean_educations = [e for e in EDUCATIONS if e.get("institution")]
    if not clean_educations:
        st.write("No education history to display.")
    for edu in clean_educations:
        title = edu.get('degree', '') + " — " + edu.get('institution') if edu.get('degree') else edu.get('institution')
        with st.expander(title):
            st.write(f"**Duration:** {edu.get('duration')}")
            if edu.get('score'):
                st.write(f"**Score:** {edu.get('score')}")
            if edu.get('gpa'):
                st.write(f"**GPA:** {edu.get('gpa')}")
            details = [d for d in edu.get("details", []) if d]
            if details:
                st.write("**Details:**")
                for detail in details:
                    st.write(f"- {detail}")

    st.subheader("Work Experience(s)")
    clean_experiences = [e for e in EXPERIENCES if e.get("role") and e.get("company")]
    if not clean_experiences:
        st.write("No work experience to display.")
    for exp in clean_experiences:
        with st.expander(f"{exp.get('role')} at {exp.get('company')}"):
            st.write(f"**Duration:** {exp.get('duration')}")
            st.write(f"**Description:** {exp.get('description')}")
            details = [d for d in exp.get("details", []) if d]
            if details:
                st.write("**Key Achievements:**")
                for detail in details:
                    st.write(f"- {detail}")

    st.subheader("Project(s)")
    clean_projects = [p for p in PROJECTS if p.get("title")]
    if not clean_projects:
        st.write("No projects to display.")
    for project in clean_projects:
        with st.expander(project.get("title")):
            st.write(project.get("description"))
            techs = [t for t in project.get("technologies", []) if t]
            if techs:
                st.write(f"**Technologies:** {', '.join(techs)}")
            if project.get("link"):
                st.link_button("View Project", project.get("link"))

    st.subheader("Skill(s)")
    clean_skills = [s for s in SKILLS if s.get("category")]
    if not clean_skills:
        st.write("No skills to display.")
    for skill_cat in clean_skills:
        items = [i for i in skill_cat.get("items", []) if i]
        if items:
            st.write(f"**{skill_cat.get('category')}**")
            st.write(", ".join(items))

with col2:
    st.subheader("Certification(s)")
    clean_certs = [c for c in CERTIFICATIONS if c.get("name")]
    if not clean_certs:
        st.write("No certifications to display.")
    for cert in clean_certs:
        st.write(f"{cert.get('name')} — {cert.get('issuer')}")
        date = cert.get('date', '')
        cred_id = cert.get('credential_id', '')
        if date or cred_id:
            caption_parts = []
            if date:
                caption_parts.append(f"Date: {date}")
            if cred_id:
                caption_parts.append(f"Credential ID: {cred_id}")
            st.caption(" | ".join(caption_parts))
    
    st.subheader("Language(s)")
    clean_languages = [lang for lang in LANGUAGES if lang.get("name")]
    if not clean_languages:
        st.write("N/A")
    for lang in clean_languages:
        st.write(f"**{lang.get('name')}:** {lang.get('proficiency', 'N/A')}")
    st.subheader("CV")
    st.download_button(
        label="Download CV",
        data=cv_bytes,
        file_name=CV_FILENAME,
        mime="application/pdf",
        use_container_width=True
    )
    st.subheader("Contact(s)")
    st.write(f"**Email:** {CONTACT.get('email', 'N/A')}")
    st.write(f"**Phone:** {CONTACT.get('phone', 'N/A')}")
    st.subheader("Social Media(s)")
    clean_socials = [s for s in SOCIALS if s.get("name") and s.get("url")]
    for social in clean_socials:
        st.link_button(
            social.get("name"), 
            social.get("url"), 
            use_container_width=True
        )


st.html("<div class='footer'>©2025 Rifqi Anshari Rasyid.</div>")
