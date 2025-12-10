import streamlit as st
from utils import extract_text_from_file, load_sentence_transformer, embed_texts, compute_cosine_similarity
from gsheet_utils import load_sheet
from gsheet_drive_utils import download_drive_file
import numpy as np
import re

st.set_page_config(page_title='CV Matcher', layout='wide')

# --- Custom CSS for modern look ---
st.markdown('''
    <style>
    body {
        background: #111827 !important;
    }
    .main, .block-container, .stApp, .st-cq, .st-emotion-cache-1wrcr25, .st-emotion-cache-uf99v8, .st-emotion-cache-1kyxreq, .st-emotion-cache-13ln4jf, .st-emotion-cache-1gulkj5, .st-emotion-cache-1d391kg, .st-emotion-cache-1v0mbdj, .st-emotion-cache-1dp5vir, .st-emotion-cache-1y4p8pa, .st-emotion-cache-1avcm0n, .st-emotion-cache-1r6slb0 {
        background: #111827 !important;
    }
    .main {
        background: #111827 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        border-radius: 18px;
        background: rgba(17,24,39,0.95) !important;
        box-shadow: 0 4px 24px 0 rgba(17,24,39, 0.10);
    }
    .stButton>button {
        background: linear-gradient(90deg, #8B5CF6 0%, #F59E0B 100%);
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px 0 rgba(124, 58, 237, 0.13);
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #7C3AED 0%, #D97706 100%);
        color: #fff;
        box-shadow: 0 4px 16px 0 rgba(124, 58, 237, 0.22);
    }
    .stMetric {
        background: linear-gradient(90deg, #8B5CF6 0%, #F59E0B 100%) !important;
        border-radius: 10px;
        padding: 0.5rem 0.5rem 0.2rem 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px 0 rgba(124, 58, 237, 0.09);
        color: #fff !important;
    }
    .stTable, .stDataFrame {
        background: #111827 !important;
        color: #F3F4F6 !important;
        border-radius: 10px;
        box-shadow: 0 2px 8px 0 rgba(17,24,39, 0.09);
    }
    .stExpander {
        background: #1e293b !important;
        color: #F3F4F6 !important;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 8px 0 rgba(17,24,39, 0.09);
    }
    .stFileUploader, .stFileUploader > div, .stFileUploader > section {
        background: #1e293b !important;
        color: #F3F4F6 !important;
        border-radius: 10px;
    }
    .stTextInput > div > input, .stTextInput > div > div > input, .stTextArea > div > textarea {
        background: #1e293b !important;
        color: #F3F4F6 !important;
        border-radius: 8px;
        border: 1px solid #374151 !important;
    }
    .stSidebar, .css-1d391kg, .css-1v0mbdj {
        background: #111827 !important;
        color: #F3F4F6 !important;
    }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #8B5CF6 0%, #F59E0B 100%);
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px 0 rgba(124, 58, 237, 0.13);
        transition: 0.2s;
    }
    .stDownloadButton>button:hover {
        background: linear-gradient(90deg, #7C3AED 0%, #D97706 100%);
        color: #fff;
        box-shadow: 0 4px 16px 0 rgba(124, 58, 237, 0.22);
    }
    .stAlert, .stWarning, .stError, .stException, .st-emotion-cache-13ln4jf, .st-emotion-cache-1kyxreq {
        background: #1e293b !important;
        color: #F59E0B !important;
        border-radius: 8px;
        font-size: 1rem;
        border: 1px solid #374151 !important;
    }
    .stHeader>div>h1 {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-weight: 800;
        color: #7C3AED;
        letter-spacing: 0.5px;
    }
    .stSubheader>div>h2 {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-weight: 700;
        color: #F59E0B;
    }
    </style>
''', unsafe_allow_html=True)

# Only show the title, blurb, and file uploader if not in dashboard mode
if not st.session_state.get('dashboard', False):
    st.title('🚀 CV Matcher')
    st.markdown('''<div style="font-size:1.2rem; color:#2563eb; font-weight:500; margin-bottom:1.5rem;">Upload a job requirements file and let AI rank your candidates by best fit. <br> <span style="color:#4F8DFD;">Google Drive & Google Sheets integration. Modern, fast, and beautiful.</span></div>''', unsafe_allow_html=True)
    job_file = st.file_uploader('Upload job requirements (PDF/DOCX/TXT)', type=['pdf','docx','txt'], key='jobdesc')
    if job_file:
        st.session_state['job_file'] = job_file
else:
    # Remove file uploader and blurb in dashboard mode
    job_file = st.session_state.get('job_file', None)

model_name = st.sidebar.text_input('Sentence Transformer model', value='all-MiniLM-L6-v2')

st.sidebar.markdown("---")
sheet_url = st.sidebar.text_input("Google Sheet URL", "")
if st.sidebar.button("Load Submissions") and sheet_url:
    df = load_sheet(sheet_url)
    st.write("Latest Submissions:", df)
    st.write("Detected columns:", list(df.columns))

def extract_drive_file_id(url):
    """Extracts the file ID from any Google Drive link format."""
    if not url:
        return None
    patterns = [
        r"/file/d/([\w-]+)",
        r"id=([\w-]+)",
        r"/uc\?export=download&id=([\w-]+)",
        r"/open\?id=([\w-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    # If the link is just the file ID
    if re.match(r"^[\w-]{20,}$", url):
        return url
    return None

if st.sidebar.button("Run CV Matching") and sheet_url:
    df = load_sheet(sheet_url)
    st.write("Running CV Matching...")
    job_file = st.session_state.get('job_file', None)
    if job_file is None:
        st.info('Please upload a job requirements file to continue.')
    else:
        job_text = extract_text_from_file(job_file)
        if not job_text:
            st.error('Could not extract text from requirements file')
        else:
            st.subheader('Requirements preview')
            st.write(job_text[:2000])
            model = load_sentence_transformer(model_name)
            job_emb = embed_texts(model, job_text)[0]
            match_results = []
            found_any_cv = False
            service_account_email = 'cv-matcher-service@cv-matcher-480713.iam.gserviceaccount.com'
            for idx, row in df.iterrows():
                name = row.get('Name') or row.get('Full Name') or row.get('   Full Name  ')
                file_url = row.get('CV File URL') or row.get('CV Upload') or row.get('upload yoour cv') or row.get('  Upload Your CV (PDF or DOCX)  ')
                file_id = extract_drive_file_id(file_url)
                st.info(f"Processing: {name}, CV Link: {file_url}, File ID: {file_id}")
                if not file_id:
                    st.warning(f"No valid Google Drive file ID found for {name} (row {idx+1})")
                    continue
                found_any_cv = True
                from gsheet_drive_utils import share_file_with_service_account, ensure_service_account_in_shared_drive
                try:
                    share_file_with_service_account(file_id, service_account_email)
                    local_path = download_drive_file(file_id)
                    with open(local_path, 'rb') as f:
                        cv_text = extract_text_from_file(f)
                    cv_emb = embed_texts(model, cv_text)[0]
                    sim = compute_cosine_similarity(job_emb, cv_emb)
                    result = dict(row)
                    result['similarity'] = sim
                    result['cv_text'] = cv_text
                    match_results.append(result)
                except Exception as share_err:
                    st.warning(f"Sharing failed, attempting to add service account to shared drive: {share_err}")
                    try:
                        ensure_service_account_in_shared_drive(file_id, service_account_email)
                        share_file_with_service_account(file_id, service_account_email)
                        local_path = download_drive_file(file_id)
                        with open(local_path, 'rb') as f:
                            cv_text = extract_text_from_file(f)
                        cv_emb = embed_texts(model, cv_text)[0]
                        sim = compute_cosine_similarity(job_emb, cv_emb)
                        result = dict(row)
                        result['similarity'] = sim
                        result['cv_text'] = cv_text
                        match_results.append(result)
                    except Exception as drive_err:
                        result = dict(row)
                        result['similarity'] = 0
                        result['cv_text'] = f'Error: {drive_err}'
                        match_results.append(result)
                        st.error(f"Failed to add service account to shared drive or re-share file: {drive_err}")
            if not found_any_cv:
                st.error('No valid CV links found in the Google Sheet. Please check your sheet columns and links.')
            if not match_results:
                st.error('No applicants were processed. Please check your data and try again.')
            else:
                match_results = sorted(match_results, key=lambda r: r['similarity'], reverse=True)
                st.session_state['match_results'] = match_results
                import pandas as pd
                import re
                # Detect name column for summary table
                name_col_candidates = ['name', 'fullname', 'full_name', 'candidate', 'applicant', 'applicantname', 'candidate_name']
                def normalize_col(col):
                    return re.sub(r'[^a-z0-9]', '', col.strip().lower())
                normalized_col_map = {normalize_col(col): col for col in df.columns}
                name_col = None
                for norm in name_col_candidates:
                    if norm in normalized_col_map:
                        name_col = normalized_col_map[norm]
                        break
                table_data = []
                for idx, r in enumerate(match_results, 1):
                    sim_clamped = max(0.0, r['similarity'])
                    candidate_name = r.get(name_col) if name_col else f"Candidate {idx}"
                    table_data.append({
                        'Position': f'{idx}'+('st' if idx==1 else 'nd' if idx==2 else 'rd' if idx==3 else 'th'),
                        'Candidate': candidate_name,
                        'Match %': float(f"{sim_clamped*100:.2f}")
                    })
                df_results = pd.DataFrame(table_data)
                st.table(df_results)
                import matplotlib.pyplot as plt
                import seaborn as sns
                st.subheader('Match Percentage Visualization')
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x='Candidate', y='Match %', data=df_results, ax=ax, palette='viridis')
                ax.set_ylim(0, 100)
                ax.set_ylabel('Match Percentage (%)')
                ax.set_xlabel('Candidate')
                ax.set_title('Candidate Match Percentage Ranking')
                for i, v in enumerate(df_results['Match %']):
                    ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
                st.pyplot(fig)
                for idx, r in enumerate(match_results, 1):
                    preview_text = r['cv_text']
                    # Clean up preview text for readability
                    if isinstance(preview_text, str):
                        preview_text = preview_text.replace('\r', '\n')
                        preview_text = '\n'.join([line.strip() for line in preview_text.splitlines() if line.strip()])
                        preview_text = re.sub(r'\n{2,}', '\n\n', preview_text)  # Collapse multiple blank lines
                    # Robust name column detection for preview
                    name_col_candidates = ['name', 'fullname', 'full_name', 'candidate', 'applicant', 'applicantname', 'candidate_name']
                    def normalize_col(col):
                        return re.sub(r'[^a-z0-9]', '', col.strip().lower())
                    normalized_col_map = {normalize_col(col): col for col in r.keys()}
                    name_col = None
                    for norm in name_col_candidates:
                        if norm in normalized_col_map:
                            name_col = normalized_col_map[norm]
                            break
                    candidate_name = r.get(name_col) if name_col else f"Candidate {idx}"
                    with st.expander(f"{idx}. {candidate_name} - {r['similarity']*100:.2f}% match: Preview CV text"):
                        st.write(preview_text[:2000])

# DASHBOARD SECTION
import streamlit as st
from streamlit import columns
st.sidebar.markdown('---')
st.sidebar.header('Dashboard')
if 'dashboard' not in st.session_state:
    st.session_state['dashboard'] = False
if not st.session_state['dashboard']:
    if st.sidebar.button('Show Dashboard'):
        st.session_state['dashboard'] = True
else:
    if st.sidebar.button('Hide Dashboard'):
        st.session_state['dashboard'] = False

match_results_dashboard = st.session_state.get('match_results', [])
if st.session_state.get('dashboard', False):
    # Only show dashboard header, not the main app title or blurb
    st.header('📊 CV Matcher Dashboard')
    st.markdown('---')
    if match_results_dashboard:
        import pandas as pd
        import matplotlib.pyplot as plt
        # --- Summary Cards ---
        st.subheader('Summary Statistics')
        col1, col2, col3, col4 = st.columns(4)
        df_dash = pd.DataFrame(match_results_dashboard)
        total_applicants = len(df_dash)
        highest = max([max(0.0, r['similarity'])*100 for r in match_results_dashboard])
        lowest = min([max(0.0, r['similarity'])*100 for r in match_results_dashboard])
        avg = np.mean([max(0.0, r['similarity'])*100 for r in match_results_dashboard])
        missing = sum(['Error:' in r['cv_text'] for r in match_results_dashboard])
        col1.metric('👥 Total Applicants', total_applicants)
        col2.metric('🏆 Highest Match %', f"{highest:.2f}%")
        col3.metric('📉 Lowest Match %', f"{lowest:.2f}%")
        col4.metric('⚠️ Invalid CVs', missing)
        st.markdown('---')
        # --- Match Distribution ---
        st.subheader('📈 Match Distribution')
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist([max(0.0, r['similarity'])*100 for r in match_results_dashboard], bins=10, color='#4F8DFD', edgecolor='white')
        ax.set_xlabel('Match Percentage', fontsize=11)
        ax.set_ylabel('Number of Candidates', fontsize=11)
        ax.set_title('Distribution of Match %', fontsize=13, fontweight='bold')
        st.pyplot(fig)
        st.markdown('---')
        # --- Data export ---
        st.subheader('⬇️ Export Results')
        # Robust name column detection for export
        name_col_candidates = ['name', 'fullname', 'full_name', 'candidate', 'applicant', 'applicantname', 'candidate_name']
        def normalize_col(col):
            return re.sub(r'[^a-z0-9]', '', col.strip().lower())
        normalized_col_map = {normalize_col(col): col for col in df_dash.columns}
        name_col = None
        for norm in name_col_candidates:
            if norm in normalized_col_map:
                name_col = normalized_col_map[norm]
                break
        export_names = [r.get(name_col) if name_col else f"Candidate {i+1}" for i, r in enumerate(match_results_dashboard)]
        df_export = pd.DataFrame({
            'Candidate': export_names,
            'Match %': [max(0.0, r['similarity'])*100 for r in match_results_dashboard]
        })
        st.download_button('Download Results as CSV', df_export.to_csv(index=False), file_name='cv_match_results.csv', help='Download the ranked results as a CSV file.')
        st.markdown('---')
        # --- Email Shortlisted Candidates ---
        st.subheader('✉️ Send Email to Shortlisted Applicants')
        # Improved robust detection of email and name columns
        import re
        def normalize_col(col):
            return re.sub(r'[^a-z0-9]', '', col.strip().lower())
        normalized_col_map = {normalize_col(col): col for col in df_dash.columns}
        possible_email_names = ['email', 'e-mail', 'emailaddress', 'email_address', 'e-mailaddress', 'e_mail', 'mail']
        possible_name_names = ['name', 'fullname', 'full_name', 'candidate', 'applicant', 'applicantname', 'candidate_name']
        email_col = None
        name_col = None
        for norm in possible_email_names:
            if norm in normalized_col_map:
                email_col = normalized_col_map[norm]
                break
        for norm in possible_name_names:
            if norm in normalized_col_map:
                name_col = normalized_col_map[norm]
                break
        if not email_col or not name_col:
            st.warning(f"Could not automatically detect email or name column. Available columns: {list(df_dash.columns)}. Please check your sheet headers.")
        else:
            top_n_email = st.number_input('Number of top candidates to email', min_value=1, max_value=len(df_dash), value=3)
            shortlisted = df_dash.sort_values('similarity', ascending=False).head(top_n_email)
            st.write('Shortlisted candidates:', shortlisted[[name_col, email_col, 'similarity']])
            subject = st.text_input('Email Subject', value='Congratulations! You have been shortlisted')
            message = st.text_area('Email Message', value='Dear {name},\n\nCongratulations! You have been shortlisted for the next stage. We will contact you soon.\n\nBest regards,\nRecruitment Team')
            sender_email = st.text_input('Your Email (Gmail recommended)', key='sender_email')
            sender_password = st.text_input('App Password (see Gmail app password instructions)', type='password', key='sender_password')
            send_emails = st.button('Send Emails to Shortlisted')
            if send_emails:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                success_list = []
                fail_list = []
                if not sender_email or not sender_password:
                    st.error('Please enter your email and app password.')
                else:
                    for _, row in shortlisted.iterrows():
                        to_email = row[email_col]
                        candidate_name = row[name_col]
                        if not to_email or not isinstance(to_email, str):
                            continue
                        msg = MIMEMultipart()
                        msg['From'] = sender_email
                        msg['To'] = to_email
                        msg['Subject'] = subject
                        personalized = message.replace('{name}', str(candidate_name))
                        msg.attach(MIMEText(personalized, 'plain'))
                        try:
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                                server.login(sender_email, sender_password)
                                server.sendmail(sender_email, to_email, msg.as_string())
                            success_list.append(to_email)
                        except Exception as e:
                            st.error(f"Failed to send to {to_email}: {e}")
                            fail_list.append(to_email)
                    st.success(f"Sent {len(success_list)} emails to shortlisted candidates.")
        st.markdown('---')
        # --- CV Review ---
        st.subheader('📝 CV Review')
        # Use robust name column detection for CV preview
        name_col_candidates = ['name', 'fullname', 'full_name', 'candidate', 'applicant', 'applicantname', 'candidate_name']
        def normalize_col(col):
            return re.sub(r'[^a-z0-9]', '', col.strip().lower())
        normalized_col_map = {normalize_col(col): col for col in df_dash.columns}
        name_col = None
        for norm in name_col_candidates:
            if norm in normalized_col_map:
                name_col = normalized_col_map[norm]
                break
        for idx, r in enumerate(match_results_dashboard, 1):
            candidate_name = r.get(name_col) if name_col else f"Candidate {idx}"
            preview_text = r.get('cv_text', '')
            if isinstance(preview_text, str):
                preview_text = preview_text.replace('\r', '\n')
                preview_text = '\n'.join([line.strip() for line in preview_text.splitlines() if line.strip()])
                import re
                preview_text = re.sub(r'\n{2,}', '\n\n', preview_text)
            st.expander(f"{idx}. {candidate_name} - {max(0.0, r['similarity'])*100:.2f}% match: Preview CV text").write(preview_text[:2000])
    else:
        st.info('No match results to display yet. Run CV Matching first.')
