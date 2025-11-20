import streamlit as st
from utils import extract_text_from_file, load_sentence_transformer, embed_texts, compute_cosine_similarity
import numpy as np

st.set_page_config(page_title='CV Matcher', layout='wide')

st.title('CV Matcher')

st.markdown('''
Upload a job requirements file (PDF/DOCX/TXT) and several candidate CVs. The app will rank candidates by similarity to the job requirements using a SentenceTransformer model.
''')

model_name = st.sidebar.text_input('Sentence Transformer model', value='all-MiniLM-L6-v2')

model = load_sentence_transformer(model_name)

req_file = st.file_uploader('Upload job requirements (PDF/DOCX/TXT)', type=['pdf','docx','txt'])

if req_file is not None:
    req_text = extract_text_from_file(req_file)
    if not req_text:
        st.error('Could not extract text from requirements file')
    else:
        st.subheader('Requirements preview')
        st.write(req_text[:2000])

        req_emb = embed_texts(model, req_text)[0]

        st.markdown('---')

        cv_files = st.file_uploader('Upload candidate CVs (multiple)', type=['pdf','docx','txt'], accept_multiple_files=True)
        if cv_files:
            results = []
            for f in cv_files:
                text = extract_text_from_file(f)
                emb = embed_texts(model, text)[0]
                sim = compute_cosine_similarity(req_emb, emb)
                results.append({'name': f.name, 'similarity': sim, 'text': text})

            results = sorted(results, key=lambda r: r['similarity'], reverse=True)

            st.subheader('Ranked candidates')
            if results:
                import pandas as pd
                import matplotlib.pyplot as plt
                import seaborn as sns
                table_data = []
                for idx, r in enumerate(results, 1):
                    table_data.append({
                        'Position': f'{idx}'+('st' if idx==1 else 'nd' if idx==2 else 'rd' if idx==3 else 'th'),
                        'Candidate': r['name'],
                        'Match %': float(f"{r['similarity']*100:.2f}")
                    })
                df = pd.DataFrame(table_data)
                st.table(df)

                # Visualization: Bar chart of match %
                st.subheader('Match Percentage Visualization')
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x='Candidate', y='Match %', data=df, ax=ax, palette='viridis')
                ax.set_ylim(0, 100)
                ax.set_ylabel('Match Percentage (%)')
                ax.set_xlabel('Candidate')
                ax.set_title('Candidate Match Percentage Ranking')
                for i, v in enumerate(df['Match %']):
                    ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
                st.pyplot(fig)

                for idx, r in enumerate(results, 1):
                    with st.expander(f"{idx}. {r['name']} - {r['similarity']*100:.2f}% match: Preview CV text"):
                        st.write(r['text'][:2000])
