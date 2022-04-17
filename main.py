#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:29:31 2022

@author: ubuntu
"""
import streamlit as st
import arxiv
# import re
import os
import pandas as pd

from utils import display_pdf
from model import DataModel

#todo: move this to configfile
app_name = 'NLP, Myself and I'
app_description = 'Analyze, synthezise and translate scientific papers using NLP techniques'
app_layout = 'wide'
app_title = 'NLPM&I'
default_input_url = 'https://arxiv.org/pdf/2101.02288.pdf'
arxiv_default_download_path = './papers'


#SIDEBAR
st.set_page_config(page_title=app_title, layout=app_layout)
st.sidebar.text(app_name)
st.header(app_name)
st.subheader(app_description)

#ARXIV CONTROLS
arxiv_input_url = st.text_input('Enter arxiv url', placeholder=default_input_url, value=default_input_url)
#todo: this should be a regex
if arxiv_input_url[0:22] != 'https://arxiv.org/pdf/' and \
   arxiv_input_url[0:35] != 'https://arxiv.org/ftp/arxiv/papers/':
    st.text('url not matching, abort')
    st.stop()

arxiv_id = arxiv_input_url.split('/')[-1].replace('.pdf','')
arxiv_search = arxiv.Search(id_list=[arxiv_id])
arxiv_paper = next(arxiv_search.results())
arxiv_pdf_filename = '_'.join([i for i in arxiv_paper.title.lower().split(' ')]) + '.pdf'

if not os.path.exists(f'{arxiv_default_download_path}/{arxiv_pdf_filename}'):
    with st.spinner(text='Downloading PDF'):
        arxiv_paper.download_pdf(arxiv_default_download_path, arxiv_pdf_filename)

if arxiv_paper:
    nice_display = lambda x, name: pd.DataFrame([x]).T.rename(columns={0:name})
    justify = lambda txt: f"<p align='justify'> {txt}  </p>"
    
    pdf_side, analytics_side  = st.columns(2)
    pdf_display = display_pdf(arxiv_default_download_path, arxiv_pdf_filename)
    
    with st.spinner(text='Generating data model'):
        analytics_model = DataModel(path=arxiv_default_download_path,file=arxiv_pdf_filename)
    
    with pdf_side:
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        st.subheader('Text Analytics')
        st.table(nice_display(analytics_model.data['stat'], 'clean data stats'))
        
        st.subheader('LDA Model')
        st.table(analytics_model.dtf_topics)
    
    with analytics_side:
        
        st.header(arxiv_paper.title)
        st.text(f'Published: {str(arxiv_paper.published)}     Updated: {str(arxiv_paper.updated)}')
        st.text(f'Categories: {", ".join(arxiv_paper.categories)}')
        st.text(', '.join(str(auth) for auth in arxiv_paper.authors))
        st.pyplot(analytics_model.wc_fig)
        
        # st.text()
        st.subheader('Key phrases')
        st.markdown(justify(f'{", ".join(list(analytics_model.data["tr"]["key_phrases"])[0:15])}'), unsafe_allow_html = True)
        
        st.subheader('Key sentences')
        key_sentences_txt=".\n\n".join(analytics_model.data["tr"]["key_sentences"].split("."))
        st.markdown(justify(key_sentences_txt), unsafe_allow_html = True)

        # st.header('Summarization:')
                
        
