#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:49:08 2022

@author: ubuntu
"""
import base64

def display_pdf(path, name):
    # Opening file from file path
    file = f'{path}/{name}'
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    # pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="600" height="600" type="application/pdf"></iframe>'

    return pdf_display
