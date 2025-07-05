#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 18:31:08 2025

@author: sazzad
"""

import numpy as np
import pickle
import os
import streamlit as st

SCRIPT_DIR = os.path.dirname(__file__)
model_path = os.path.join(SCRIPT_DIR, 'trained_mail_model.sav')

with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
        model = artifacts['model']
        vectorizer = artifacts['vectorizer']
        
def mail_prediction(input_mail):
    input_features = vectorizer.transform([input_mail] if isinstance(input_mail, str) else input_mail)
    prediction = model.predict(input_features)

    if (prediction[0] == 0):
      return 'Spam'
    else:
      return 'Ham'
      
def main():
    st.title("Email prediction Web App")
    input_mail = st.text_input("Insert the mail")
    
    result = '';
    
    if st.button("Email test result"):
        result = mail_prediction(input_mail)
        
    st.success(result)
    
    
if __name__ == '__main__' :
    main()
