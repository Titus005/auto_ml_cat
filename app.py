import streamlit as st
import pandas as pd
import os

# For profiling
import ydata_profiling 
from streamlit_ydata_profiling import st_profile_report

# For ML
from pycaret.classification import setup,compare_models,pull,save_model

# when runing on localhost execute the below code
# streamlit run app.py --server.enableXsrfProtection false







st.write('Just some music, while you explore the app ')
st.audio('lofi.mp3') 

with st.sidebar:
    st.write('Even a cat can make an ML Model')
    st.image('cutie-cat.gif')
    st.title('Auto Stream ML')
    choice = st.radio('Navigation',['Upload','Profiling','ML','Download','Contact me','meoww!'])
    st.info('This is a "ZABARDAST application that allows you to make an automated pipline using ydata_profiling, pycaret')


if os.path.exists('datasets.csv'):
    df = pd.read_csv('datasets.csv',index_col=None)


if choice == "Upload":
    file =  st.file_uploader('Upload your dataset file here')
    st.image('cat-waiting.gif')
    if file:
        df = pd.read_csv(file,index_col=None)
        df.to_csv('datasets.csv',index=None)
        st.dataframe(df)



if choice == "Profiling":
    st.title('Automated Exploratory Data Analysis ')
    pr = df.profile_report()
    st_profile_report(pr)



if choice == "ML":
    algo = ''
    st.title('Finding the best model ')
    
    y = st.selectbox(' Select your target ',list(df.columns))

    if st.button('Train the model'):
        setup(df,target=y)
        setup_df = pull()
        st.info('This is the ML experiment settings ')
        st.dataframe(setup_df.astype(str)
        best_model = compare_models()
        compare_df = pull()
        st.info('This is the ML Model Ranking')
        st.dataframe(compare_df)
        best_model
        save_model(best_model,"best_model")


if choice == "Download":
   
    with open('best_model.pkl','rb') as f:
        st.download_button("Download the Model file",f,'trained_model.pkl')
    st.image('iapprove-yes.gif')

if choice == 'Contact me':
    st.image('B_group.jpg',caption="i'm the guy with specs")
    st.write('[My Linkedin](https://www.linkedin.com/in/pratik-ramteke-21573317a/)')
    st.write('[My GitHub](https://github.com/PratikPhysics)')
    st.write('[My YouTube](https://www.youtube.com/@pratikgizmo6436)')
    st.write('For private Data Science and Machine Learning Training : call me : 7588399515,7620162941')
    st.image('cute-pose.gif')

if choice == 'meoww!':
    st.write(" Nah!! You're not a Cat!")
    st.image('anime-cat.gif')
    


