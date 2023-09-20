import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


matplotlib.use("Agg")
st.set_option('deprecation.showPyplotGlobalUse', False)

best_features=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP','MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE']

gender_dict={"male":1,"female":2}
feature_dict={"No":1,"Yes":2}

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return value

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return key
def load_model(model_file):
    loaded_model=joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model
        
def get_fvalue(val):
    feature_dict={"No":1,"Yes":2}
    for key,value in feature_dict.items():
        if val==key:
            return value

def main():
    
    
        st.subheader("Home")
        st.text("what is parkinson?")
        st.text("""hile genetics is thought to play a role in Parkinson's, in most cases the disease does not seem to run in families.
                Many researchers now believe that Parkinson's results from a combination of genetic and environmental factors,
                such as exposure to toxins""")
        st.text("What does Parkinson's disease do to a person?")
        st.text("""Over time, Parkinson's disease may slow your movement, making simple tasks difficult and time-consuming.
                Your steps may become shorter when you walk. It may be difficult to get out of a chair.
                You may drag or shuffle your feet as you try to walk.""")
        
        
        
        # 'MDVP_Fo_Hz', 'MDVP_Fhi_Hz', 'MDVP_Flo_Hz', 'MDVP_Jitter', 'MDVP_Jitter_Abs', 'MDVP_RAP',
    # 'MDVP_PPQ', 'Jitter_DDP','MDVP_Shimmer', 'MDVP_Shimmer_dB',
    # 'Shimmer_APQ3', 'Shimmer_APQ5',
    # 'MDVP_APQ', 'Shimmer_DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    # 'spread1', 'spread2', 'D2', 'PPE'


        st.subheader("Predictive analysis")
        MDVP_Fo_Hz=st.number_input("MDVP Fo(Hz)",format="%.7f")
        MDVP_Fhi_Hz=st.number_input("MDVP Fhi(Hz)",format="%.7f")
        MDVP_Flo_Hz=st.number_input("MDVP Flo(Hz)",format="%.7f")
        MDVP_Jitter=st.number_input("MDVP Jitter(%)",format="%.7f")
        MDVP_Jitter_Abs=st.number_input("MDVP Jitter(Abs)",format="%.7f")
        MDVP_RAP=st.number_input("MDVP RAP",format="%.7f")
        MDVP_PPQ = st.number_input("MDVP PPQ",format="%.7f")
        Jitter_DDP = st.number_input("Jitter (DDP)",format="%.7f")
        MDVP_Shimmer = st.number_input("MDVP Shimmer",format="%.7f")
        MDVP_Shimmer_dB = st.number_input("MDVP Shimmer(dB)",format="%.7f")
        Shimmer_APQ3 = st.number_input("Shimmer APQ3",format="%.7f")
        Shimmer_APQ5 = st.number_input("Shimmer APQ5",format="%.7f")
        MDVP_APQ = st.number_input("MDVP APQ",format="%.7f")
        Shimmer_DDA = st.number_input("Shimmer DDA",format="%.7f")
        NHR = st.number_input("NHR",format="%.7f")
        HNR = st.number_input("HNR",format="%.7f")
        RPDE = st.number_input("RPDE",format="%.7f")
        DFA = st.number_input("DFA",format="%.7f")
        spread1 = st.number_input("spread1",format="%.7f")
        spread2 = st.number_input("spread2",format="%.7f")
        D2 = st.number_input("D2",format="%.7f")
        PPE = st.number_input("PPE",format="%.7f")
        
        feature_list = [
    MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs,
    MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
    Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
    RPDE, DFA, spread1, spread2, D2, PPE
]

        sample_data=np.array(feature_list).reshape(1,-1)
        
        if st.button("predict"):
            
                loader_model=load_model("xgb.pkl")
                p=loader_model.predict(sample_data)
                if p==1:
                    st.success("safe")
                    st.text("Prediction Probabilistic Score using {}".format("Xgboost"))
                    
                elif p==0:
                    st.warning("danger")
                    st.text("Prediction Probabilistic Score using {}".format("Xgboost"))
                    
            
        

if __name__ == "__main__":
    main()