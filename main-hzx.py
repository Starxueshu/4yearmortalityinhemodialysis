# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import shap

st.header("Predicting Four-Year mortality in hemodialysis patients using machine learning techniques: Development and Validation Study")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Age = st.sidebar.slider("Age (years)", 40, 80)
Albumin = st.sidebar.slider("Albumin (g/L)", 30, 50)
N109L = st.sidebar.slider("Neutrophil (×10^9/L)", 3, 12)
EF = st.sidebar.slider("Ejection fraction (%)", 40, 80)
IDH= st.sidebar.selectbox("Ischemic heart disease", ("No", "Yes"))

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[IDH, Age, Albumin, N109L, EF]],
                     columns=["IDH", "Age", "Albumin", "N109L", "EF"])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Four-Year mortality: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.439:
        st.success(f"Risk group: low-risk group")
    else:
        st.error(f"Risk group: High-risk group")
    if prediction < 0.439:
        st.success(f"Regarding low-risk individuals, while those have a lower likelihood of four-year mortality, it is essential to closely monitor their cardiac function, blood cell count, and nutrition condition. This enables prompt detection and intervention in case of unexpected mortality. Furthermore, in individuals assessed as low-risk, a more conservative approach to hemodialysis management may be suitable. This includes avoiding unnecessary pharmacological therapy or frequent detection, and restricting interventions to specific clinical indications.")
    else:
        st.error(f"High-risk individuals should be thoroughly evaluated and optimized in their hemodialysis life. This includes optimizing their cardiac function including increasing heart ejection fraction and decreasing ischemic heart disease, avoiding infection, promoting nutrition condition, and even considering pharmacological interventions to improve those parameters. In addition, a comprehensive hemodialysis management plan should be implemented for high-risk patients. This may involve suitable ultrafiltration dehydration during hemodialysis, appropriate choice of dialysis methods, proper nutrition and regular routine cardiac color Doppler ultrasound based on individual needs.")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_train0 = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ["IDH", "Age", "Albumin", "N109L", "EF"]]
    y_train = y_train0.Group
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    # st.text(shap_value)

    shap.initjs()
    # image = shap.plots.force(shap_value)
    # image = shap.plots.bar(shap_value)
    st.pyplot(shap.plots.waterfall(shap_value[0]))
    st.pyplot(shap.plots.force(shap_value[0], matplotlib=True))
    #st.pyplot(shap.plots.bar(shap_value[0]))
    st.text(f"Note: Blue items indicate protective factors, while red items indicate risk factors.")
    st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Model introduction')
st.markdown('This openly available AI tool, exclusively crafted for research purposes, has been intricately engineered to thoroughly evaluate the risk of four-year mortality in hemodialysis patients through advanced machine learning techniques. Employing the eXGBM model, the AI application showcases a commendable prediction performance with an AUC of 0.933. This powerful tool has the potential to empower healthcare professionals, enabling them to make more precise and timely decisions, ultimately contributing to enhanced patient outcomes.')
