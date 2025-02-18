import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap
#导入训练集数据
train_data =pd.read_csv("D:\\numpy\\001科研\\训练集建模因素.csv",encoding='utf-8')
print(train_data.shape) #获取目前的形状
print(train_data.columns)
trainy=train_data.OUTCOME
trainy.head()
trainx=train_data.drop('OUTCOME',axis=1)
trainx.head()
#导入测试集数据
test_data =pd.read_csv("D:\\numpy\\001科研\\测试集建模因素.csv",encoding='utf-8')
print(test_data.shape) #获取目前的形状
print(test_data.columns)
testy=test_data.OUTCOME
testy.head()
testx=test_data.drop('OUTCOME',axis=1)
testx.head()
#训练集数据标准化，建议用StandardScaler标准化连续变量
scaler = StandardScaler()
continuous_columns = ['P_F', 'LAC']  
columns_to_copy = ['OUTCOME','decision_time','Nutritional_Methods','blood_glucose_0_7.8-10','blood_glucose_1_11.1','blood_glucose_2_2.8','mechanical_ventilation']  
scaled_continuous_train = scaler.fit_transform(train_data[continuous_columns]) # 只选择连续变量列进行fit_transform 
scaled_data_train = pd.DataFrame(scaled_continuous_train, columns=continuous_columns)  
scaled_data_train[columns_to_copy] = train_data[columns_to_copy]
trainy=scaled_data_train.OUTCOME
trainy.head()
trainx=scaled_data_train.drop('OUTCOME',axis=1)
trainx.head()
#测试集数据标准化
scaled_continuous_test = scaler.fit_transform(test_data[continuous_columns]) # 只选择连续变量列进行fit_transform 
scaled_data_test = pd.DataFrame(scaled_continuous_test, columns=continuous_columns)    
scaled_data_test[columns_to_copy] = test_data[columns_to_copy]
testy=scaled_data_test.OUTCOME
testy.head()
testx=scaled_data_test.drop('OUTCOME',axis=1)
testx.head()
from sklearn.metrics import roc_auc_score as AUC

rfc=RandomForestClassifier(max_depth= 3, min_samples_leaf=6, min_samples_split=2, n_estimators=160,
                           random_state=1,class_weight = 'balanced').fit(trainx, trainy.astype('int'))

pred_rfc1 = rfc.predict_proba(trainx)
print("AUC_train",AUC(trainy.astype('int'),pred_rfc1[:, 1]))
joblib.dump(rfc, 'rfc12.pkl')
# Load the model
model = joblib.load('rfc12.pkl')        
# Define feature options         
cp_options = {
          
    1: 'Typical angina (1)',
          
    2: 'Atypical angina (2)',
          
    3: 'Non-anginal pain (3)',
          
    4: 'Asymptomatic (4)'
          
}
          

          
restecg_options = {
          
    0: 'Normal (0)',
          
    1: 'ST-T wave abnormality (1)',
          
    2: 'Left ventricular hypertrophy (2)'
          
}
          

          
slope_options = {
          
    1: 'Upsloping (1)',
          
    2: 'Flat (2)',
          
    3: 'Downsloping (3)'
          
}
          

          
thal_options = {
          
    1: 'Normal (1)',
          
    2: 'Fixed defect (2)',
          
    3: 'Reversible defect (3)'
          
}
# Define feature names
feature_names = ['decision_time', 'Nutritional_Methods', 'blood_glucose_0_7.8-10',
       'blood_glucose_1_11.1', 'blood_glucose_2_2.8', 'mechanical_ventilation',
       'P_F', 'LAC']
# Streamlit user interface
st.title("Death Predictor")
decision_time= st.selectbox("decision_time (0=in 6 hour, 1=above 6 hour):", options=[0, 1], format_func=lambda x: 'in 6 hour (0)' if x == 0 else 'above 6 hour (1)')
Nutritional_Methods= st.selectbox("Nutritional_Methods (0=EN, 1=PN):", options=[0, 1], format_func=lambda x: 'EN (0)' if x == 0 else 'PN (1)')
blood_glucose_0= st.selectbox("blood_glucose_0_7.8-10 (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
blood_glucose_1= st.selectbox("blood_glucose_1_11.1 (0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
blood_glucose_2= st.selectbox("blood_glucose_2_2.8(0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
mechanical_ventilation=st.selectbox("mechanical_ventilation(0=NO, 1=YES):", options=[0, 1], format_func=lambda x: 'NO (0)' if x == 0 else 'YES (1)')
P_F = st.number_input("P_F:", min_value=1, max_value=850, value=150)
LAC= st.number_input("LAC:", min_value=1, max_value=35, value=1)
# Process inputs and make predictions
feature_values = [decision_time,Nutritional_Methods,blood_glucose_0,blood_glucose_1,blood_glucose_2,mechanical_ventilation,P_F,LAC ]
features = np.array([feature_values])
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:
          
        advice = (
          
            f"According to our model, you have a high risk of mortality . "
          
            f"The model predicts that your probability of having High mortality risk is {probability:.1f}%. "
          
            "While this is just an estimate, it suggests that you may be at significant risk. "
           )
          
    else:
          
        advice = (
          
            f"According to our model, you have a low mortality risk. "
          
            f"The model predicts that your probability of not having low mortality risk is {probability:.1f}%. "
          
            )
          

          
    st.write(advice)
# Calculate SHAP values and display force plot
import shap
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
shap.force_plot(explainer.expected_value[0], shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
st.image("shap_force_plot.png")