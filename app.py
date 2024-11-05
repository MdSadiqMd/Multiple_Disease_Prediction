import streamlit as st # type: ignore
import pickle
import os
from streamlit_option_menu import option_menu # type: ignore
import plotly.graph_objects as go # type: ignore
import pandas as pd # type: ignore

# Initialize session state for storing test history
if 'diabetes_history' not in st.session_state:
    st.session_state.diabetes_history = []
if 'heart_history' not in st.session_state:
    st.session_state.heart_history = []
if 'kidney_history' not in st.session_state:
    st.session_state.kidney_history = []

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide", page_icon="👨‍🦰🤶")

working_dir = os.path.dirname(os.path.abspath(__file__))

# Load saved models
try:
    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes.pkl', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart.pkl', 'rb'))
    kidney_disease_model = pickle.load(open(f'{working_dir}/saved_models/kidney.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Function to create comparison graphs
def create_comparison_graphs(current_values, history, metrics, title):
    if not history:
        return
    
    fig = go.Figure()
    
    prev_test = history[-1]
    fig.add_trace(go.Bar(
        name='Previous Test',
        x=metrics,
        y=[prev_test.get(metric, 0) for metric in metrics],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Current Test',
        x=metrics,
        y=[float(current_values.get(metric, 0)) for metric in metrics],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title=f'{title} - Comparison of Test Results',
        barmode='group',
        yaxis_title='Values',
        xaxis_title='Metrics'
    )
    
    return fig

# Sidebar for navigation
with st.sidebar:
    selected = option_menu("Multiple Disease Prediction", 
                ['Diabetes Prediction',
                 'Heart Disease Prediction',
                 'Kidney Disease Prediction'],
                 menu_icon='hospital-fill',
                 icons=['activity','heart', 'person'],
                 default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction Using Machine Learning")
    
    col1, col2, col3 = st.columns(3)
    current_values = {}

    with col1:
        current_values['Pregnancies'] = Pregnancies = st.text_input("Number of Pregnancies")
    with col2:
        current_values['Glucose'] = Glucose = st.text_input("Glucose Level")
    with col3:
        current_values['BloodPressure'] = BloodPressure = st.text_input("BloodPressure Value")
    with col1:
        current_values['SkinThickness'] = SkinThickness = st.text_input("SkinThickness Value")
    with col2:
        current_values['Insulin'] = Insulin = st.text_input("Insulin Value")
    with col3:
        current_values['BMI'] = BMI = st.text_input("BMI Value")
    with col1:
        current_values['DiabetesPedigreeFunction'] = DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction Value")
    with col2:
        current_values['Age'] = Age = st.text_input("Age")

    # Code for BMI categories
    NewBMI_Overweight = NewBMI_Underweight = NewBMI_Obesity_1 = NewBMI_Obesity_2 = NewBMI_Obesity_3 = 0
    NewInsulinScore_Normal = NewGlucose_Low = NewGlucose_Normal = NewGlucose_Overweight = NewGlucose_Secret = 0

    diabetes_result = ""
    
    if st.button("Diabetes Test Result"):
        # BMI Classification
        if float(BMI) <= 18.5:
            NewBMI_Underweight = 1
        elif 18.5 < float(BMI) <= 24.9:
            pass
        elif 24.9 < float(BMI) <= 29.9:
            NewBMI_Overweight = 1
        elif 29.9 < float(BMI) <= 34.9:
            NewBMI_Obesity_1 = 1
        elif 34.9 < float(BMI) <= 39.9:
            NewBMI_Obesity_2 = 1
        elif float(BMI) > 39.9:
            NewBMI_Obesity_3 = 1
        
        # Insulin Classification
        if 16 <= float(Insulin) <= 166:
            NewInsulinScore_Normal = 1

        # Glucose Classification
        if float(Glucose) <= 70:
            NewGlucose_Low = 1
        elif 70 < float(Glucose) <= 99:
            NewGlucose_Normal = 1
        elif 99 < float(Glucose) <= 126:
            NewGlucose_Overweight = 1
        elif float(Glucose) > 126:
            NewGlucose_Secret = 1

        # Make prediction
        user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness), float(Insulin),
                     float(BMI), float(DiabetesPedigreeFunction), float(Age), NewBMI_Underweight,
                     NewBMI_Overweight, NewBMI_Obesity_1, NewBMI_Obesity_2, 
                     NewBMI_Obesity_3, NewInsulinScore_Normal, NewGlucose_Low,
                     NewGlucose_Normal, NewGlucose_Overweight, NewGlucose_Secret]
        
        prediction = diabetes_model.predict([user_input])
        
        diabetes_result = "The person has diabetes" if prediction[0] == 1 else "The person does not have diabetes"
        
        # Store test result
        test_result = {
            'Pregnancies': float(Pregnancies),
            'Glucose': float(Glucose),
            'BloodPressure': float(BloodPressure),
            'BMI': float(BMI),
            'Age': float(Age),
            'Result': diabetes_result
        }
        st.session_state.diabetes_history.append(test_result)
        
        # Display results and graphs
        st.success(diabetes_result)
        
        if len(st.session_state.diabetes_history) > 1:
            metrics = ['Glucose', 'BloodPressure', 'BMI', 'Age']
            comparison_fig = create_comparison_graphs(current_values, st.session_state.diabetes_history, metrics, 'Diabetes')
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            st.subheader("Trend Analysis")
            metrics_df = pd.DataFrame(st.session_state.diabetes_history)
            st.line_chart(metrics_df[metrics])
            
            st.subheader("Test History")
            st.dataframe(metrics_df)
        else:
            st.info("Complete another test to see comparison graphs and trends.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction Using Machine Learning")
    col1, col2, col3 = st.columns(3)
    current_values = {}

    with col1:
        current_values['age'] = age = st.text_input("Age")
    with col2:
        current_values['sex'] = sex = st.text_input("Sex")
    with col3:
        current_values['cp'] = cp = st.text_input("Chest Pain Types")
    with col1:
        current_values['trestbps'] = trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        current_values['chol'] = chol = st.text_input("Serum Cholesterol in mg/dl")
    with col3:
        current_values['fbs'] = fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        current_values['restecg'] = restecg = st.text_input('Resting Electrocardiographic results')
    with col2:
        current_values['thalach'] = thalach = st.text_input('Maximum Heart Rate achieved')
    with col3:
        current_values['exang'] = exang = st.text_input('Exercise Induced Angina')
    with col1:
        current_values['oldpeak'] = oldpeak = st.text_input('ST depression induced by exercise')
    with col2:
        current_values['slope'] = slope = st.text_input('Slope of the peak exercise ST segment')
    with col3:
        current_values['ca'] = ca = st.text_input('Major vessels colored by fluoroscopy')
    with col1:
        current_values['thal'] = thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect')

    heart_disease_result = ""
    
    if st.button("Heart Disease Test Result"):
        user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
        prediction = heart_disease_model.predict([user_input])
        
        heart_disease_result = "This person has heart disease" if prediction[0] == 1 else "This person does not have heart disease"
        
        test_result = {
            'Age': float(age),
            'Cholesterol': float(chol),
            'Blood Pressure': float(trestbps),
            'Max HR': float(thalach),
            'Result': heart_disease_result
        }
        st.session_state.heart_history.append(test_result)
        
        st.success(heart_disease_result)
        
        if len(st.session_state.heart_history) > 1:
            metrics = ['Cholesterol', 'Blood Pressure', 'Max HR', 'Age']
            comparison_fig = create_comparison_graphs(current_values, st.session_state.heart_history, metrics, 'Heart Disease')
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            st.subheader("Trend Analysis")
            metrics_df = pd.DataFrame(st.session_state.heart_history)
            st.line_chart(metrics_df[metrics])
            
            st.subheader("Test History")
            st.dataframe(metrics_df)
        else:
            st.info("Complete another test to see comparison graphs and trends.")

# Kidney Disease Prediction Page
if selected == 'Kidney Disease Prediction':
    st.title("Kidney Disease Prediction Using Machine Learning")
    
    col1, col2, col3 = st.columns(3)
    current_values = {}

    with col1:
        current_values['age'] = age = st.text_input('Age')
    with col2:
        current_values['blood_pressure'] = blood_pressure = st.text_input('Blood Pressure')
    with col3:
        current_values['specific_gravity'] = specific_gravity = st.text_input('Specific Gravity')
    with col1:
        current_values['albumin'] = albumin = st.text_input('Albumin')
    with col2:
        current_values['sugar'] = sugar = st.text_input('Sugar')
    with col3:
        current_values['red_blood_cells'] = red_blood_cells = st.text_input('Red Blood Cells')
    with col1:
        current_values['pus_cell'] = pus_cell = st.text_input('Pus Cell')
    with col2:
        current_values['serum_creatinine'] = serum_creatinine = st.text_input('Serum Creatinine')
    with col3:
        current_values['blood_urea'] = blood_urea = st.text_input('Blood Urea')
    with col1:
        current_values['hemoglobin'] = hemoglobin = st.text_input('Hemoglobin')
    with col2:
        current_values['diabetes_mellitus'] = diabetes_mellitus = st.text_input('Diabetes Mellitus')
    with col3:
        current_values['coronary_artery_disease'] = coronary_artery_disease = st.text_input('Coronary Artery Disease')
    with col1:
        current_values['appetite'] = appetite = st.text_input('Appetite')
    with col2:
        current_values['pedal_edema'] = pedal_edema = st.text_input('Pedal Edema')
    with col3:
        current_values['anemia'] = anemia = st.text_input('Anemia')

    kidney_disease_result = ""
    
    if st.button("Kidney Disease Test Result"):
        user_input = [float(age), float(blood_pressure), float(specific_gravity), float(albumin), float(sugar), float(red_blood_cells), 
                     float(pus_cell), float(serum_creatinine), float(blood_urea), float(hemoglobin), float(diabetes_mellitus), 
                     float(coronary_artery_disease), float(appetite), float(pedal_edema), float(anemia)]
        prediction = kidney_disease_model.predict([user_input])
        
        kidney_disease_result = "This person has kidney disease" if prediction[0] == 1 else "This person does not have kidney disease"
        
        test_result = {
            'Age': float(age),
            'Blood Pressure': float(blood_pressure),
            'Specific Gravity': float(specific_gravity),
            'Albumin': float(albumin),
            'Blood Urea': float(blood_urea),
            'Result': kidney_disease_result
        }
        st.session_state.kidney_history.append(test_result)
        
        st.success(kidney_disease_result)
        
        if len(st.session_state.kidney_history) > 1:
            metrics = ['Blood Pressure', 'Specific Gravity', 'Albumin', 'Blood Urea', 'Age']
            comparison_fig = create_comparison_graphs(current_values, st.session_state.kidney_history, metrics, 'Kidney Disease')
            st.plotly_chart(comparison_fig, use_container_width=True)
            
            st.subheader("Trend Analysis")
            metrics_df = pd.DataFrame(st.session_state.kidney_history)
            st.line_chart(metrics_df[metrics])
            
            st.subheader("Test History")
            st.dataframe(metrics_df)
        else:
            st.info("Complete another test to see comparison graphs and trends.")