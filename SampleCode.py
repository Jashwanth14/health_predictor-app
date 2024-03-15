import streamlit as st
import pickle

# Function to predict diabetes
def predict_diabetes():
    st.title('Diabetes Prediction using ML')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input('Number of Pregnancies')
        glucose = st.number_input('Glucose Level')
        blood_pressure = st.number_input('Blood Pressure value')

    with col2:
        skin_thickness = st.number_input('Skin Thickness value')
        insulin = st.number_input('Insulin Level')
        bmi = st.number_input('BMI value')

    with col3:
        diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function value')
        age = st.number_input('Age of the Person')
    
    if st.button('Diabetes Test Result'):
        # Check if any input field is empty
        if any([pregnancies == '', glucose == '', blood_pressure == '', skin_thickness == '', insulin == '', bmi == '', diabetes_pedigree_function == '', age == '']):
            st.warning("Please fill in all the fields.")
        else:
            diab_prediction = diabetes_model.predict([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
            if diab_prediction[0] == 1:
                st.success('The person is diabetic')
            else:
                st.success('The person is not diabetic')

# Function to predict heart disease
def predict_heart_disease():
    st.title('Heart Disease Prediction using ML')
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input('Age')
        sex = st.number_input('Sex')
        cp = st.number_input('Chest Pain types')
        trestbps = st.number_input('Resting Blood Pressure')
        chol = st.number_input('Serum Cholestoral in mg/dl')
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
    
    with col2:
        restecg = st.number_input('Resting Electrocardiographic results')
        thalach = st.number_input('Maximum Heart Rate achieved')
        exang = st.number_input('Exercise Induced Angina')
        oldpeak = st.number_input('ST depression induced by exercise')
        slope = st.number_input('Slope of the peak exercise ST segment')
        ca = st.number_input('Major vessels colored by flourosopy')
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
    if st.button('Heart Disease Test Result'):
        # Check if any input field is empty
        if any([age == '', sex == '', cp == '', trestbps == '', chol == '', fbs == '', restecg == '', thalach == '', exang == '', oldpeak == '', slope == '', ca == '', thal == '']):
            st.warning("Please fill in all the fields.")
        else:
            heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            if heart_prediction[0] == 1:
                st.success('The person is having heart disease')
            else:
                st.success('The person does not have any heart disease')

# Function to predict Parkinson's disease
def predict_parkinsons():
    st.title("Parkinson's Disease Prediction using ML")
    col1, col2 = st.columns(2)
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)')
        fhi = st.number_input('MDVP:Fhi(Hz)')
        flo = st.number_input('MDVP:Flo(Hz)')
        jitter_percent = st.number_input('MDVP:Jitter(%)')
        jitter_abs = st.number_input('MDVP:Jitter(Abs)')
        rap = st.number_input('MDVP:RAP')
        ppq = st.number_input('MDVP:PPQ')
        ddp = st.number_input('Jitter:DDP')
    
    with col2:
        shimmer = st.number_input('MDVP:Shimmer')
        shimmer_db = st.number_input('MDVP:Shimmer(dB)')
        apq3 = st.number_input('Shimmer:APQ3')
        apq5 = st.number_input('Shimmer:APQ5')
        apq = st.number_input('MDVP:APQ')
        dda = st.number_input('Shimmer:DDA')
        nhr = st.number_input('NHR')
        hnr = st.number_input('HNR')
        rpde = st.number_input('RPDE')
    
    col3, _ = st.columns(2)
    with col3:
        dfa = st.number_input('DFA')
        spread1 = st.number_input('spread1')
        spread2 = st.number_input('spread2')
        d2 = st.number_input('D2')
        ppe = st.number_input('PPE')
    
    if st.button("Parkinson's Test Result"):
        # Check if any input field is empty
        if any([fo == '', fhi == '', flo == '', jitter_percent == '', jitter_abs == '', rap == '', ppq == '', ddp == '', shimmer == '', shimmer_db == '', apq3 == '', apq5 == '', apq == '', dda == '', nhr == '', hnr == '', rpde == '', dfa == '', spread1 == '', spread2 == '', d2 == '', ppe == '']):
            st.warning("Please fill in all the fields.")
        else:
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
            if parkinsons_prediction[0] == 1:
                st.success("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")

# Function to handle navigation between pages
def navigate_to_prediction(selected_prediction):
    if selected_prediction == 'Diabetes Prediction':
        predict_diabetes()
    elif selected_prediction == 'Heart Disease Prediction':
        predict_heart_disease()
    elif selected_prediction == "Parkinson's Prediction":
        predict_parkinsons()

# Display title with slow pop-up effect and custom color
st.title("Multiple Disease Prediction System")

# Apply CSS for slow pop-up effect and custom color
st.markdown(
    """
    <style>
        /* CSS animation for slow pop-up effect */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        /* Apply animation to title */
        h1 {
            animation: fadeIn 2s;
            color: #9F4E3B; /* Custom color */
            font-size: 60px; /* Increase font size */
            text-align: center; /* Center align the title */
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Check if the "Start Prediction" button has been clicked
if 'prediction_started' not in st.session_state:
    st.session_state['prediction_started'] = False

# Show "Start Prediction" button if it hasn't been clicked yet
if not st.session_state['prediction_started']:
    if st.button("Start Prediction"):
        st.session_state['prediction_started'] = True

# Load the saved models
diabetes_model = pickle.load(open("diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("heart_disease_model.sav",'rb'))
parkinsons_model = pickle.load(open("parkinsons_model.sav", 'rb'))

# If "Start Prediction" button has been clicked, show disease prediction options
if st.session_state['prediction_started']:
    # Create a selectbox for navigation
    selected = st.selectbox('', ['Select the required prediction', 'Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"], key='navigation')

    # Perform action based on user selection
    if selected != 'Select the required prediction':
        # Render the selected prediction page
        navigate_to_prediction(selected)
