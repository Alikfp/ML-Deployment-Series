import streamlit as st
import requests
import json

st.title('HR-analytics App') #title to be shown
st.header('Enter the employee data:') #header to be shown in app

satisfaction_level = st.number_input(
    'satisfaction level',
    min_value=0.00,
    max_value=1.00
    )

last_evaluation = st.number_input(
    'last evaluation score',
    min_value=0.00,
    max_value=1.00
    )

number_project = st.number_input(
    'number of projects',
    min_value=1
    )

average_montly_hours = st.slider(
    'average monthly hours',
    min_value=0,
    max_value=320
    )

time_spend_company = st.number_input(
    label = 'Number of years at company',
    min_value=0
    )

Work_accident = st.selectbox(
    'If met an accident at work',
    [1,0],
    index = 1
    )

promotion_last_5years = st.selectbox(
    'Promotion in last 5 years yes=1/no=0',
    [1,0],
    index=1
    )

department = st.selectbox(
    'Department',
    ['IT', 'RandD', 'accounting', 'hr', 'management',
      'marketing', 'product_mng', 'sales', 'support', 'technical']
    )

salary = st.selectbox(
    'Salary Band',
    ['low', 'medium', 'high']
    )

names = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'department', 'salary']

params = [satisfaction_level, last_evaluation, number_project,
       average_montly_hours, time_spend_company, Work_accident,
       promotion_last_5years, department, salary]

input_data = dict(zip(names, params))

if st.button('Predict'):
    print()
    print(json.dumps(input_data))
    print()
    try:
        output_ = requests.post(url = 'http://localhost:8000/predict', data = json.dumps(input_data))
    except:
       print('Not able to connect to api server')

    print()
    print(output_.json())
    print()
    ans = eval(output_.json())
    output = 'Yes' if ans['prediction']==1 else 'No'

    if output == 'Yes':
        st.success(f"The employee might leave the company with a probability of {(ans['probability'])*100: .2f}")

    if output == 'No':
        st.success(f"The employee might not leave the company with a probability of {(1-ans['probability'])*100: .2f}")