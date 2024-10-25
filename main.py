import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from  openai import OpenAI

#openai.base_url="https://api.groq.com/openai/v1"
#openai.api_key = os.environ["GROQ_API_KEY"]


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"]
)
# Loadiong the Models
def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)

xgboost_model = load_model('xgb_model.pkl')
naive_bayes_model = load_model('nb_model.pkl')
random_forest_model = load_model('rt_model.pkl')
decision_tree_model = load_model('dt_model.pkl')
svm_model = load_model('svm_model.pkl')
knn_model = load_model('knn_model.pkl')
voting_classifier_model = load_model('voting_clf.pkl')
xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')
xgboost_featureEngineered_model = load_model('xgboost-featureEngineered.pkl')

# Preparing the inut data for the attributes
def prepare_input(credit_score, location, gender, age, tenure, balance, 
      num_products, has_credit_card, is_active_member, estimated_salary):

    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
        }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


def make_predictions(input_df, input_dict):
    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    }

    avg_probability = np.mean(list(probabilities.values()))

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob}")

    st.write(f"Average Probability: {avg_probability}")

    return avg_probability

def explain_prediction(probability, input_dict, surname):
    # Create a prompt to generate the explanation
    prompt = f"""
    You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided below.

    Here is the customer's information:
    {input_dict}

    Here are the machine learning model's top 10 most important features for predicting churn:

    --------------------------------------
    | Feature              | Importance |
    --------------------------------------
    | NumOfProducts        | 0.323888   |
    | IsActiveMember       | 0.164146   |
    | Age                  | 0.109550   |
    | Geography_Germany    | 0.091373   |
    | Balance              | 0.052786   |
    | Geography_France     | 0.046463   |
    | Gender_Female        | 0.045283   |
    | Geography_Spain      | 0.036855   |
    | CreditScore          | 0.035005   |
    | EstimatedSalary      | 0.032655   |
    --------------------------------------

    Here are summary statistics for churned customers:
    {df[df['Exited'] == 1].describe()}

    Here are summary statistics for non-churned customers:
    {df[df['Exited'] == 0].describe()}

    - If the customer has over a 40% risk of churning, generate a 3-sentence explanation of why they are at risk of churning.
    - If the customer has less than a 40% risk of churning, generate a 3-sentence explanation of why they might not be at risk of churning.
    - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.

    Don't mention the probability of churning, or the machine learning model, or say anything like "Based on the machine learning model's prediction and top 10 most important features," just explain the prediction.
    """

    # Debug: Display the prompt
    print("EXPLANATION PROMPT:", prompt)

    # Generate the explanation using the language model
    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    # Return the content of the generated response
    return raw_response.choices[0].message.content


# Title of the app
st.title("Customer Churn Prediction")

# Load the data
df = pd.read_csv('churn.csv')

# Create a list of customers for selection
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

# Customer selection dropdown
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    # Extract selected customer ID and surname
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]

    # Get details of the selected customer
    selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        # Input for credit score
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,  # Adjusted max value to typical credit score range
            value=int(selected_customer['CreditScore'])
        )

        # Dropdown for location
        location = st.selectbox(
            "Location", ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(selected_customer['Geography'])
        )

        # Radio buttons for gender
        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == "Male" else 1
        )

        # Input for age
        age = st.number_input(
            "Age", 
            min_value=18, 
            max_value=80, 
            value=int(selected_customer['Age'])
        )

        # Input for tenure
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer['Tenure'])
        )

    with col2:

      balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=float(selected_customer['Balance']))

      num_products = st.number_input(
        "Number of Products",
        min_value=1,
        max_value=10,
        value=int(selected_customer['NumOfProducts']))

      has_credit_card = st.checkbox(
        "Has Credit Card",
        value=bool(selected_customer['HasCrCard']))

      is_active_member = st.checkbox(
        "Is Active Member",
        value=bool(selected_customer['IsActiveMember']))

      estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)

    avg_probability = make_predictions(input_df, input_dict)

        # Explanation
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

    st.markdown("---")

    st.subheader("Explanation of Prediction")

    st.markdown(explanation)
