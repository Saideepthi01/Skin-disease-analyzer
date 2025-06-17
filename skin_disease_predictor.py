import streamlit as st
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Title
st.title("AI-Based Skin Disease Diagnosis System")
st.write("Select your symptoms ")

# Load and preprocess dataset
@st.cache_data
def load_and_train_model():
    df = pd.read_csv('skin_disease.csv')

    # Fill NaNs and process symptoms
    symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]
    df[symptom_cols] = df[symptom_cols].fillna("")
    df['All_Symptoms'] = df[symptom_cols].values.tolist()
    df['All_Symptoms'] = df['All_Symptoms'].apply(lambda x: [s.lower().strip() for s in x if s])

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['All_Symptoms'])
    y = df['Disease']
    
    X_train,X_test ,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return df, mlb, model, accuracy

# Load model and data
df, mlb, model, accuracy = load_and_train_model()

st.sidebar.subheader("How to use:")
st.sidebar.write("1. Select the symptoms you are experiencing.")
st.sidebar.write("2. Click on 'Predict Disease' to get the predicted disease and its description.")
st.sidebar.write("3. Precautions will also be provided.")
st.sidebar.write("4. You can search for more information on Google.")
st.sidebar.write("This model is trained on a dataset of skin diseases and their symptoms.")
st.sidebar.subheader("Note:")
st.sidebar.write("This is a basic model and should not be used as a substitute for professional medical advice.")
st.sidebar.write("Always consult a healthcare professional for accurate diagnosis and treatment.")
st.sidebar.write("### Model Accuracy")
st.sidebar.write("The model has an accuracy of {:.2f}%".format(accuracy * 100))

# Get all unique symptoms
all_symptoms = sorted(mlb.classes_)

# User symptom selection
selected_symptoms = st.multiselect("Select Symptoms:", all_symptoms)

# Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Prediction
        input_vector = mlb.transform([selected_symptoms])
        prediction = model.predict(input_vector)[0]

        # Get additional info
        info = df[df['Disease'] == prediction].iloc[0]
        description = info['Description']
        precautions = [info.get(f'Precaution_{i}', '') for i in range(1, 5)]
        precautions = [p for p in precautions if p]

        # Display results
        st.subheader("Predicted Disease:")
        st.success(prediction)

        if description == "":
            st.warning("No description available for this disease.")
        else:
            st.subheader("Description:")
            st.write(description)

        if not precautions[0] == "nan":
            st.warning("No precautions available for this disease.")
        else:
            st.subheader("Precautions:")
            for i, prec in enumerate(precautions, 1):
                st.write(f"**{i}.** {prec}")

        search_query = prediction.replace(" ", "+")
        google_url = f"https://www.google.com/search?q={search_query}+skin+disease"
        st.markdown(f"[Search on GOOGLE for more info]({google_url})", unsafe_allow_html=True)