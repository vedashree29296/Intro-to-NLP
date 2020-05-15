import streamlit as st
import json
import pandas as pd

with open("data.json", "r") as f:
    data = json.load(f)
df = pd.DataFrame(data)
st.title("What Next?")

st.header("Predictive Modelling in Data Science")

st.markdown("#### Consider a simple example: \n\n")
st.table(df)

st.header("General Steps in Data Modelling")
st.markdown("#### What can be the target variable here ? ")

st.write(
    "A common start to modelling is to define what output do you want to predict unless its already specified.\n This is called as the Target variable \n"
)

st.markdown("#### What can be the feature variables here ? ")

st.write(
    "\n The next step is to identify the most important variables that affect the target variable. These are known as Feature variables\n "
)

st.markdown("#### What feature variables can be encoded here?")

st.write(
    "Once these are identified, a preliminary data normalisation is carried out where the categorical data is encoded to numerical data."
)

st.markdown("#### What algorithm to use ?")
st.write(
    "Next, depending on the nature of the data, a proper algorithm, or a set of algorithms are chosen for modelling"
)
st.image("Algorithms.png")
st.markdown("#### How did the Model perform?")
st.write(
    "Finally, We need to check how accurate the selected model is. Based on this, the model is can be either chosen,discarded, or tweaked for optimum performance"
)
st.image("TrainTest.png")

st.subheader("And then we move towards Machine Learning....")
