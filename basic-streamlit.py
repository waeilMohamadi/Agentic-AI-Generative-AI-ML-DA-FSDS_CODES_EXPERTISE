# How to print Hello, World! in Streamlist
import streamlit as st
st.title("Hello, world!")
# how to write text in streamlit
st.write("This is my first Streamlit app!")
# how to create input box in streamlit
name = st.text_input("Name:")
password = st.text_input("Password:")
# how to create a button in strealit
if st.button("Click me"):
    st.write(f"Hello, {name}!")
elif password == "admin":
    st.write(f"Welcome, {password}!")
else:
    st.write("Please enter your name and password.")
# how to create a slider in streamlit
age = st.slider("Age:", 0,100,25)
# how to create a slectbox in streamlit
color = st.selectbox("Favortite color:", ["Red", "Green", "Blue"])
if st.button("Submit"):
    st.write(f"You are {age} years old and your favorite color is {color}.")
# how to create a checkbox in strealit
if st.checkbox("Show more info"):
    st.write("This is a simple streamlit app.")
# how to create a dataframe in streamlit
import pandas as pd
import numpy as np
data = pd.DataFrame(np.random.randn(10,3), columns=list('ABC'))
# how to display a dataframe in streamlit
st.dataframe(data)
# how to create a line chart in streamlit
st.line_chart(data)
# how to create a bar chart in streamlit
st.bar_chart(data)
# how create slider to filter data in strealit
value = st.slider("Filter data by column A:",float(data['A'].min()), float(data['A'].max()), float(data['A'].mean()))
# filter data based on slider value
filtered_data = data[data['A'] >= value]
st.write("filtered data:")
st.dataframe(filtered_data)
# how to create a sidebar in streamlit
st.sidebar.title("Welcome to the sidebar!")
st.sidebar.write("This is the siderbar content.")
st.sidebar.text_input("Mohamadi")
st.sidebar.slider("Random numbers:",0,100,50)
st.sidebar.button("Submit")
st.sidebar.checkbox("Show more options")
st.sidebar.selectbox("Choose an option:", ["DSA", "Web Dev", "Java", "Python"])
# how to create columns in streamlit
col1, col2 = st.columns(2)
with col1:
    st.header("column 1")
    st.write("This is column 1.")
with col2:
    st.header("column 2")
    st.write("This is column 2")
# how to upload a file in streamlit
uploaded_file = st.file_uploader("Upload a file:")
if uploaded_file is not None:
    st.write("File uploaede successfully!")
    st.write(uploaded_file.name)
    st.write(uploaded_file.size)
    st.write(uploaded_file.type)
    # read file content
    file_content = uploaded_file.read()
    st.write(file_content)
# how to display an image in streamlit
from PIL import Image
image = Image.open(r"C:\Users\THINKPAD E15\OneDrive\Desktop\workshop\fahd.jpg")
st.header("This is Waeil's Image")
st.image(image, caption="Fahd Image", width=300)
