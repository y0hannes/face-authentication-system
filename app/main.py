import streamlit as st

st.title("AI Face Login System")
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Register", "Login"])

if page == "Register":
    from app.pages import register
    register.show()
elif page == "Login":
    from app.pages import login
    login.show()