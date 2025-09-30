# app.py
import streamlit as st
import attendive
import career
import summary

def main():
    st.set_page_config(page_title="Smart Classroom", layout="wide")
    st.title("🎓 Smart Classroom")

    st.sidebar.title("📌 Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Attentiveness Tracker", "Career Path Generator", "Content Summarizer"]
    )

    if page == "Attentiveness Tracker":
        attendive.main()
    elif page == "Career Path Generator":
        career.main()
    elif page == "Content Summarizer":
        summary.main()

if __name__ == "__main__":
    main()


#_______________________________________________________________________________________________


# import streamlit as st
# from auth import login_page, register_page
# from home import teacher_home, student_home

# def main():
#     st.set_page_config(page_title="Drona - Smart Classroom", layout="wide")
#     st.title("🌟 DRONA - Smart Classroom")

#     # Initialize session_state variables
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False
#         st.session_state.role = None

#     if not st.session_state.logged_in:
#         # Landing page with role selection
#         choice = st.radio("Choose an option:", ["Login", "Register"], horizontal=True)
        
#         if choice == "Login":
#             role = login_page()
#         else:
#             role = register_page()
        
#         if role:  # Successful login/register
#             st.session_state.logged_in = True
#             st.session_state.role = role
#             st.experimental_rerun()  # Rerun so sidebar appears correctly
#     else:
#         # User is logged in, show appropriate home page
#         if st.session_state.role == "Teacher":
#             teacher_home()
#         elif st.session_state.role == "Student":
#             student_home()

# if __name__ == "__main__":
#     main()






