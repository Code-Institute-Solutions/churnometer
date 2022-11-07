import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
        f"* We suspect customers are churning with low tenure levels: Correct. "
        f"The correlation study at Churned Customer Study supports that. \n\n"

        f"* A customer survey showed our customers appreciate fibre Optic. "
        f"A churned user typically has Fibre Optic, as demonstrated by a Churned Customer Study. "
        f"This insight will be used by the survey team for further discussions and investigations."
    )
