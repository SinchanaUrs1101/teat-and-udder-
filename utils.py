import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def display_prediction_probabilities(condition_probs):
    """
    Display the prediction probabilities as a horizontal bar chart.
    Now shows REAL model probabilities with confidence indicators.
    """
    # Sort conditions by probability (descending) - using REAL probabilities
    sorted_probs = dict(sorted(condition_probs.items(), key=lambda item: item[1], reverse=True))

    # Create lists of conditions and probabilities
    conditions = list(sorted_probs.keys())
    probabilities = list(sorted_probs.values())  # REAL probabilities

    # Add confidence assessment
    max_prob = max(condition_probs.values())
    if max_prob > 0.8:
        confidence = "High Confidence"
        confidence_color = 'green'
    elif max_prob > 0.6:
        confidence = "Medium Confidence" 
        confidence_color = 'orange'
    else:
        confidence = "Low Confidence"
        confidence_color = 'red'
    
    # Display confidence indicator
    st.markdown(f"**Model Confidence: <span style='color:{confidence_color}'>{confidence}</span>** (Max: {max_prob:.1%})", unsafe_allow_html=True)

    # Display as a bar chart with REAL probabilities
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(conditions, probabilities, color='skyblue')

    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width + 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.1%}',
                va='center', fontsize=10)

    # Add titles and labels
    ax.set_title('Real Model Probability Distribution')
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1.1)

    # Highlight the highest probability bar
    bars[0].set_color('lightcoral')

    st.pyplot(fig)

def display_treatment_info(condition, treatment_info):
    """
    Display treatment information for the detected condition.

    Args:
        condition: The detected condition name
        treatment_info: Dictionary containing treatment information for each condition
    """
    if condition in treatment_info:
        st.markdown("### Recommended Treatment")

        # Create tabs for different aspects of treatment
        immediate, management, prevention = st.tabs(["Immediate Actions", "Management", "Prevention"])

        with immediate:
            st.markdown("#### Immediate Actions")
            st.markdown(treatment_info[condition]["immediate"])

        with management:
            st.markdown("#### Management Practices")
            st.markdown(treatment_info[condition]["management"])

        with prevention:
            st.markdown("#### Prevention")
            st.markdown(treatment_info[condition]["prevention"])

        # Display veterinary consultation note
        st.warning("**Note**: Always consult with a veterinarian for proper diagnosis and treatment plan.")

