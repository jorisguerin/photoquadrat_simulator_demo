import streamlit as st
import numpy as np
import pandas as pd

from samplers import random_sampling_map, free_transect_sampling_map,\
                      parallel_transect_sampling_map, ND_transect_sampling_map, sample_quadrats_streamlit
from utils import display_map_streamlit, compute_proportions, display_sample_quadrats_streamlit
from parameters import CLASS_PARAMS


# Base parameters

map_path = "data/map.npy"
pixel_size = 0.01

# Set page configuration
st.set_page_config(
    page_title="Benthic Quadrat Sampling Simulator",
    page_icon="🪸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(0deg, #add8e6 0%, #1e3c72 100%);
        padding: 2rem;
        border-radius: 10px;
        color: black;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .sampling-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_map(map_path):
    """Load the benthic map from file"""
    if map_path.endswith('.npy'):
        return np.load(map_path)
    else:
        st.error("Please provide a .npy file with your preprocessed map")
        return None


# === MAIN STREAMLIT APP ===
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1> 🪸 Benthic Photo-quadrat Sampling Simulator</h1>
        <p>Interactive tool for simulating different photo-quadrat sampling strategies to estimate benthic cover proportions</p>
        <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.7;">
        Code by Joris Guérin | UMR Espace-Dev/IRD | joris.guerin@ird.fr
        </p>
        <p style="margin-top: 10px; font-size: 0.85em; opacity: 0.7;">
        Map by Daniele Ventura | Department of Environmental Biology/Sapienza Università di Roma | daniele.ventura@uniroma1.it
        </p>
    </div>
    """, unsafe_allow_html=True)


    # Load map
    try:
        benthic_map = load_map(map_path)
        if benthic_map is None:
            return
    except Exception as e:
        st.error(f"Error loading map: {str(e)}")
        return

    # st.sidebar.success(f"✅ Map loaded: \n {benthic_map.shape[0]} × {benthic_map.shape[1]} pixels")

    # Sampling method selection
    st.sidebar.header("Field Sampling Design")
    sampling_method = st.sidebar.selectbox(
        "Sampling method:",
        ["Random Sampling", "Free Transects", "Parallel Transects", "Non-directional Transects"],
        help="Select the quadrat sampling strategy to use"
    )
    quadrat_size = st.sidebar.number_input("Quadrat Size (m)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    if sampling_method == "Random Sampling":
        n_samples = st.sidebar.number_input("Number of Quadrats", min_value=1, max_value=500, value=50, step=1)
    elif "Transects" in sampling_method:
        n_transects = st.sidebar.number_input("Number of Transects", min_value=1, max_value=50, value=10, step=1)
        quadrats_per_transect = st.sidebar.number_input("Number of Quadrats per Transect", min_value=2, max_value=20, value=5, step=1)
        distance_between_quadrats = st.sidebar.number_input("Distance Between Quadrats (m)", min_value=0.5, max_value=20.0,
                                                    value=2.0, step=0.1)
        if "Parallel" in sampling_method:
            distance_between_transects = st.sidebar.number_input("Distance Between transects (m)", min_value=0.5,
                                                                max_value=20.0,
                                                                value=2.0, step=0.1)
    st.sidebar.header("Quadrat Annotation")

    annotation_mode = st.sidebar.selectbox(
        "Points per Quadrat:",
        ["All points", "Custom number"],
        help="Choose whether to use all points in each quadrat or specify a number"
    )
    if annotation_mode is "Custom number":
        points_per_quadrat = st.sidebar.number_input("Number of Points per Quadrat",
                                                     min_value=1, max_value=500, value=10, step=1)
    else:
        points_per_quadrat = None

    sample_points = None
    samples = None
    # Generate sampling button
    if st.sidebar.button("Generate Sampling", type="primary"):
        with st.spinner("Generating sampling pattern..."):
            try:
                if sampling_method == "Random Sampling":
                    sample_points, samples = random_sampling_map(benthic_map, n_samples, quadrat_size)

                elif sampling_method == "Free Transects":
                    sample_points, samples = free_transect_sampling_map(benthic_map, n_transects,
                                                                        quadrats_per_transect,
                                                                        distance_between_quadrats)

                elif sampling_method == "Non-directional Transects":
                    sample_points, samples = ND_transect_sampling_map(benthic_map, n_transects,
                                                                      quadrats_per_transect,
                                                                      distance_between_quadrats)

                else:
                    sample_points, samples = parallel_transect_sampling_map(benthic_map, n_transects,
                                                                            quadrats_per_transect,
                                                                            distance_between_quadrats,
                                                                            distance_between_transects)

            except Exception as e:
                st.error(f"Error generating samples: {str(e)}")

    # Main content area
    st.subheader("Benthic Cover Map")

    fig = display_map_streamlit(benthic_map, CLASS_PARAMS, fast_display=True, sample_points=sample_points)
    st.pyplot(fig)

    st.subheader("Display Collected Quadrats")

    if points_per_quadrat is not None:
        points, points_locations = sample_quadrats_streamlit(samples, points_per_quadrat)
    else:
        points, points_locations = None, None

    sample_fig = display_sample_quadrats_streamlit(samples, CLASS_PARAMS, max_samples=24, points=points_locations)

    if sample_fig is not None:
        st.pyplot(sample_fig)

        # Show info about samples
        n_total = 0
        if samples is not None:
            n_total = len(samples)
        n_shown = min(24, n_total)
        if n_total > 24:
            st.caption(f"Showing first {n_shown} of {n_total} samples")
        else:
            st.caption(f"Showing all {n_shown} samples")

        if annotation_mode is "All points":
            st.caption("All pixels are used to compute cover proportions")
        else:
            st.caption("Only points corresponding to red crosses are used to compute cover proportions")



    st.subheader("📊 Results")

    classes = [CLASS_PARAMS[i]['name'] for i in range(len(CLASS_PARAMS))]
    true_proportions = compute_proportions(benthic_map, n_classes=len(classes) - 1)

    if points is not None:
        proportions = compute_proportions(points)
    elif samples is not None:
        proportions = compute_proportions(samples)
    else:
        proportions = None

    # Create results table
    results_data = []
    for i in range(len(true_proportions)):
        class_id = i + 1
        if class_id in CLASS_PARAMS:
            true_pct = true_proportions[i] * 100

            if proportions is not None:
                estimated_pct = proportions[i] * 100
                error_pct = estimated_pct - true_pct
            else:
                estimated_pct = None
                error_pct = None

            # Only include classes with some presence
            if true_pct > 0.01:
                results_data.append({
                    'Benthic Class': CLASS_PARAMS[class_id]['name'],
                    'True Cover (%)': f"{true_pct:.2f}",
                    'Estimated Cover (%)': f"{estimated_pct:.2f}" if estimated_pct is not None else "—",
                    'Error (%)': f"{error_pct:+.2f}" if error_pct is not None else "—"
                })

    if results_data:
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, hide_index=True, use_container_width=True)
    else:
        st.info("Generate sampling to see results")


if __name__ == "__main__":
    main()