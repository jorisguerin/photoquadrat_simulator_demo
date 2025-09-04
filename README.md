# Benthic Photo-quadrat Sampling Simulator 🪸

An interactive Streamlit application for simulating different 
photo-quadrat sampling strategies to estimate benthic cover proportions 
in marine environments.

**Link to Web App**: https://benthic-sampling-simulator.streamlit.app/

## Overview

This tool helps marine biologists and researchers evaluate 
the effectiveness of different sampling strategies for estimating 
benthic community composition. 
It allows you to compare various quadrat sampling methods and 
assess their accuracy against known ground truth data.

## How to Use

1. Choose your sampling method (Random, Free Transects, Parallel Transects, etc.)
2. Set quadrat size and number of samples in the sidebar
3. Click "Generate Sampling" to create your sampling pattern
4. View the collected quadrats and compare estimated vs. true cover proportions

## Credits

**Code:** Joris Guérin (UMR Espace-Dev/IRD)  
**Map:** Daniele Ventura (Sapienza Università di Roma)

---

### For Developers
To run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`