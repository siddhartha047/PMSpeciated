speciate08132023.csv  - USEPA SPECIATE database exported to csv   
https://www.epa.gov/air-emissions-modeling/speciate

improve_params.csv - Species (features) in IMPROVE network collected PM2.5 samples
https://vista.cira.colostate.edu/Improve/improve-program/

profile_assigned_forSPECIATEdata.csv - Saikat's manual designation of source categories for USEPA SPECIATE profiles.
grouped SPECIATE source profiles to common source categries


SpeciateML_v1.py  - Python script to read and process data; and then develop a KNN model on SPECIATE database

usepa_final_with_assigned_profile.csv  - process and merged SPECIATE database with manual sopurce categories and columns selected with IMPROVE species (features)
Processed and written by python script "SpeciateML_v1.py"

PMF_PudgetSound2_7Factors.csv  - PMF output (time averaged for each species) with 7 factors - used to test KNN model

