SCREENSHOT:
<img width="1360" height="768" alt="image" src="https://github.com/user-attachments/assets/0822e482-2212-42ca-928d-98cfabe05bd1" />

INSTRUCTIONS:
1. Set base path in all 4 codes. Datasets can be found in the data folder.
2. Run datasetcreationformodel.py
-check a new file of 2700 builds named scenario_pc_builds to be created in the same directory.(adjust path)
-THIS STEP CAN BE SKIPPED, I HAVE ALREADY GIVEN THIS FILE OTHERWISE IT TAKES AROUND 45 MINS TO GENERATE THIS FILE.
3. Run randomforestmodel.py
-generates 5 .pkl files based on separate scenarios.(adjust path)
4. Run backend.py to ensure everything is in place.
5. Run app.py

Create builds as per your need!

pip install Pillow requests matplotlib reportlab numpy pandas
pip install pandas numpy joblib
pip install scikit-learn
