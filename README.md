# Udacity Project Machine Learning - Enron Mail Dataset

# Introduction
The goal of the project is to study a dataset of emails of a company called Enron. This company went bankrupt after a the fraud was detected.

"Enron's complex financial statements were confusing to shareholders and analysts. In addition, its complex business model and unethical practices required that the company use accounting limitations to misrepresent earnings and modify the balance sheet to indicate favorable performance." source: https://en.wikipedia.org/wiki/Enron_scandal

### Script overview

Note that 2 scripts where run with python2, since the udacity input code to convert xml to csv and to set up the sql database is written in python 2. The assessment is done with python3, since it is my preferred version. I would recommend udacity to change the input scripts to python3.

List of scripts within the project:
- sample_py2.py -> down sizing of .osm file (python2)
- 01_unique_tags.py -> count unique tags (python3)
- 02_problematic_tags.py -> list problematic tags (python3)
- 03_audit_streetname.py -> audit street names (python3)
- 04_audit-Postcode.py -> audit post codes (python3)
- 11_data_2_csv_py2.ipynb -> convert xml data to csv files (python2)
- 13_build_sql_py2.ipynb -> set-up sql database (python2)
- 15_query_py3.ipynb -> SQL queries of the choosen data set (python3)

### References:
- https://medium.com/@williamkoehrsen/machine-learning-with-python-on-the-enron-dataset-8d71015be26d
- Udacity course input

![Screenshot](pics/osm_area.png)
