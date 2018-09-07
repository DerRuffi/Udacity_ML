# Udacity Project Machine Learning - Enron Mail Dataset

## Introduction
The goal of the project is to study a dataset of emails of a company called Enron. This company went bankrupt after a the fraud was detected.

"Enron's complex financial statements were confusing to shareholders and analysts. In addition, its complex business model and unethical practices required that the company use accounting limitations to misrepresent earnings and modify the balance sheet to indicate favorable performance." source: https://en.wikipedia.org/wiki/Enron_scandal

### Project goal
The goals of the project are:
- Get familar with dataset, clean dataset, check dataset for outliers
- Visualize dependencies of features
- Create new features (feature engineering)
- Set up ML model to indentify if a person was a "POI" or not
- Select best features to achieve a good presicion and recall score

### Script overview

The script "poi_id.py" and the created .pkl files are located at the sub-folder "final_project".
The project evaluator will test these using the tester.py script.


### References:
- https://medium.com/@williamkoehrsen/machine-learning-with-python-on-the-enron-dataset-8d71015be26d
- https://stackoverflow.com/questions/44511636/matplotlib-plot-feature-importance-with-feature-names

## Data Exploration
First, I will import the dict into a pandas Dataframe, since it will make the data exploration and clean up much easier for me. <br>
According to the documentation of the enron mail dataset the NAN values of financial data are related to a 0.
This is not true for the email address, but replacing a NAN with a 0 here will not have an influence on results, since the email address is not a candidate for a feature.

```python
df = pd.DataFrame(data_dict)
df = df.T
df = df.replace('NaN', 0)
df.dtypes
```
Total number of data points:
```python
print "Number of persons within the dataset:", len(data_dict)
```
146, but 1 value is the "total" row, which is removed later.

Number of POI:
```python
df['poi'].value_counts()
```
False    127
True      18



![Screenshot](pics/osm_area.png)
