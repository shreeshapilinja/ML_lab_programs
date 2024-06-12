import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

heart_disease = pd.read_csv('heart.csv')
heart_disease = heart_disease.replace('?', np.nan)

# Define the Bayesian network structure
model = BayesianModel([
    ('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'), ('heartdisease', 'chol')
])
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

heart_disease_infer = VariableElimination(model)

# Calculate probability of Heart Disease given Age=63
q = heart_disease_infer.query(variables=['heartdisease'], evidence={'age': 63})
print(q)

q = heart_disease_infer.query(variables=['heartdisease'], evidence={'chol': 233})  # cholesterol
print(q)

q = heart_disease_infer.query(variables=['heartdisease'], evidence={'age': 63, 'sex' :1,'trestbps':130})
print(q)