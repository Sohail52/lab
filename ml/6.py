import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data = pd.read_csv("heart.csv")
model = BayesianNetwork([
    ('Age', 'HeartDisease'),
    ('Gender', 'HeartDisease'),
    ('ChestPainType','HeartDisease'),
    ('ExerciseInducedAngina','HeartDisease'),
    ('HeartDisease','RestingECG'),
    ('HeartDisease','Cholesterol')
])
model.fit(data, estimator=MaximumLikelihoodEstimator)
cpd_HeartDisease = MaximumLikelihoodEstimator(model, data).estimate_cpd('HeartDisease')
inference = VariableElimination(model)
print(inference.query(variables=['HeartDisease'], evidence={'RestingECG': 1}))
