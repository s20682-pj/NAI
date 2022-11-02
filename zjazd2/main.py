"""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
Artificial intelligence is increasingly being used in every aspect of life and work.
It is also being used more and more in medicine. Our system can determine how high the chance
of future heart disease by giving age, cholesterol, blood pressure, and bmi.

Score 0-50 - low risk
Score 50-75 medium risk
Score 75-100 high risk

System requirements:
- Python 3.10
- skfuzzy
- Numpy
"""
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


"""
Declaration of what parameters will be retrieved and what values they will have
"""
cholesterol = ctrl.Antecedent(np.arange(100, 301, 1), 'cholesterol')
blood_pressure = ctrl.Antecedent(np.arange(100, 221, 1), 'blood_pressure')
age = ctrl.Antecedent(np.arange(0, 81, 1), 'age')
risk = ctrl.Consequent(np.arange(0, 100, 1), 'risk')


"""
Declaration of ranges of values of each variable and outcome
"""
cholesterol['low'] = fuzz.trapmf(np.arange(100, 301, 1), [100, 100, 150, 200])
cholesterol['average'] = fuzz.trapmf(np.arange(100, 301, 1), [175, 200, 225, 250])
cholesterol['high'] = fuzz.trapmf(np.arange(100, 301, 1), [240, 275, 300, 300])

blood_pressure['low'] = fuzz.trapmf(np.arange(100, 221, 1), [100, 100, 120, 150])
blood_pressure['average'] = fuzz.trimf(np.arange(100, 221, 1), [135, 145, 165])
blood_pressure['high'] = fuzz.trapmf(np.arange(100, 221, 1), [155, 190, 220, 220])

age['low'] = fuzz.trapmf(np.arange(0, 81, 1), [0, 0, 30, 50])
age['average'] = fuzz.trimf(np.arange(0, 81, 1), [40, 55, 75])
age['high'] = fuzz.trapmf(np.arange(0, 81, 1), [60, 85, 100, 100])

risk['low'] = fuzz.trimf(risk.universe, [0, 25, 50])
risk['medium'] = fuzz.trimf(risk.universe, [25, 50, 75])
risk['high'] = fuzz.trimf(risk.universe, [50, 75, 100])

# cholesterol.view()
# blood_pressure.view()
# age.view()
# risk.view()


"""
Rules according to which the risk is calculated
"""
rule1 = ctrl.Rule(age['low'] & cholesterol['average'] & blood_pressure['low'], risk['low'])
rule2 = ctrl.Rule(age['low'] & cholesterol['low'] & blood_pressure['low'], risk['low'])
rule3 = ctrl.Rule(age['average'] & cholesterol['low'] & blood_pressure['low'], risk['low'])
rule4 = ctrl.Rule(age['low'] & cholesterol['low'] & blood_pressure['high'], risk['low'])
rule5 = ctrl.Rule(age['high'] & cholesterol['low'] & blood_pressure['low'], risk['medium'])
rule6 = ctrl.Rule(age['average'] & cholesterol['high'], risk['medium'])
rule7 = ctrl.Rule(age['low'] & cholesterol['average'] & blood_pressure['average'], risk['medium'])
rule8 = ctrl.Rule(age['average'] & cholesterol['low'] & blood_pressure['high'], risk['medium'])
rule9 = ctrl.Rule(age['low'] & cholesterol['high'] & blood_pressure['high'], risk['high'])
rule10 = ctrl.Rule(age['average'] & cholesterol['high'] & blood_pressure['high'], risk['high'])
rule11 = ctrl.Rule(age['average'] & cholesterol['average'] & blood_pressure['high'], risk['high'])
rule12 = ctrl.Rule(age['high'] & cholesterol['high'] & blood_pressure['high'], risk['high'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12])
risk_disease = ctrl.ControlSystemSimulation(risk_ctrl)

"""
Providing the values of the variables
Cholesterol: 100-300
Blood pressure: 100-220
Age: 0-80
"""
risk_disease.input['cholesterol'] = 299
risk_disease.input['blood_pressure'] = 219
risk_disease.input['age'] = 79

risk_disease.compute()
print("Cholesterol: 299, blood pressure: 219, age: 79")
print(risk_disease.output['risk'])
#risk.view(sim=risk_disease)

risk_disease.input['cholesterol'] = 142
risk_disease.input['blood_pressure'] = 114
risk_disease.input['age'] = 25

risk_disease.compute()
print("Cholesterol: 142, blood pressure: 114, age: 25")
print(risk_disease.output['risk'])

risk_disease.input['cholesterol'] = 257
risk_disease.input['blood_pressure'] = 195
risk_disease.input['age'] = 57

risk_disease.compute()
print("Cholesterol: 257, blood pressure: 195, age: 57")
print(risk_disease.output['risk'])

risk_disease.input['cholesterol'] = 115
risk_disease.input['blood_pressure'] = 174
risk_disease.input['age'] = 71

risk_disease.compute()
print("Cholesterol: 115, blood pressure: 174, age: 71")
print(risk_disease.output['risk'])

risk_disease.input['cholesterol'] = 268
risk_disease.input['blood_pressure'] = 175
risk_disease.input['age'] = 60

risk_disease.compute()
print("Cholesterol: 268, blood pressure: 175, age: 60")
print(risk_disease.output['risk'])