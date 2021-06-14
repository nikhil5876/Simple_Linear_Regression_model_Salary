import pandas as pd

#read data
db = pd.read_csv('SalaryData.csv')

y = db['Salary']
x = db['YearsExperience']

#Reshape data 
x = x.values
x = x.reshape(-1,1)

from sklearn.linear_model import LinearRegression

#create model
mind = LinearRegression()
mind.fit(x,y)

#predict salary
experience = int(input("Enter Experience: "))
predicted_salary = mind.predict([[experience]])
print(predicted_salary)