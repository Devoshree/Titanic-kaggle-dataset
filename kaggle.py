import pandas as pd
from sklearn.ensemble import RandomForestClassifier
X=pd.read_csv("F:/Downloads/train.csv")
y=X.Survived
print X.describe()
#Filled missing values of age with mean
X['Age'].fillna(X.Age.mean(),inplace='True')
print X.describe()
#to just get the numeric variables..........
numeric_variables=list(X.dtypes[X.dtypes!='object'].index)
#print X[numeric_variables].head()
X.drop(['Name','Ticket','PassengerId'],axis=1,inplace='True')
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
X['Cabin']=X.Cabin.apply(clean_cabin)
categorical_variables=['Sex','Cabin','Embarked']
for variables in categorical_variables:
    X[variables].fillna('Missing',inplace='True')
    dummies=pd.get_dummies(X[variables],prefix=variables)
    X=pd.concat([X,dummies],axis=1)
    X.drop([variables],inplace=True,axis=1)
    
#print X
#fitting into model
model= RandomForestClassifier(n_estimators=100,random_state=42,oob_score=True)
model.fit(X, y)
#print model.oob_score_
print X.head()
test=pd.read_csv("F:/Downloads/test.csv")
x_new=test.loc[:]
new_pred=model.predict(x_new)
print new_pred
pd.DataFrame({"PassengerId":test.PassengerId, "Survived":new_pred}).set_index("PassengerId").to_csv('random2.csv')


