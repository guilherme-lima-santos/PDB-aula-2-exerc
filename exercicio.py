import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r'04_dados_exercicio') #raw
features = dataset.iloc[0:, :-1].values
#print(features)
classe = dataset.iloc[:,-1].values
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(features[:,1:3])
features[:, 1:3] =  imputer.transform(features[:,1:3])
ColumnTransformer = ColumnTransformer(transformers =[('esonder',OneHotEncoder(), [0] )],
remainder = 'passthrough')

features = np.array(ColumnTransformer.fit_transform(features))

LabelEncoder = LabelEncoder()

classe = LabelEncoder.fit_transform(classe)

#print ('===========features=========')
#print(features)
#print('======classe=====')
#print (classe)

features_treinamento, features_teste , classe_treinamento , classe_teste = train_test_split(features,classe,test_size=0.2 , random_satet = 1 )

#print('=======features_treinamento=======')
#print(features_treinamento)
#print('=======features_teste=======')
#print(features_teste)
#print('=======classe_treinamento=======')
#print(classe_treinamento)
#print('=======classe_teste=======')
#print(classe_teste)


StandardScaler = StandardScaler()

features_treinamento[:, 3:] = StandardScaler.fit_transform(features_treinamento[:, 3:])

features_teste[:, 3:] = StandardScaler.transform(features_teste[:, 3:])

print(features_treinamento)


print(features_teste)


