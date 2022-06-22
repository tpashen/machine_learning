#создайте здесь свой групповой проект!
#создай здесь свой индивидуальный проект!

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df2=pd.read_csv('test.csv')
print(df.head())
print(df.info())
print(df2.info())


s=df[df['result']==1]['sex'].value_counts()
print(s)

education=df[df['result']==1]['education_form'].value_counts()
print(education)

# преобразуем значения в целые числа

#relation — семейное положение
def set_int(rel):
   return int(rel)
df['relation'] = df['relation'].apply(set_int)
print(df['relation'])
df2['relation'] = df2['relation'].apply(set_int)


#followers_count — количество подписчиков.

df['followers_count'] = df['followers_count'].apply(set_int)
print(df['followers_count'])
df2['followers_count'] = df2['followers_count'].apply(set_int)


#graduation — год окончания обучения
df['graduation'] = df['graduation'].apply(set_int)
print(df['graduation'])
df2['graduation'] = df2['graduation'].apply(set_int)

#langs — список языков, которыми владеет пользователь
print(df['langs'])
def set_langs(lan):
   k=1
   for a in lan:
      if a==';':
         k=k+1
   return k
df['langs'] = df['langs'].apply(set_langs)
print(df['langs'])
df2['langs'] = df2['langs'].apply(set_langs)

# life_main — главное в жизни
def set_int2(rel):
   if rel=='False':
      rel=-1
   return int(rel)
df['life_main'] = df['life_main'].apply(set_int2)
print(df['life_main'])
df2['life_main'] = df2['life_main'].apply(set_int2)


# people_main — главное в людях
df['people_main'] = df['people_main'].apply(set_int2)
print(df['people_main'])
df2['people_main'] = df2['people_main'].apply(set_int2)



# преобразование стобца форма обучения в три фиктивных переменные
pd.get_dummies(df['education_form'])
print(pd.get_dummies(df['education_form']))
df[list(pd.get_dummies(df['education_form']).columns)] =pd.get_dummies(df['education_form'])
df.drop('education_form', axis = 1, inplace = True)

pd.get_dummies(df2['education_form'])
print(pd.get_dummies(df2['education_form']))
df2[list(pd.get_dummies(df2['education_form']).columns)] =pd.get_dummies(df2['education_form'])
df2.drop('education_form', axis = 1, inplace = True)


# преобразование стобца occupation_type — текущее занятие пользователя (школа, университет, работа) в три фиктивных переменные
pd.isnull(df['occupation_type']) 
pd.isnull(df2['occupation_type']) 
education=df[df['result']==1]['occupation_type'].value_counts()
print(education)
# преобразование стобца текущее занятие пользователя (школа, университет, работа) в три фиктивных переменные
pd.get_dummies(df['occupation_type'])
print(pd.get_dummies(df['occupation_type']))
df[list(pd.get_dummies(df['occupation_type']).columns)] =pd.get_dummies(df['occupation_type'])
df.drop('occupation_type', axis = 1, inplace = True)

pd.get_dummies(df2['occupation_type'])
print(pd.get_dummies(df2['occupation_type']))
df2[list(pd.get_dummies(df2['occupation_type']).columns)] =pd.get_dummies(df2['occupation_type'])
df2.drop('occupation_type', axis = 1, inplace = True)

# преобразование стобца occupation_type — текущее занятие пользователя (школа, университет, работа) в три фиктивных переменные
pd.isnull(df['education_status']) 
pd.isnull(df2['education_status']) 
education=df[df['result']==1]['education_status'].value_counts()
print(education)
# преобразование стобца текущее занятие пользователя (школа, университет, работа) в три фиктивных переменные
pd.get_dummies(df['education_status'])
print(pd.get_dummies(df['education_status']))
df[list(pd.get_dummies(df['education_status']).columns)] =pd.get_dummies(df['education_status'])
df.drop('education_status', axis = 1, inplace = True)

pd.get_dummies(df2['education_status'])
print(pd.get_dummies(df2['education_status']))
df2[list(pd.get_dummies(df2['education_status']).columns)] =pd.get_dummies(df2['education_status'])
df2.drop('education_status', axis = 1, inplace = True)



# удаляем столбцы
df.drop(['bdate','has_photo','has_mobile','city','last_seen','occupation_name','career_start','career_end'],axis = 1, inplace = True)
pd.isnull(df['id']) 
df2.drop(['bdate','has_photo','has_mobile','city','last_seen','occupation_name','career_start','career_end'],axis = 1, inplace = True)
print(df2.info())
print(df.info())

# Шаг 2. Создание модели
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

'''
X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
ID = X_test['id'] 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

result = pd.DataFrame({'id': ID, 'result':y_pred})

result.to_csv('res.csv', index = False)
'''


X = df.drop('result', axis = 1)
y = df['result']
X_train=X
y_train=y
X_test=df2
#y_test=df.tail(len(df2))

#y_test=df[df['sex']==1].head(len(df2))
#y_test=df[df['sex']==1].tail(len(df2))
y_test=df[df['university']==1].tail(len(df2))
y_test=y_test['result']


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
 
classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)
 
y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))


ID = df2['id']

result = pd.DataFrame({'id': ID, 'result':y_pred})
result.to_csv('res10.csv', index = False)
