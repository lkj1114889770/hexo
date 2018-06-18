---
title: kaggle入门：Predict survival on the Titanic
date: 2017-09-12 10:31:55
tags:
	- 机器学习
---

[Kaggle](https://www.kaggle.com/)是一个数据分析建模的应用竞赛平台，学习了机器学习的算法，这是个很好的应用平台。Predict survival on the Titanic作为kaggle入门级别的比赛，也是我接触kaggle的第一个实践项目，最后的结果虽然不够优秀，仅Top20%左右，但是还是将第一次的实践过程mark一下。
![](https://i.imgur.com/FEDyFse.jpg)

<!-- more -->

## 数据分析
从kaggle的网站上下载好比赛数据（train.csv和test.csv），泰坦尼克号问题就是根据乘客的个人信息，分析是否能够活下来，训练集提供了乘客信息以及存活状况，测试集仅提供信息，需要预测能否存活，其实就是一个二分类问题。

下面读入训练集数据，开始初步分析。

    import pandas as pd
	import numpy as np
	from pandas import Series,DataFrame
	data_train = pd.read_csv("train.csv")
	data_train.info()

可以看到数据信息：

	RangeIndex: 891 entries, 0 to 890
	Data columns (total 12 columns):
	PassengerId    891 non-null int64
	Survived       891 non-null int64
	Pclass         891 non-null int64
	Name           891 non-null object
	Sex            891 non-null object
	Age            714 non-null float64
	SibSp          891 non-null int64
	Parch          891 non-null int64
	Ticket         891 non-null object
	Fare           891 non-null float64
	Cabin          204 non-null object
	Embarked       889 non-null object
	dtypes: float64(2), int64(5), object(5)

数据中存在缺项，其中不同的字段表示的信息为


PassengerId    乘客ID
Survived       是否存活
Pclass         船舱等级
Name           姓名
Sex            性别
Age            年龄
SibSp          兄弟姐妹个数
Parch          父母小孩个数
Ticket         船票信息
Fare           票价
Cabin          船舱
Embarked       登船港口

	import matplotlib.pyplot as plt
	fig = plt.figure()
	fig.set(alpha=0.2)  # 设定图表颜色alpha参数
	data_train.Age[data_train.Survived == 0].plot(kind='kde')   
	data_train.Age[data_train.Survived == 1].plot(kind='kde')
	plt.xlabel(u"年龄")
	plt.ylabel(u"密度")
	plt.title(u"从年龄看获救情况")
	plt.legend((u"未获救",u"获救"),loc='best')
	plt.show()

![](https://i.imgur.com/Dt77VZG.png)

年龄越小，获救的概率还是越高的，小孩还是要优先嘛。

	#看看各乘客等级的获救情况
	fig = plt.figure()
	fig.set(alpha=0.2)
	
	Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
	Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
	df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
	df.plot(kind='bar', stacked=True)
	plt.title(u"各乘客等级的获救情况")
	plt.xlabel(u"乘客等级") 
	plt.ylabel(u"人数") 
	plt.show()

![](https://i.imgur.com/CAp7eKj.png)

船舱等级越高，获救概率也是越高，还是上层社会的人容易获救啊。

	#看看各性别的获救情况
	fig = plt.figure()
	fig.set(alpha=0.2)  
	
	Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
	Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
	df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
	df.plot(kind='bar', stacked=True)
	plt.title(u"按性别看获救情况")
	plt.xlabel(u"性别") 
	plt.ylabel(u"人数")
	plt.show()

![](https://i.imgur.com/1PG3g2x.png)

女性获救概率更高，嗯，Lady first.

姓名中含有一些称谓信息，也代表着乘客的身份以及社会地位。
	
	data_train.groupby(data_train['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0]))['Survived'].mean()

得到的结果如下：
	
	Name
	Capt            0.000000
	Col             0.500000
	Don             0.000000
	Dr              0.428571
	Jonkheer        0.000000
	Lady            1.000000
	Major           0.500000
	Master          0.575000
	Miss            0.697802
	Mlle            1.000000
	Mme             1.000000
	Mr              0.156673
	Mrs             0.792000
	Ms              1.000000
	Rev             0.000000
	Sir             1.000000
	the Countess    1.000000
	Name: Survived, dtype: float64

还是要看身份的，有身份的人容易获救啊。

	#信息中船舱有无对于获救情况影响
	fig = plt.figure()
	fig.set(alpha=0.2) 
	
	Cabin_Has= data_train.Survived[data_train.Cabin.notnull()].value_counts()
	Cabin_None = data_train.Survived[data_train.Cabin.isnull()].value_counts()
	df=pd.DataFrame({u'有船舱号':Cabin_Has, u'无船舱号':Cabin_None})
	df.plot(kind='bar', stacked=True)
	plt.title(u"有无船舱号的获救情况")
	plt.xlabel(u"船舱号有无") 
	plt.ylabel(u"人数") 
	plt.show()

![](https://i.imgur.com/C9tiryu.png)

看起来有船舱号的人获救概率高一点。

	#信息中Embarked登船港口的影响
	fig = plt.figure()
	fig.set(alpha=0.2) 
	
	Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
	Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
	df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
	df.plot(kind='bar', stacked=True)
	plt.title(u"不同登船港口的获救情况")
	plt.xlabel(u"乘客等级") 
	plt.ylabel(u"人数") 
	plt.show()

![](https://i.imgur.com/8CZL0cP.png)

C港口获救概率高一点。


## 数据处理
下面就需要对训练集以及测试集的数据进行处理，以便算法的处理。

	data_test = pd.read_csv('test.csv')
	data_test['Survived']=3 #为了能够合并在一起，test数据添加Suvrvived一列，不过数值为不存在的3
	data_combine=pd.concat([data_train,data_test])
	data_combine.info()

可以看到合并数据信息：

	Int64Index: 1309 entries, 0 to 417
	Data columns (total 12 columns):
	Age            1046 non-null float64
	Cabin          295 non-null object
	Embarked       1307 non-null object
	Fare           1308 non-null float64
	Name           1309 non-null object
	Parch          1309 non-null int64
	PassengerId    1309 non-null int64
	Pclass         1309 non-null int64
	Sex            1309 non-null object
	SibSp          1309 non-null int64
	Survived       1309 non-null int64
	Ticket         1309 non-null object
	dtypes: float64(2), int64(5), object(5)

### 缺省数据的处理
上面信息中可以看到Age、Cabin、Embarked、Fare有数据缺失，下面需要进行处理。

Cabin缺失数据很多，可以将有无Cabin记录为特征；Embarked仅缺失2个，可以取取值最多的来填补。

	#对船舱号缺值进行处理，有船舱号标识为1，无船舱号标识为0
	data_combine.loc[(data_combine.Cabin.notnull()), 'Cabin' ] = 1
	data_combine.loc[(data_combine.Cabin.isnull()), 'Cabin' ] = 0
	#有两行数据缺失Embarked值，补全为Embarked的最多取值S
	data_combine.loc[(data_combine.Embarked.isnull()),'Embarked'] = 'S'

Fare有一个缺值，取所有Fare的平均值来填补。

	data_combine.Fare.fillna(data_combine.Fare.mean(), inplace=True) #Fare有一个缺值，用均值填充

Age缺省数值也是不多，但是不像Fare和Embarked那么少，考虑使用算法来拟合，采用随机森林的回归进行预测拟合填补。

	from sklearn.ensemble import RandomForestRegressor
	
	### 使用 RandomForestClassifier 填补缺失的年龄属性
	def set_missing_ages(df):
	    # 把已有的数值型特征取出来丢进Random Forest Regressor中
	    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
	    # 乘客分成已知年龄和未知年龄两部分
	    known_age = age_df[age_df.Age.notnull()].as_matrix()
	    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
	    # y即目标年龄
	    y = known_age[:, 0]
	    # X即特征属性值
	    X = known_age[:, 1:]
	    # fit到RandomForestRegressor之中
	    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
	    rfr.fit(X, y)
	    # 用得到的模型进行未知年龄结果预测
	    predictedAges = rfr.predict(unknown_age[:, 1::])
	    # 用得到的预测结果填补原缺失数据
	    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
	    return df, rfr
	
	data_combine, rfr = set_missing_ages(data_combine)

### 特征处理

对名字可以提取到称谓作为一个特征，根据SibSp和Parch可以知道家庭情况，再添加一个FamilySize特征。

	data_combine['title']=data_combine['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
	data_combine['FamilySize']=data_combine['SibSp'] + data_combine['Parch']

下面将一些特征取值为文本的转化成数值取值。

	#将所有特征转换成数值型编码
	# Sex
	df = pd.get_dummies(data_combine['Sex'],prefix='Sex')
	data_combine = pd.concat([data_combine,df],axis=1).drop('Sex',axis=1)
	
	# Embarked
	df = pd.get_dummies(data_combine['Embarked'],prefix='Embarked')
	data_combine = pd.concat([data_combine,df],axis=1).drop('Embarked',axis=1)
	
	# title
	data_combine['title']=data_combine['title'].astype('category')
	data_combine['title']=data_combine['title'].cat.codes
	
	# Pclass
	df = pd.get_dummies(data_combine['Pclass'],prefix='Pclass')
	data_combine = pd.concat([data_combine,df],axis=1).drop('Pclass',axis=1)
	
	
	data_combine.drop(['Name','SibSp','Parch','Ticket'],axis=1,inplace=True)

### 算法预测
预测算法采用集成学习中的随机森林，作一个典型的二分类。	

	from sklearn.ensemble import RandomForestClassifier
	X_train = data_combine.iloc[:891,:].drop(["PassengerId","Survived"], axis=1)
	Y_train = data_combine.iloc[:891,:]["Survived"]
	X_test = data_combine.iloc[891:,:].drop(["PassengerId","Survived"], axis=1)
	clf = RandomForestClassifier(n_estimators=300,min_samples_leaf=4)
	clf.fit(X_train, Y_train)
	Y_test = clf.predict(X_test)
	gender_submission = pd.DataFrame({'PassengerId':data_test.iloc[:,0],'Survived':Y_test})
	gender_submission.to_csv('gender_submission.csv', index=None)

## submission
将保存的gender_submission.csv文件提交到kaggle，得到的0.79904，毕竟第一次实践，进入了Top20%，虽然不够优秀，也还可以，算法继续学习之后还待改进。
![](https://i.imgur.com/jOHX4b4.png)



