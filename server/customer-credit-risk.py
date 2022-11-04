import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest,chi2,f_regression
from sklearn.linear_model import RidgeClassifier,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import DBSCAN,KMeans,AgglomerativeClustering
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score , confusion_matrix, accuracy_score,roc_auc_score,precision_score,recall_score,classification_report
from sklearn.model_selection import cross_validate
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import cufflinks as cf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import linear_model, datasets
from sklearn.metrics import  accuracy_score, confusion_matrix
from bayes_opt import BayesianOptimization
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn import cluster, datasets, preprocessing, metrics
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.metrics import silhouette_score
from apyori import apriori
from mlxtend.frequent_patterns import apriori, association_rules




columns_name= ['Status_Account', 'Duration_Of_Credit', 'Credit_History', 'Purpose_Of_Credit', 'Credit_Amount',
         'Savings_Account ', 'Customer_Employment', 'Income_Rate', 'Personal_Status', 'Credit_Debtors_Guarantors',
         'Customer_Residence', 'Customer_Property', 'Age', 'Plans', 'Housing_Situation ',
         'Number_Existing_Credits', 'Job', 'Liability_Nbr', 'Telephone', 'Foreign', 'Label']




df=pd.read_csv('project_data.csv',header=0,names=columns_name,sep=";")





df.head(5)





#find the null values in each column
df.isnull().sum().to_frame().rename(columns={0:'Null values'})





df.describe()




df.info()
df.nunique()




plt.figure(figsize=(8,8))
sns.countplot(df['Label'])
plt.xticks(rotation=45)



def bar_plot(df,x,y,title,colors=None,text=None):
    fig = px.bar(x=x,
                 y=y,
                 text=text,
                 color_discrete_sequence=px.colors.qualitative.Prism,
                 data_frame=df,
                 color=colors,
                 barmode='group',
                 template="simple_white")

    texts = [temp[col].values for col in y]
    for i, t in enumerate(texts):
        fig.data[i].text = t
        fig.data[i].textposition = 'outside'

    fig['layout'].title=title

    for trace in fig.data:
        trace.name = trace.name.replace('_',' ').capitalize()

    fig.update_yaxes(tickprefix="", showgrid=True)

    fig.show()

df['GoodOrBad'] = df['Label'].apply(lambda x: 'Good' if x==1 else 'Bad')


df['Personal_Status']=df['Personal_Status'].map({'A91':"male - divorced/separated",'A92':"female - divorced/separated/married", 'A93':"male - single",'A94':"male - married/widowed",'A95':"female - single" })


df["Sex"]=df["Personal_Status"].map({ 'male - divorced/separated':"male",'female - divorced/separated/married':"female", 'male - single':"male",'male - married/widowed':"male",'female - single':"female" })


temp = pd.DataFrame()
for sex in df['Sex'].unique().tolist():
    temp[sex] = df[df['Sex']==sex]['GoodOrBad'].value_counts()

temp = temp.rename(columns={0:'male',1:'female'}).T

fig = make_subplots(rows=2, cols=2,
                    horizontal_spacing=0.2,
                    specs=[[{"type": "pie",'rowspan':2}, {"type": "bar",'rowspan':2}],
                           [             None          ,             None           ]])

fig.add_trace(go.Pie(labels=temp.columns,
                     sort=False,
                     hole=0.3,
                     showlegend=False,
                     direction='clockwise',
                     domain={'x': [0.15, 0.85], 'y': [0.15, 0.85]},
                     textinfo='label+percent+text',
                     values=temp.loc['female'].values,
                     textposition='inside',
                     marker={'line': {'color': 'white', 'width': 1.5}},
                     name='female'),1,1)

fig.add_trace(go.Pie(labels=temp.columns,
                     values=temp.loc['male'].values,
                     sort=False,
                     showlegend=False,
                     direction='clockwise',
                     textinfo='label+percent+text',
                     domain={'x':[0.1,0.8],'y':[0,1]},
                     hole=0.8,
                     textposition='outside',
                     marker={'line': {'color': 'white', 'width': 1.5}},
                     name='male'),1,1)


fig.add_trace(go.Bar(x=temp.index,
                     marker_color='salmon',
                     y=temp['Good'],
                     name='Good',
                     text=temp['Good'].values,
                     textposition='auto'),1,2)

fig.add_trace(go.Bar(x=temp.index,
                     y=temp['Bad'],
                     name='Bad',
                     text=temp['Bad'].values,
                     textposition='auto'),1,2)

fig.show()


#lets make Sex column more clear to visualize
temp = pd.DataFrame()
for sex in df['Personal_Status'].unique().tolist():
    temp[sex] = df[df['Personal_Status']==sex]['GoodOrBad'].value_counts()

temp = temp.rename(columns={0:'male - divorced/separated',1:'female - divorced/separated/married',2:'male - single',3:'male - married/widowed',4:'female - single'}).T.reset_index()
temp['Total sum'] = temp.sum(axis=1)

temp
bar_plot(temp,
         'index',
         ['Total sum','Good','Bad'],
         title='Good or Bad by Personal Status')


df['age_category'] = pd.cut(df['Age'], bins=[19,29,39,49,59,69,77],labels=["19-28","29-39","40-49","50-59","60-69","70-79"]).to_frame()

temp = pd.DataFrame()
for age in df['age_category'].unique().tolist():
    temp[age] = df[df['age_category']==age]['GoodOrBad'].value_counts()

temp = temp.T.reset_index()
temp['Total sum'] = temp.sum(axis=1)


bar_plot(temp,
         'index',
         ['Total sum','Good','Bad'],
         title='Good Or Bad Customer by age')


df['Purpose_Of_Credit']=df['Purpose_Of_Credit'].map({'A40':"car (new)",
                                                     'A41':"car (used)",
                                                     'A42':"furniture/equipment",
                                                     'A43':"radio/television",
                                                     'A44':"domestic appliances",
                                                     'A45':"repairs",
                                                     'A46':"education",
                                                     'A47':"vacation",
                                                     'A48':"retraining",
                                                     'A49':"business",
                                                     'A410':"others" })



# Sets the figure size temporarily but has to be set again the next plot
fig=sns.catplot(x="Purpose_Of_Credit", y="Credit_Amount", hue="GoodOrBad", kind="bar", data=df,height=7 )
plt.xticks(rotation=45)
plt.figure(figsize=(40,40))
fig.savefig('example.png')


temp = pd.DataFrame()
for c in df['Purpose_Of_Credit'].unique().tolist():
    temp[c] = df[df['Purpose_Of_Credit']==c]['GoodOrBad'].value_counts()

temp = temp.T.reset_index()
temp['Total sum'] = temp.sum(axis=1)

bar_plot(temp,
         'index',
         ['Total sum','Good','Bad'],
         title='Good Or Bad Customer by the purpose of Credit')


num_features=['Duration_Of_Credit','Credit_Amount','Income_Rate','Customer_Residence','Age','Number_Existing_Credits','Liability_Nbr']


# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_features = df.columns[categorical_feature_mask].tolist()
categ= categorical_features[:-3]
# instantiate labelencoder object
le = LabelEncoder()
# apply le on categorical feature columns
df[categ] = df[categ].apply(lambda col: le.fit_transform(col))


df[num_features] = MinMaxScaler().fit_transform( df[num_features])


data=df.iloc[:, :-3]


y = data["Label"]
X = data.drop("Label",axis=1)



#lets see the correlation between columns and target column
corr = data.corr()['Label'].to_frame()
corr = corr.rename(columns={'Label':'Correlation with target'})
corr.sort_values(by='Correlation with target',ascending=False)[1:].style.background_gradient(axis=1,cmap=sns.light_palette('Pink', as_cmap=True))


def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)

    regressor_OLS.summary()
    return x, columns


SL = 0.05
data_modeled, selected_columns = backwardElimination(data.iloc[:,:-1].values, data.iloc[:,-1].values, SL, data.iloc[:,:-1].columns)


selected_columns


def best_features(df,target):
   features = list(df.columns)
   features.remove(target)
   y = df[target]
   X=df[features]
   #apply SelectKBest class to extract top 10 best features
   bestfeatures = SelectKBest(score_func=f_classif, k='all')
   fit = bestfeatures.fit(X,y)
   dfscores = pd.DataFrame(fit.scores_)
   dfcolumns = pd.DataFrame(X.columns)
   #concat two dataframes for better visualization
   featureScores = pd.concat([dfcolumns,dfscores],axis=1)
   featureScores.columns = ['Specs','Score']  #naming the dataframe columns
   return featureScores.nlargest(10,'Score')  #print 10 best features


best=best_features(data,'Label')
needed_features=best['Specs'].values
needed_features


cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
corr = data[needed_features].corr()
corr.iplot(kind='heatmap',
           colorscale='Blues',
           hoverinfo='all',
           layout = go.Layout(title='Correlation Heatmap for the correlation between our columns',
                              titlefont=dict(size=20)))


columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False


newData=data[needed_features]
selected_columns = newData.columns[columns]
selected_columns.shape


X = data[needed_features]


# split into train test set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# # List of ML models to be tested

model_name = ["Random Forest","KNN","Logistic Regression","SVM","Gradient Boosting","BaggingClassifier"]
clfModel_score=[]


# # Random Forest

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score , confusion_matrix, accuracy_score,roc_auc_score,precision_score,recall_score,classification_report
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5)
# Fit the random search model
rf_random.fit(X_train, y_train)


best_rf = RandomForestClassifier(n_estimators = rf_random.best_estimator_.get_params()['n_estimators'])
best_rf.fit(X_train, y_train)
# Predict target vector
pred_rf=best_rf.predict(X_test)

print("Accuracy score: {:.2f}".format(accuracy_score(y_test,pred_rf)),"\n")
clfModel_score.append( recall_score(y_test,pred_rf)  )
print("F1:",f1_score(y_test, pred_rf))
print("Recall:",recall_score(y_test, pred_rf))
print(" ROC AUC:",roc_auc_score(y_test, pred_rf))
print("Precision:",precision_score(y_test, pred_rf))
print(confusion_matrix(y_test,pred_rf))
print("Classification Report:",)
print (classification_report(y_test, pred_rf))

#plot cm
cm=confusion_matrix(y_test,pred_rf)
sns.heatmap(cm, annot=True, cmap="YlGnBu" ,fmt='g') # font size
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted class')
plt.show()


# # KNN

#List Hyperparameters that we want to tune.
n_neighbors = list(range(1,30))
p=[1,2]
#Convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p)
#Create new KNN object
knn_2 = KNeighborsClassifier()
#Use GridSearch
clf = GridSearchCV(knn_2, hyperparameters, cv=5)
#Fit the model
best_model = clf.fit(X_test, y_test)
#Print The value of best Hyperparameters
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
#check mean score for the top performing value of n_neighbors
print('Best Score:', best_model.best_score_)


# build KNN model and choose n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors = best_model.best_estimator_.get_params()['n_neighbors'])
# train the model
knn.fit(X_train, y_train)
# get the predict value from X_test
pred = knn.predict(X_test)
# print the score
print("Accuracy score: {:.2f}".format(accuracy_score(y_test,pred)))
clfModel_score.append( recall_score(y_test,pred)  )
print("F1:",f1_score(y_test, pred))
print("Recall:",recall_score(y_test, pred))
print(" ROC AUC:",roc_auc_score(y_test, pred))
print("Precision:",precision_score(y_test, pred))
print("Classification Report:",)
print (classification_report(y_test, pred))

#plot cm
cm=confusion_matrix(y_test,pred)
sns.heatmap(cm, annot=True, cmap="YlGnBu" ,fmt='g') # font size
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted class')
plt.show()


# choose k between 1 to 20
k_range = range(1, 20)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_test, y_test, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
# As we can see the best K is between 6–13, after 13 the accuracy got decreased due to under-fitting.


# # Logistic Regression

# Create logistic regression
logistic = linear_model.LogisticRegression()

# Create regularization penalty space
penalty = ['l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty )

# # Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)

# # Fit grid search
best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
print('Best Score for LR:', clf.best_score_)


# i cann add cross validate here with the parametre i found with Grid search
# penalty=best_model.best_estimator_.get_params()['penalty']
best_lr = LogisticRegression(C=0.01)

best_lr.fit(X_train, y_train)
# Predict target vector
pred=best_lr.predict(X_test)

print("Accuracy score: {:.2f}".format(accuracy_score(y_test,pred)))
clfModel_score.append(  recall_score(y_test,pred)  )
print("F1:",f1_score(y_test, pred))
print("Recall:",recall_score(y_test, pred))
print(" ROC AUC:",roc_auc_score(y_test, pred))
print("Precision:",precision_score(y_test, pred))
print(confusion_matrix(y_test,pred))
print("Classification Report:",)
print (classification_report(y_test, pred))


#plot cm
cm=confusion_matrix(y_test,pred)
sns.heatmap(cm, annot=True, cmap="YlGnBu" ,fmt='g') # font size
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted class')
plt.show()


# # SVM

def estimator(C, gamma):
    # initialize model
    model = SVC(C=C,gamma=gamma, degree=1, random_state=0)
    # set in cross-validation
    result = cross_validate(model, X_train, y_train, cv=10)
    # result is mean of test_score
    return np.mean(result['test_score'])


hparams = {"C": (0.01, 50), "gamma": ( 0.0001,0.001)}


# give model and hyperparameter to optmizer
svc_bayesopt = BayesianOptimization(estimator, hparams)


# # maximize means optimization
svc_bayesopt.maximize(init_points=5, n_iter=10, acq='ucb')


print(svc_bayesopt.max)


# example
estimator = SVC(C=30, gamma=0.001)
estimator.fit(X_train, y_train)
y_pred_svm = estimator.predict(X_test)
print("Accuracy score: {:.2f}".format(accuracy_score(y_test,y_pred_svm)))
clfModel_score.append( recall_score(y_test,y_pred_svm) )
print("F1:",f1_score(y_test, y_pred_svm))
print("Recall:",recall_score(y_test, y_pred_svm))
print(" ROC AUC:",roc_auc_score(y_test, y_pred_svm))
print("Precision:",precision_score(y_test, y_pred_svm))
print("Classification Report:",)
print (classification_report(y_test, y_pred_svm))
#plot cm
cm=confusion_matrix(y_test,y_pred_svm)
sns.heatmap(cm, annot=True, cmap="YlGnBu" ,fmt='g') # font size
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted class')
plt.show()


# # Gradient Boosting

# define models and parameters
model = GradientBoostingClassifier()
n_estimators = [10, 50, 100]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]
# define grid search
grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']



best_grad = GradientBoostingClassifier(n_estimators = grid_result.best_estimator_.get_params()['n_estimators'])
best_grad.fit(X_train, y_train)
# Predict target vector
pred_gred=best_grad.predict(X_test)

print("Accuracy score: {:.2f}".format(accuracy_score(y_test,pred_gred)),"\n")
clfModel_score.append(recall_score(y_test,pred_gred) )
print("F1:",f1_score(y_test, pred_gred))
print("Recall:",recall_score(y_test, pred_gred))
print(" ROC AUC:",roc_auc_score(y_test, pred_gred))
print("Precision:",precision_score(y_test, pred_gred))
print(confusion_matrix(y_test,pred_gred))
print("Classification Report:",)
print (classification_report(y_test, pred_gred))
#plot cm
cm=confusion_matrix(y_test,pred_gred)
sns.heatmap(cm, annot=True, cmap="YlGnBu" ,fmt='g') # font size
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted class')
plt.show()


y_prob = best_grad.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
newy=y_test.replace([ 1, 2], [0,1])
false_positive_rate, true_positive_rate, thresholds = roc_curve(newy, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


scoring = ['accuracy','precision', 'recall','roc_auc', 'f1']
scores = cross_validate(best_grad, X_train, y_train, cv=5,scoring=scoring, return_train_score=False)


for metricname in scores.keys():
    average_score = np.average(scores[metricname])
    print('%s : %f' % (metricname, average_score))


# # BaggingClassifier

# define dataset
# define models and parameters
model = BaggingClassifier()
n_estimators = [10,50,100]
# define grid search
grid = dict(n_estimators=n_estimators)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

best_tree = BaggingClassifier(n_estimators = grid_result.best_estimator_.get_params()['n_estimators'])
best_tree.fit(X_train, y_train)
# Predict target vector
pred_tree=best_lr.predict(X_test)

print("Accuracy score: {:.2f}".format(accuracy_score(y_test,pred_tree)),"\n")
clfModel_score.append( recall_score(y_test,pred_tree) )
print("F1:",f1_score(y_test, pred_tree))
print("Recall:",recall_score(y_test, pred_tree))
print(" ROC AUC:",roc_auc_score(y_test, pred_tree))
print("Precision:",precision_score(y_test, pred_tree))
print("Classification Report:",)
print (classification_report(y_test, pred_tree))
#plot cm
cm=confusion_matrix(y_test,y_pred_svm)
sns.heatmap(cm, annot=True, cmap="YlGnBu" ,fmt='g') # font size
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted class')
plt.show()


# # Plotting the result of the test

sns.barplot(model_name,clfModel_score)
plt.xticks(rotation=20)


# # Clustering

data_cluster=pd.read_csv('project_data.csv',header=0,names=columns_name,sep=";")
data_cluster.drop("Label",axis=1,inplace=True)
feature_cluster=["Duration_Of_Credit","Credit_Amount","Age"]
data_to_cluster=data_cluster[feature_cluster]
scaled_data_to_cluster=pd.DataFrame(MinMaxScaler().fit_transform(data_to_cluster),columns=feature_cluster)


distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(scaled_data_to_cluster)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(10,5))
sns.lineplot(range(1, 20), distortions,marker='o',color='red')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')


k_clusters = []
sil_coeffecients = []

for n_cluster in range(2,20):
    kmeans = KMeans(n_clusters = n_cluster).fit(scaled_data_to_cluster)
    label = kmeans.labels_
    sil_coeff = silhouette_score(scaled_data_to_cluster, label)
    print("For n_clusters={}, Silhouette Coefficient = {}".format(n_cluster, sil_coeff))
    sil_coeffecients.append(sil_coeff)
    k_clusters.append(n_cluster)

plt.plot(k_clusters, sil_coeffecients)
plt.ylabel('Silhouette Coefficient'), plt.xlabel('No. of Clusters')
plt.show()

sil_best = max(sil_coeffecients)
k_best_index = sil_coeffecients.index(sil_best)
k_best = k_clusters[k_best_index]
print("Optimum Number of Clusters:", k_best)


pred_kmeans = KMeans(n_clusters=3, random_state=0).fit_predict(scaled_data_to_cluster)
fig=px.scatter_3d(data_to_cluster,x="Duration_Of_Credit",y="Credit_Amount",z="Age",color=pred_kmeans)
fig.update_layout(scene = dict(
                    xaxis_title='Duration_Of_Credit',
                    yaxis_title='Credit_Amount',
                    zaxis_title='Age'),
                    width=700
                )
fig.show()


# # Pattern Mining

cat_feat=categorical_features[:-3]
for col in data.columns:
    if col not in num_features :
        data[col] = data[col].astype('category')
data.info()


good_customers_data=pd.get_dummies(data[cat_feat][y==1])
association_results =apriori(good_customers_data,min_support=0.6,use_colnames=True)

rules = association_rules(association_results, metric="lift", min_threshold=0.5)
rules


print(len(rules))
print(rules.sort_values('lift', ascending=True))


bad_customers_data=pd.get_dummies(data[cat_feat][y==2])
association_results_bad =apriori(bad_customers_data,min_support=0.6,use_colnames=True)

rules = association_rules(association_results_bad, metric="lift", min_threshold=0.5)
rules

