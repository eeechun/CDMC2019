import numpy as np
import pandas as pd
from gensim import utils
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import svm,linear_model
from sklearn.linear_model import LinearRegression


def readLabel(label):
    df=pd.read_csv(r"./CDMC2019Task2Train.csv")
    df=df.iloc[:,1:]
    #print(df.shape)

    for i in range(df.shape[0]):
        label.append(df.iloc[i,:].to_numpy().flatten())
    label=np.array(label)
    print("label:\n",label)
    print("shape of label:",label.shape)
    return label

def training(k):
    data=[]
    for i in range(1,k):      
        i="%04d"%i
        with open(r"./TRAIN/"+str(i)+".seq") as f:
            r=[]
            for line in f:  
                r+=utils.simple_preprocess(line)
            
                '''temp = line.replace("\n", " ")  
                c+=temp''' #causes memory error
            data.append(r)
    data = np.array(data)
    data.reshape(-1, 1)

    print("type of data")
    print(type(data))
    print("shape of data")
    print(data.shape)
    print("==========================================")
    return data

def tfidf(trainData,labels):
    data=[]
    for s in trainData:
        str=""
        for word in s:
            str+=word+' '
        data.append(str)        
        
    tfidfVectotizer=TfidfVectorizer()
    data=tfidfVectotizer.fit_transform(data)
    data=data.toarray()
    data=np.array(data)

    label=labels[:]
    label=np.ravel(label)
    return data, label

#knn
def knn_model(X_train, Y_train, X_test, Y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    y_predict = knn.predict(X_test)
    print("[KNN][TFIDF]")   
    print(f"accuracy score: {accuracy_score(Y_test,y_predict)}")
    print(f"F1 score: {f1_score(Y_test,y_predict,average ='weighted')}")
    print(f"precision score: {precision_score(Y_test,y_predict,average ='weighted')}")
    print(f"recall score: {recall_score(Y_test,y_predict,average = 'weighted')}")
    print("==========================================")

#random forest model
def rf_model(X_train, Y_train, X_test, Y_test):
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    forest.fit(X_train, Y_train)
    y_predicted = forest.predict(X_test)
    print("[rf][TFIDF]")    
    print(f"accuracy score: {accuracy_score(Y_test,y_predicted)}")
    print(f"F1 score: {f1_score(Y_test,y_predicted,average ='weighted')}")
    print(f"precision score: {precision_score(Y_test,y_predicted,average ='weighted')}")
    print(f"recall score: {recall_score(Y_test,y_predicted,average = 'weighted')}")
    print("==========================================")

#svm
def svm_model(X_train, Y_train, X_test, Y_test):
    svm_lin=svm.SVC(kernel='linear')
    svm_lin.fit(X_train,Y_train)
    y_predicted=svm_lin.predict(X_test)
    print("[SVM_linear][TFIDF]")
    print(f"accuracy score: {accuracy_score(Y_test,y_predicted)}")
    print(f"F1 score: {f1_score(Y_test,y_predicted,average ='weighted')}")
    print(f"precision score: {precision_score(Y_test,y_predicted,average ='weighted')}")
    print(f"recall score: {recall_score(Y_test,y_predicted,average = 'weighted')}")
    print("==========================================")

    #svm rbf
    svm_rbf=svm.SVC(kernel='rbf')
    svm_rbf.fit(X_train,Y_train)
    y_predicted=svm_rbf.predict(X_test)
    print("[SVM_RBF][TFIDF]")
    print(f"accuracy score: {accuracy_score(Y_test,y_predicted)}")
    print(f"F1 score: {f1_score(Y_test,y_predicted,average ='weighted')}")
    print(f"precision score: {precision_score(Y_test,y_predicted,average ='weighted')}")
    print(f"recall score: {recall_score(Y_test,y_predicted,average = 'weighted')}")
    print("==========================================")

    #svm poly
    svm_poly=svm.SVC(kernel='poly')
    svm_poly.fit(X_train,Y_train)
    y_predicted=svm_poly.predict(X_test)
    print("[SVM_poly][TFIDF]")
    print(f"accuracy score: {accuracy_score(Y_test,y_predicted)}")
    print(f"F1 score: {f1_score(Y_test,y_predicted,average ='weighted')}")
    print(f"precision score: {precision_score(Y_test,y_predicted,average ='weighted')}")
    print(f"recall score: {recall_score(Y_test,y_predicted,average = 'weighted')}")
    print("==========================================")


if __name__ == "__main__":
    label=[]
    label=readLabel(label)
    train=training(4168)

    data, label = tfidf(train,label)
    X_train, X_test, Y_train, Y_test = train_test_split(data,label,test_size=0.2)
    knn_model(X_train, Y_train, X_test, Y_test)
    rf_model(X_train, Y_train, X_test, Y_test)
    svm_model(X_train, Y_train, X_test, Y_test)
