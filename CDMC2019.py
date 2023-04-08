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
    df=pd.read_csv(r"C:\Users\tammy\lab\CDMC2019\CDMC2019\CDMC2019Task2Train.csv")
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
        with open(r"C:\Users\tammy\lab\CDMC2019\CDMC2019\TRAIN\\"+str(i)+".seq") as f:
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

'''def vectorizer(model,word):
    model=Word2Vec(word, sg=0)
    vectorSize=model.wv.vector_size
    wv_res=np.zeros(vectorSize)
    print(wv_res)
    ctr=1
    for w in word:
        if w in model.wv:
            ctr+=1
            wv_res+=model.wv[w]
    wv_res=wv_res/ctr
    return wv_res'''

def MeanEmbeddingVectorizer(model, data):
    print("【MEAN EMBEDDING VECTORIZER】")
    docVec_list=[]

    for i in data:
        count = 0.0
        docVec = 0.0

        for word in i:
            count += 1
            docVec += model.wv[word]

        docVec /= count
        docVec_list.append(docVec)

    print(docVec_list[:])
    return docVec_list

def w2v(trainData,labels):

    data=trainData[:]
    data=np.array(data)
    label=labels[:]
    label=np.ravel(label)

    model_cbow = Word2Vec(data, sg=0) #cbow
    print("w2v model:")
    print(model_cbow)
    print("==========================================")
    '''print("most similar")
    model=model_cbow.wv.most_similar(positive=['exit'],topn=10)
    print(model)
    print("==========================================")'''

    '''words=list(model_cbow.wv.index_to_key)
    print(words)
    #print(model_cbow.wv)
    model_cbow.save('model.bin')
    new_cbow=Word2Vec.load('model.bin')'''
    
    new_cbow=MeanEmbeddingVectorizer(model_cbow,data)
    X_cbow_train,X_cbow_test,Y_cbow_train,Y_cbow_test= train_test_split(new_cbow,label,test_size=0.2)

    #logistic regression
    reg=LinearRegression()
    reg.fit(X_cbow_train,Y_cbow_train)
    Y_cbow_predict=reg.predict(X_cbow_test)
    print("[linear regression][w2v]")
    print("coefficients:",reg.coef_)
    print("variance score:",accuracy_score(Y_cbow_test,Y_cbow_predict))
    print("==========================================")

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

    X_train,X_test,Y_train,Y_test= train_test_split(data,label,test_size=0.2)

    #knn
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

    tfidf(train,label)
    #w2v(train,label)