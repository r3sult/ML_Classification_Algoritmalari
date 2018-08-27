def main():
   
    devam = True
    while (devam):
        
        X,y = DataRead()
        X_train,X_test,y_train,y_test = Split(X,y)
        
        print(50*"=")
        print("Data seti görmek için 0\nLogistic Regression sonucu için 1\nKNN sonucu için 2\nSVC sonucu için 3\nNaive Bayes sonucu için 4\nDecision Tree sonucu için 5\nRandom Forest sonucu için 6\nÇıkmak için 'h'")
        print(50*"=")
        
        cevap = input("Hangi işlemi yapmak istersiniz ?")
        
        if cevap == "h":
            
            devam=False
            
        elif cevap == "0":
            
            print(data)
            
        elif cevap == "1":
            
            lr_y_pred = LogisticReg(X_train,y_train,X_test)
            lr_cm = Acc(y_test,lr_y_pred)
            print("\nLogistic Regression Sonuç:"+str(lr_cm)+"\n")
           
        elif cevap == "2":
            
            knn_y_pred = KNN(X_train,y_train,X_test)
            knn_cm = Acc(y_test,knn_y_pred)
            print("\nKNN Sonuç:"+str(knn_cm)+"\n")
            
        elif cevap == "3":
            
            svc_y_pred = Svc(X_train,y_train,X_test)
            svc_cm = Acc(y_test,svc_y_pred)
            print("\nSVC Sonuç:"+str(svc_cm)+"\n")
            
        elif cevap == "4":
            
            gnb_y_pred = NaiveBayes(X_train,y_train,X_test)
            gnb_cm = Acc(y_test,gnb_y_pred)
            print("\nNaive Bayes Sonuç:"+str(gnb_cm)+"\n")
            
        elif cevap == "5":
            
            dt_y_pred = DecisionTree(X_train,y_train,X_test)
            dt_cm = Acc(y_test,dt_y_pred)
            print("\nDecison Tree Sonuç:"+str(dt_cm)+"\n")
            
        elif cevap == "6":
            
            rf_y_pred = RandomForest(X_train,y_train,X_test)
            rf_cm = Acc(y_test,rf_y_pred)
            print("\nRandom Forest Sonuç:"+str(rf_cm)+"\n")
        
        else:
            
            print("\nSadece aşşağıdaki işlemler yapılabilir...")
        
def DataRead():
    
    import pandas as pd 

    global data
    data = pd.read_excel("Iris.xls")
    X = data.iloc[:,0:-1].values
    y = data.iloc[:,-1].values
    return X,y


def Split(X,y):
    
    from sklearn.cross_validation import train_test_split

    X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=1/3,random_state=0)
    return X_train,X_test,y_train,y_test


def LogisticReg(X_train,y_train,X_test):
    
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    return y_pred


def KNN (X_train,y_train,X_test):
    
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=5,metric="minkowski")
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    return y_pred


def Svc (X_train,y_train,X_test):
    
    from sklearn.svm import SVC

    svc = SVC(kernel="rbf")
    svc.fit(X_train,y_train)
    y_pred = svc.predict(X_test)
    return y_pred


def NaiveBayes(X_train,y_train,X_test):
    
    from sklearn.naive_bayes import GaussianNB

    gnb = GaussianNB()
    gnb.fit(X_train,y_train)
    y_pred = gnb.predict(X_test)
    return y_pred


def DecisionTree(X_train,y_train,X_test):
    
    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier(criterion="entropy")
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)
    return y_pred


def RandomForest(X_train,y_train,X_test):
    
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=10,criterion="gini")
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    return y_pred


def Acc (y_test,y_pred):
    
    from sklearn import metrics

    cm = metrics.confusion_matrix(y_test,y_pred)
    
    basari = 0
    deger = 0
    
    for i in cm:
        basari +=i[deger]
        deger+=1
    
    basari = basari/(len(data.index)/3) 
    return basari


if __name__ == '__main__':
    main()
