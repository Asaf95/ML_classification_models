 
## 4277410  Machine Learning task 4 Classification
### Asaf Ben-Menachem

table of contents<br>
* Task requirements
* The flow of the code
* The code

### Task requirements
You are provided with a list of clients.
<br>For each client, you are given with their height, salary, and whether this client made a purchase.
<br>Your task is to implement a set of classifiers, to predict if a future client will make a purchase. 

For your prediction, you should train the following models:
* K-nn (based on Euclidian distance)
* Logistic regression 
* Support Vector Classifier
    - Linear SVC 
    - Polynomial SVC of degree m (m will be set in the code)
    - Gaussian SVC  

#### What to submit? 
Python code implementing the task and the data generated for the scientific report. <br> 
A scientific report (as a Word file): 

* For the k-nn, evaluate the effect of k on the f1-score. For this evaluation, make 1,000 random train-test splits and provide the mean and STD f1-score for each k in the range 1 to 20 (included). Specify what value of K will be chosen and why 
* F1-score for the Logistic regression and Linear SVC models. For this evaluation, make 1,000 random train-test splits and provide the mean and STD f1-score (for each model). 
* For the polynomial SVC, evaluate the effect of the degree (m) on the f1-score. For this evaluation, make 1,000 random train-test splits and provide the mean and STD f1-score for each m in the range 2 to 5 (included). Specify what value of m will be chosen and why 
* For the gaussian SVC, evaluate the effect of C on the f1-score. For this evaluation, make 1,000 random train-test splits and provide the mean and STD f1-score for each c in the following set of values (0.2,0.5,1.2,1.8,3). Specify what value of C will be chosen and why    
To make a fair analysis (and future comparison of the models), make sure to use the same 1,000 
train-test splits to evaluate all models and their hyperparameters. 

## The flow of the code
The basic idea is the split X and y to train and test and StandardScaler one time and use them for all the models. 
<br>
The results from the model will be stored in global list, for each  model we need to Different list because we will fill them in at the same time. <br>
And go over this for 1000 times (by defining the number_of_run = 1000)

    def main(x, y):
        # creting all the variables that is needed for this model in global mode (to be used in functions)
        global number_of_run, run, rows_logis_reg, rows_knn, rows_svc_l, rows_svc_p, rows_svc_r
        number_of_run = 1000; run = 1
        rows_logis_reg = []; rows_svc_l = []; rows_knn = []; rows_svc_p = []; rows_svc_r = []

        # Run 1000 runs with the same split data to all models
        for run in range(1, number_of_run + 1):
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=None)
            sc = StandardScaler()
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)
            "GO over all  the models with the same "
            k_nn_result = k_nn(x_train, x_test, y_train, y_test)
            log_re_result = log_re(x_train, x_test, y_train, y_test)
            svc_l_result = svc_l(x_train, x_test, y_train, y_test)
            svc_g_result = svc_g(x_train, x_test, y_train, y_test)
            svc_p_result = svc_p(x_train, x_test, y_train, y_test)

In every model (that will each in a different function) the function will test the F1Score and add it to a new dictionary <br> 
now it depends on if we are checking inner variable into the model or not (for example in K_NN we need to check 20 different 20 variables)
if there  is need to check inner virables indise the function we will do inner for and collect the result to the dictionary <br>

        value_dic = {}  # this will storage the
        for i in range(1, 21):
            classifier = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
            classifier.fit(x_train, y_train)
            test_pred = classifier.predict(x_test)
            sum_of_f1 = f1_score(y_test, test_pred, average='weighted')
            value = 'k_nn neighbors = ' + str(i)
            value_dic[value] = sum_of_f1
        rows_knn.append(value_dic)

that will collect all the scores for all the 1000 runs. <br>
When reached the 1000 run the function will return the mean and the std (*1)  of the 1000 runs (*2).<dr>

        if number_of_run == run:  # when th
            result = bestr(rows_knn)
            plotme(x_train, x_test, y_train, y_test, classifier,'k_nn')
            return result


After going through the 1000 runs for all the models the results will be used to one DataFrame and print it to the user.

the function sorts the best mean (list with more than one column) and return it to the function.<br>
statistic measures |    k_nn_result | log_re_result | svc_l_result | svc_g_result | svc_p_result <br>
mean__________               |  0.91190300     |  0.818174000   |  0.82387100    |  0.916804000  |    0.914802 <br>
std____________                |  0.01249500     |  0.014740000    |  0.01476500 |     0.011327000 |      0.011442  <br>


        df = pd.concat([k_nn_result, log_re_result, svc_l_result, svc_g_result, svc_p_result], axis=1,
                       keys=['k_nn_result', 'log_re_result', 'svc_l_result', 'svc_g_result', 'svc_p_result'], sort=True)
        print(df)

*1: for creating that mean and the std it will be by using the function bestr(), that function gets the list<dr>
convert it to DataFrame and use agg for calculating the mean and the str.
<br>




    def bestr(rows) -> pd.DataFrame:
        """
        bestr() this function makes the mean and the std of each
        :param rows: this variable has all the 1000 runs that been done in this code
        """
        old_df = pd.DataFrame(rows)
        new_df = pd.DataFrame()
        new_df['mean'] = old_df.agg(func=mean, axis=0)
        new_df['std'] = old_df.agg(func=np.std)
        result = new_df.sort_values(by='mean', ascending=False)
        #result = result.reset_index()
        return result.iloc[0]

*2: we will define the run and the number_of_run as a global variable and add an if condition to the model function. <dr>



## The results
Using a function learned in class to show the graphs of all the models


    def plotme(x_train, x_test, y_train, y_test, model, name) -> None:
        """
         plotme predicted according to his  models a target prediction value based on independent variables
         :param x_train: This includes your all independent variables,these will be used to train the model
         :param x_test: This is remaining  portion of the independent variables from the data,
                will be used to make predictions to test the accuracy of the model.
         :param y_train: This is your dependent variable which needs to be predicted by this model
         :param y_test: This data has category labels for your test data,
                these labels will be used to test the accuracy between actual and predicted categories.
         :param model: the model that is been used
         :param name: the name of the model that is been used
        """
        X_set, y_set = x_test, y_test
    
        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                             np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1,
                                       step=0.01))
        plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
                     cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title(name)
        plt.xlabel('Height')
        plt.ylabel('Salary')
        plt.legend()
        plt.show()

the plots are:

<img height="300" src="C:\Users\asaf\PycharmProjects\task5\graph\K_NN.png" title="K_nn" width="500.4"/>
<br> </br>

<img height="300" src="C:\Users\asaf\PycharmProjects\task5\graph\log_re.png" title="Log_r" width="500.4"/>
<br> </br>

<img height="300" src="C:\Users\asaf\PycharmProjects\task5\graph\SVC_G.png" title="SVC_G" width="500.4"/>
<br> </br>

<img height="300" src="C:\Users\asaf\PycharmProjects\task5\graph\SVC_L.png" title="SVC_L" width="500.4"/>
<br> </br>

<img src="[url=http://www.up2me.co.il/v.php?file=41789189.png][img]http://www.up2me.co.il/thumbs/41789189.png[/img][/url]" width="100" height="100">

<img height="300" src="C:\Users\asaf\PycharmProjects\task5\graph\SVC_P.png" title="SVC_P" width="500.4"/>
<br> </br>
![K_NN](https://github.com/Asaf95/task5/blob/main/graph/K_NN.png)
 graph/K_NN.png
