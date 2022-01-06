    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from matplotlib.colors import ListedColormap
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from statistics import mean
    from matplotlib.axes._axes import _log as matplotlib_axes_logger
    matplotlib_axes_logger.setLevel('ERROR')

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

        plt.show()


    def svc_l(x_train, x_test, y_train, y_test) -> pd.DataFrame:
        """
         svc_l predicted according to his  models a target prediction value based on independent variables
         :param x_train: This includes your all independent variables,these will be used to train the model
         :param x_test: This is remaining  portion of the independent variables from the data,
                will be used to make predictions to test the accuracy of the model.
         :param y_train: This is your dependent variable which needs to be predicted by this model
         :param y_test: This data has category labels for your test data,
                these labels will be used to test the accuracy between actual and predicted categories.
         :return: pd.DataFrame that hold the indices of regression quality for the Regression
                    in addition to the other regressions indices
         """
        value_dic = {}
        model = LinearSVC(loss='hinge', dual=True, max_iter=2000)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        f1_s = f1_score(y_test, y_pred)
        value = 'svc_l'
        value_dic[value] = f1_s
        rows_svc_l.append(value_dic)
        if number_of_run == run:  # when th
            result = bestr(rows_svc_l)
            plotme(x_train, x_test, y_train, y_test, model, 'svc_l')
            return result


    def svc_p(x_train, x_test, y_train, y_test) -> pd.DataFrame:
        """
         svc_p predicted according to his models a target prediction value based on independent variables
         :param x_train: This includes your all independent variables,these will be used to train the model
         :param x_test: This is remaining  portion of the independent variables from the data,
                will be used to make predictions to test the accuracy of the model.
         :param y_train: This is your dependent variable which needs to be predicted by this model
         :param y_test: This data has category labels for your test data,
                these labels will be used to test the accuracy between actual and predicted categories.
         :return: pd.DataFrame that hold the indices of regression quality for the Regression
                    in addition to the other regressions indices
         """
        value_dic = {}  # this will storage the
        for i in range(1, 6):
            model = SVC(kernel='poly', degree=i, gamma='auto', coef0=1, C=5)
            model.fit(x_train, y_train)
            test_pred = model.predict(x_test)
            sum_of_f1 = f1_score(y_test, test_pred, average='weighted')
            value = 'SVC_P degree = ' + str(i)
            value_dic[value] = sum_of_f1
        rows_svc_p.append(value_dic)
        if number_of_run == run:  # when th
            result = bestr(rows_svc_p)
            plotme(x_train, x_test, y_train, y_test, model, 'svc_p')

            return result


    def svc_g(x_train, x_test, y_train, y_test) -> pd.DataFrame:
        """
         svc_g predicted according to his models a target prediction value based on independent variables
         :param x_train: This includes your all independent variables,these will be used to train the model
         :param x_test: This is remaining  portion of the independent variables from the data,
                will be used to make predictions to test the accuracy of the model.
         :param y_train: This is your dependent variable which needs to be predicted by this model
         :param y_test: This data has category labels for your test data,
                these labels will be used to test the accuracy between actual and predicted categories.
         :return: pd.DataFrame that hold the indices of regression quality for the Regression
                    in addition to the other regressions indices
         """
        dic_gas = {}
        val_gas = [0.2, 0.5, 1.2, 1.8, 3]
        for j in val_gas:
            model = SVC(kernel='rbf', gamma=j)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            sco = f1_score(y_test, y_pred, average='weighted')
            value = 'svc_g val_gas = ' + str(j)
            dic_gas[value] = sco
        rows_svc_r.append(dic_gas)
        if run == number_of_run:
            result = bestr(rows_svc_r)
            plotme(x_train, x_test, y_train, y_test, model,'svc_g')

            return result


    def log_re(x_train, x_test, y_train, y_test) -> pd.DataFrame:
        """
         log_re predicted according to his models a target prediction value based on independent variables
         :param x_train: This includes your all independent variables,these will be used to train the model
         :param x_test: This is remaining  portion of the independent variables from the data,
                will be used to make predictions to test the accuracy of the model.
         :param y_train: This is your dependent variable which needs to be predicted by this model
         :param y_test: This data has category labels for your test data,
                these labels will be used to test the accuracy between actual and predicted categories.
         :return: pd.DataFrame that hold the indices of regression quality for the Regression
                    in addition to the other regressions indices
         """
        value_dic = {}
        log_reg = LogisticRegression(solver='liblinear')
        log_reg.fit(x_train, y_train)
        y_pred = log_reg.predict(x_test)
        f1_s_lr = f1_score(y_test, y_pred)
        value = 'log_re'
        value_dic[value] = f1_s_lr
        rows_logis_reg.append(value_dic)
        if number_of_run == run:  # when th
            result = bestr(rows_logis_reg)
            plotme(x_train, x_test, y_train, y_test, log_reg, 'log_re')
            return result


    def k_nn(x_train, x_test, y_train, y_test) -> pd.DataFrame:
        """
         k_nn predicted according to his models a target prediction value based on independent variables
         :param x_train: This includes your all independent variables,these will be used to train the model
         :param x_test: This is remaining  portion of the independent variables from the data,
                will be used to make predictions to test the accuracy of the model.
         :param y_train: This is your dependent variable which needs to be predicted by this model
         :param y_test: This data has category labels for your test data,
                these labels will be used to test the accuracy between actual and predicted categories.
         :return: pd.DataFrame that hold the indices of regression quality for the Regression
                    in addition to the other regressions indices
         """
        value_dic = {}  # this will storage the
        for i in range(1, 21):
            classifier = KNeighborsClassifier(n_neighbors=i, metric='minkowski', p=2)
            classifier.fit(x_train, y_train)
            test_pred = classifier.predict(x_test)
            sum_of_f1 = f1_score(y_test, test_pred, average='weighted')
            value = 'k_nn neighbors = ' + str(i)
            value_dic[value] = sum_of_f1
        rows_knn.append(value_dic)
        if number_of_run == run:  # when th
            result = bestr(rows_knn)
            plotme(x_train, x_test, y_train, y_test, classifier,'k_nn')
            return result


    def main(x, y):
        # creting all the variables that is needed for this model in global mode (to be used in functions)
        global number_of_run, run, rows_logis_reg, rows_knn, rows_svc_l, rows_svc_p, rows_svc_r
        number_of_run = 10; run = 1
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
        df = pd.concat([k_nn_result, log_re_result, svc_l_result, svc_g_result, svc_p_result], axis=1,
                       keys=['k_nn_result', 'log_re_result', 'svc_l_result', 'svc_g_result', 'svc_p_result'], sort=True)
        print(df)


    if __name__ == '__main__':
        CovidData = pd.read_csv('dataset.csv', encoding='')
        CovidData = CovidData.dropna(axis='columns')
        X = CovidData[['Height', 'Salary']].values
        y = CovidData['Purchased'].values
        main(X, y)
