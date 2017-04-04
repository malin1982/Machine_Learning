import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

def process_data(data):
    '''
    remove redundant columns
    '''
    #rems = ['Id', 'Soil_Type7', 'Soil_Type8', 'Soil_Type15', 'Soil_Type25']
    rems = ['Id', 'Soil_Type7', 'Soil_Type15']
#     #Add constant columns as they don't help in prediction process
#     for c in data.columns:
#         if data[c].std() == 0: #standard deviation is zero
#             rem.append(c)

    #drop the columns
    for rem in rems:
        data.drop(rem,axis=1,inplace=True)


    return data
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

def normalize_train_data(dataset):
    r, c = dataset.shape
    array = dataset.values
    X_all = array[:,0:(c-1)]
    y_all = array[:,(c-1)]
    size = 10
    X_num = X_all[:,0:size]
    X_cat = X_all[:,size:]

    X_num = StandardScaler().fit_transform(X_num)
    X_num = MinMaxScaler().fit_transform(X_num)
    X_num = Normalizer().fit_transform(X_num)

    X_all_scaled = np.concatenate((X_num, X_cat), axis=1)

    return X_all_scaled, y_all

def normalize_test_data(dataset):
    r, c = dataset.shape
    X_all = dataset.values
    y_all = []
    size = 10
    X_num = X_all[:,0:size]
    X_cat = X_all[:,size:]

    X_num = StandardScaler().fit_transform(X_num)
    X_num = MinMaxScaler().fit_transform(X_num)
    X_num = Normalizer().fit_transform(X_num)

    X_all_scaled = np.concatenate((X_num, X_cat), axis=1)

    return X_all_scaled, y_all

def train_extract(train, test):
    X_train, y_train = normalize_train_data(train)
    X_test, y_test = normalize_train_data(test)

    return X_train, y_train, X_test, y_test

def perform_cross_validation(model, train):
    '''Performs a kfold cross validation of a given model'''
    kfold_train_test = []
    extracted_features = []
    kf = StratifiedKFold(train["Cover_Type"], n_folds=10)
    for train_index, test_index in kf:
        train_kfold = train.loc[train_index]
        test_kfold = train.loc[test_index]
        extracted_features.append(tuple(train_extract(train_kfold, test_kfold)))
    score_count = 0
    score_total = 0.0
    submission = []
    print (model)
    for X_train, y_train, X_test, y_test in extracted_features:

        model.fit(X_train, y_train)
        #score = model.score(X_test, y_test)
        predictions = model.predict(X_test)
        score = f1_score(y_test, predictions, average='micro')
        test_data = pd.DataFrame({'id': y_test, 'predictions': predictions})
        submission.append(test_data)
        score_count += 1
        score_total += score
        print("Kfold score " + str(score_count) + ": " + str(score))
    average_score = score_total/float(score_count)
    print("Average score: " + str(average_score))
    return submission

def perform_predictions(model, train, test):
    '''
    Performs the final prediction on test dataset
    '''
    global Id

    submission = []
    X_train, y_train = normalize_train_data(train)
    X_test, y_test = normalize_test_data(test)

    model.fit(X_train, y_train)
    final_predictions = model.predict(X_test)

    test_data = pd.DataFrame({'Id': Id, 'Cover_Type': final_predictions})
    submission.append(test_data)
    #submission = pd.DataFrame({'id': test_clean['id'], 'prediction': weighted_prediction})

    return submission

def tuner(model, param_grid, dataset):
    X_train, y_train, _, _ = train_extract(dataset, dataset)
    tuning_scorer = make_scorer(score, greater_is_better = True)

    tuner_model = GridSearchCV(estimator=model,
                                param_grid=param_grid,
                                scoring=tuning_scorer,
                                verbose=10,
                                n_jobs=-1,
                                iid=True,
                                refit=True,
                                cv=5)

    tuner_model.fit(X_train, y_train)
    print("Best score: %0.3f" % tuner_model.best_score_)
    print("Best parameters set:")
    best_parameters = tuner_model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def score(y, y_pred):

    y_true = np.array(y, dtype=int)
    y_predict = np.array(y_pred, dtype=int)

    from sklearn.metrics import f1_score

    return f1_score(y_true, y_predict, average='micro')

def to_csv(df,out):
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    df.to_csv(out, index=False)
    return


if __name__ == '__main__':
    print ('Loading data...')
    train_raw = pd.read_csv('data/train.csv')
    test_raw = pd.read_csv('data/test.csv')
    Id = test_raw['Id']
    print ('Cleaning data...')
    train_clean = process_data(train_raw)
    test_clean = process_data(test_raw)


    print ('Training...')

    seed = 19
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    model_0 = KNeighborsClassifier(n_jobs=-1, algorithm='auto',n_neighbors=1)
    model_1 = RandomForestClassifier(n_jobs=-1, n_estimators=18, random_state=seed)
    model_2 = GradientBoostingClassifier(max_depth=8, random_state=seed)

    cv_pred_1 = perform_cross_validation(model_0, train_clean)
    cv_pred_2 = perform_cross_validation(model_1, train_clean)
    cv_pred_3 = perform_cross_validation(model_2, train_clean)

    print ('Predicting...')
    pred_1 = perform_predictions(model_0, train_clean, test_clean)
    pred_2 = perform_predictions(model_1, train_clean, test_clean)
    pred_3 = perform_predictions(model_2, train_clean, test_clean)

    print ('Ensembling...')
    cv_preds = [cv_pred_1, cv_pred_2, cv_pred_3]
    wt_final = []
    for i in range(100):
        w = np.random.dirichlet(np.ones(3),size=1)
        wt_final.append(w)
    max_average_score = 0.67
    max_weights = None
    for wt in wt_final:
        total_score = 0
        for i in range(9):
            y_true = cv_preds[0][i]['id']
            weighted_prediction = sum([wt[0][x] * cv_preds[x][i]['predictions'].astype(int).reset_index() for x in range(3)])
            weighted_prediction = [round(p) for p in weighted_prediction['predictions']]
            total_score += score(y_true, weighted_prediction)
        average_score = total_score/9.0
        if (average_score > max_average_score):
            max_average_score = average_score
            max_weights = wt
    print ('Best set of weights: ' + str(max_weights))
    print ('Corresponding score: ' + str(max_average_score))
    preds = [pred_1, pred_2, pred_3]
    weighted_prediction = sum([max_weights[0][x] * preds[x][0]['Cover_Type'].astype(int) for x in range(3)])
    weighted_prediction = [int(round(p)) for p in weighted_prediction]
    submission = pd.DataFrame({'Id': Id, 'Cover_Type': weighted_prediction})
    #submission.to_csv('submission.csv', index=False)
    to_csv(submission, 'submission.csv')
    print('Output submission file')
