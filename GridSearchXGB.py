# XGBoost on Otto dataset, Tune n_estimators and max_depth
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import xgboost as xgb
from scipy.optimize import differential_evolution
import matplotlib
matplotlib.use('Agg')

#Best: 0.327006 using {'n_estimators': 200, 'learning_rate': 0.07, 'max_depth': 8}
import sys

from matplotlib import pyplot
import numpy
global THRESHOLD
THRESHOLD = 0.19

#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
def xg_f1(y,t):
    t = t.get_label()
    #y is actual, t is predicted
    y_bin = [1. if t_cont > THRESHOLD  else 0. for t_cont in y] # binaryzing your output
    return 'f1_score', f1_score(t,y_bin)
    #f1_score(y_true, y_pred, 


def fnGridCVWrapper(*pArgs, **pAdditional):
        global THRESHOLD
        
        pmax_depth=int(pArgs[0][0])
        plearning_rate=(pArgs[0][1])
        psubsample=(pArgs[0][2])
        pAlpha=(pArgs[0][3])
        pLambda=(pArgs[0][4])
        pEstimators =int(pArgs[0][5])
        pnumboostRounds =int(pArgs[0][6])
        pGamma =int(pArgs[0][7])
        THRESHOLD=pArgs[0][8]

        pDataTrain=pArgs[1]
        pLabels =pArgs[2]
        pModel =pArgs[3]

        result= fnGridSearchCV(pDataTrain, pLabels, pModel,
            pmax_depth ,plearning_rate ,psubsample, pAlpha, pLambda , pEstimators, pnumboostRounds ,pGamma   )
        
        return result

def fnOptSearch(pDataTrain, pLabels, pModel):
    lBounds=[(3,20),(.01,.99),(.01,.95),(0,1),(2,22),(100,800),(100,600),(.1,10),(.19,.25)]
    #reg_alphas =[.05,.1,.2]
    #reg_lambda =[5,7,8,10,15]
    best_score =.25
    #fnMainWrapper([  3.54437387e+01,   1.35031494e+05,   2.39349986e-02])
    #fnRunForecast("SPYdfQuotes.csv","SPYdfQuotesTest.csv",False)
    #(array([  3.54437387e+01,   1.35031494e+05,   2.39349986e-02]),)
    #.33 r2 on 5.28
    
    
    result=differential_evolution(func=fnGridCVWrapper,bounds=lBounds,disp=1,
                     args=(pDataTrain, pLabels, pModel))
    print (result)


def fnGridSearchCV(pDataTrain, pLabels, pModel,pmax_depth ,plearning_rate ,psubsample,
                 pAlpha, pLambda, pEstimators,pnum_boost_round, pGamma ):
    # load data
    #data = read_csv('train.csv')
    #dataset = pDataTrain #data.values
    # split data into X and y
    X = pDataTrain#dataset[:,0:94]
    dtrain =X
    #y = dataset[:,94]
    # encode string class values as integers
    #label_encoded_y = LabelEncoder().fit_transform(y)
    # grid search
    model =XGBClassifier( colsample_bylevel=1,
    colsample_bytree=0.8,gamma=0.7, learning_rate=0.1, max_delta_step=0, max_depth=15,
    min_child_weight=2, missing=None, n_estimators=500, nthread=4,objective='binary:logistic',
    reg_alpha=0.2, reg_lambda=1,scale_pos_weight=1, seed=27, silent=True, subsample=0.8)

    n_estimators = [ pEstimators] #50, 100, 150,
    gammas =[.7]
    min_child_weights=None #[5,10,20,30,50]
    learning_rates=[plearning_rate] #[.4,.5,.7,.8,.9]
    subsamples =[psubsample] #[.02,.05,.1,.18,.25]
    max_depth =[pmax_depth] # [ 8,13,15,18,20,22]
    reg_alphas =[pAlpha]
    reg_lambda =[pLambda]

    print(max_depth)
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators,min_child_weight =min_child_weights,
                      learning_rate=learning_rates,subsample=subsamples, 
                      reg_alpha=reg_alphas,reg_lambda=reg_lambda)


    param_grid = dict(learning_rate=learning_rates,max_depth=max_depth,  subsample=subsamples)

    f1_scorer = make_scorer(xg_f1, greater_is_better=True)

    #kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
    #model =pModel
    #grid_search = GridSearchCV(model, param_grid, scoring='f1', n_jobs=3, cv=kfold, verbose=1)
    #grid_result = grid_search.fit(X, pLabels)
    xgb_params = {
    'eta': plearning_rate, #learning rate
	'objective': "reg:logistic",
    'max_depth': pmax_depth, #6
	'min_child_weight': 10,
	'gamma': pGamma,
	'alpha':pAlpha,
	'lambda': pLambda,
    'subsample': psubsample,
    'colsample_bytree': 0.95,
    'silent': 0,
    'n_estimators':pEstimators
    }

    # Uncomment to tune XGB `num_boost_rounds`
    cvResult = xgb.cv(xgb_params, dtrain, num_boost_round=pnum_boost_round,feval=xg_f1,maximize=True,nfold=3,
                          early_stopping_rounds=40, verbose_eval=30, as_pandas=True, seed=10)
    #grid_result = grid_search.predict_proba(X, pLabels)
    lenResults=len(cvResult['test-f1_score-mean'])
    print ("best using ",xgb_params,pnum_boost_round, cvResult['test-f1_score-mean'][lenResults-1], THRESHOLD)
    sys.stdout.flush()
    # summarize results
    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']

    #for mean, stdev, param in zip(means, stds, params):
	   # print("%f (%f) with: %r" % (mean, stdev, param))
    ## plot results
    #scores = numpy.array(means).reshape(len(max_depth), len(n_estimators))
    #for i, value in enumerate(max_depth):
    #    pyplot.plot(n_estimators, scores[i], label='depth: ' + str(value))
    #pyplot.legend()
    #pyplot.xlabel('n_estimators')
    #pyplot.ylabel('F1 score')
    #pyplot.savefig('n_estimators_vs_max_depth.png')

    #return -grid_result.best_score_
    return -cvResult['test-f1_score-mean'][lenResults-1]