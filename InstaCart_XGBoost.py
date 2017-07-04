# This script considers all the products a user has ordered
#
# We train an XGBoost model computing the probability of reorder on the "training" data
#
# For the submission, we keep the products from orders that have a probability of
# reorder higher than a threshold: THRESHOLD
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from GridSearchXGB import fnGridSearchCV, fnOptSearch
sizeDataFrame =2000

THRESHOLD = 0.19

def fnGetUserAndProductFeatures(pDfPriors, pDfOrders):
    #1. 'nb_orders - need number of times user bought the product
    dfnb_orders =pDfPriors.groupby(['product_id','user_id']).size().to_frame('UP_orders')

    dfnb_orders['product_id'] =dfnb_orders.index.get_level_values(0)
    dfnb_orders['user_id'] =dfnb_orders.index.get_level_values(1)
    #now have a data frame with user id, product id and count

    #2. last_order_id for each product by user takes a long time
    #should this be last order id for user, or last order id per prod per user?
    #dfMaxOrderId=pDfPriors.groupby(['product_id','user_id'])['order_id'].max().to_frame('UP_last_order_id')
    #dfMaxOrderId['product_id'] =dfMaxOrderId.index.get_level_values(0)
    #dfMaxOrderId['user_id'] =dfMaxOrderId.index.get_level_values(1)
    dfMaxOrderId =pDfPriors.groupby(['user_id'])['user_id','product_id','order_id','order_number'].tail(1)
    dfMaxOrderId['UP_last_order_id'] =dfMaxOrderId['order_id']
    #3. avg position in cart:

    dfAddtoCartMean=pDfPriors.groupby(['product_id','user_id'])['add_to_cart_order'].mean().to_frame('UP_average_pos_in_cart')
    dfAddtoCartMean['product_id'] =  dfAddtoCartMean.index.get_level_values(0)
    dfAddtoCartMean['user_id'] =  dfAddtoCartMean.index.get_level_values(1)

    #user_id does not exist in 1 of these dataframes
    dfResult =dfnb_orders.merge(dfMaxOrderId,on =['product_id','user_id'], suffixes =('_del',''))
    del dfnb_orders
    #dfResult.drop('product_id_del',inplace =True)
    #dfResult.drop('user_id_del',inplace =True)
    dfResult =dfResult.merge(dfAddtoCartMean,on =['product_id','user_id'], suffixes =('_del',''))

    del dfAddtoCartMean
    #dfResult.drop('product_id_del',inplace =True)
    #dfResult.drop('user_id_del',inplace =True)

    #4.compute count of product ordered in last 2 orders:
    #if a product was orderd in last 2 orders, then 2, if only 1 of last 2 orders then 1
    dfTempLast2Orders =pDfOrders.groupby(['user_id'])['user_id','order_number'].tail(2)
    #join on priors only get last 2 orders by product, user
    dfTempLast2Orders =dfTempLast2Orders.merge(pDfPriors,on =['order_number','user_id'],suffixes =('_del',''))

    ##dfTempLast2Orders.drop('product_id_del',inplace =True)
    #dfTempLast2Orders.drop('user_id_del',inplace =True)

    dfTempLast2Orders =dfTempLast2Orders.groupby(['product_id','user_id']).size().to_frame('NumTimesInLast2Orders')

    dfTempLast2Orders['product_id'] =dfTempLast2Orders.index.get_level_values(0)
    dfTempLast2Orders['user_id'] =dfTempLast2Orders.index.get_level_values(1)


    dfResult =dfResult.merge(dfTempLast2Orders,on =['product_id','user_id'], suffixes =('_del',''))

    del dfTempLast2Orders
    #dfResult.drop('product_id_del',inplace =True)
    #dfResult.drop('user_id_del',inplace =True)

    return dfResult


def fnGetProductReorderRatioAcrossUsers(pDfPriors):
    ''' ratio is # users who reordered a product div by # users who ordered the product'''
    dfInit =pDfPriors.groupby(['product_id','user_id']).size().to_frame()
    df =dfInit

    df['product_id'] =df.index.get_level_values(0)

    #get number of users who ordered the product
    dfCntUsersWhoOrderedProduct =df.groupby('product_id').size().to_frame('NumOrdersAcrossUsers')
    dfCntUsersWhoOrderedProduct['product_id'] =dfCntUsersWhoOrderedProduct.index.get_level_values(0)

    dfInitReorder =pDfPriors[pDfPriors['reordered']==1].groupby(['product_id','user_id']).size().to_frame()
    df =dfInitReorder

    df['product_id'] =df.index.get_level_values(0)

    #get number of users who ordered the product
    dfCntUsersWhoREOrderedProduct =df.groupby('product_id').size().to_frame('NumReordersAcrossUsers')
    dfCntUsersWhoREOrderedProduct['product_id'] =dfCntUsersWhoREOrderedProduct.index.get_level_values(0)
    
    dfResult = dfCntUsersWhoOrderedProduct.join(dfCntUsersWhoREOrderedProduct, on='product_id',how='left',rsuffix='_re')

    dfResult['ReorderRatioAcrossUsers'] =dfResult['NumReordersAcrossUsers']/dfResult['NumOrdersAcrossUsers']

    dfResult['ReorderRatioAcrossUsers'].fillna(0, inplace=True)

    dfResult =dfResult[['product_id','ReorderRatioAcrossUsers']]
    return dfResult

def xg_f1(y,t):
    t = t.get_label()
    y_bin = [1. if y_cont > THRESHOLD else 0. for y_cont in y] # binaryzing your output
    return 'f1',f1_score(t,y_bin)

def fEncodeValues(pData):
    for c in pData.columns:
        if pData[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(pData[c].values)) 
            pData[c] = lbl.transform(list(pData[c].values))
    
    return pData



def clearNAs(pDf):
    for fld in pDf.columns:
        pDf[fld].fillna(0, inplace=True)
    return pDf




def fnMain():
    #import lightgbm as lgb
    IDIR = '../input/'
    IDIR = 'C:\\KAGGLE\\InstaCart\\'
    blnLoadCSVFiles =True

    if blnLoadCSVFiles:
	    print('loading prior')
	    priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
				    'order_id': np.int32,
				    'product_id': np.uint16,
				    'add_to_cart_order': np.int16,
				    'reordered': np.int8})

    if blnLoadCSVFiles:
	    print('loading train')
	    train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
				    'order_id': np.int32,
				    'product_id': np.uint16,
				    'add_to_cart_order': np.int16,
				    'reordered': np.int8})

	    print('loading orders')
	    orders = pd.read_csv(IDIR + 'orders.csv', dtype={
			    'order_id': np.int32,
			    'user_id': np.int32,
			    'eval_set': 'category',
			    'order_number': np.int16,
			    'order_dow': np.int8,
			    'order_hour_of_day': np.int8,
			    'days_since_prior_order': np.float32})

	    print('loading products') #note that order id comes in later, not in this file**
	    products = pd.read_csv(IDIR + 'products.csv', dtype={
			    'product_id': np.uint16,
			    'order_id': np.int32,
			    'aisle_id': np.uint8,
			    'department_id': np.uint8},
			    usecols=['product_id', 'aisle_id', 'department_id'])

	    print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
	    print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
	    print('train {}: {}'.format(train.shape, ', '.join(train.columns)))

	    ###

    if blnLoadCSVFiles:
	    print('computing product f')
	    prods = pd.DataFrame()
	    prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
	    prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
	    prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
	    products = products.join(prods, on='product_id')
	    products.set_index('product_id', drop=False, inplace=True)
	    del prods


	    print('add order info to priors')
	    orders.set_index('order_id', inplace=True, drop=False)
	    priors = priors.join(orders, on='order_id', rsuffix='_')
	    priors.drop('order_id_', inplace=True, axis=1)

	    ### user features


	    print('computing user f')
	    usr = pd.DataFrame()
	    usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
	    usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

	    users = pd.DataFrame()
	    users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
	    users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
	    users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)

	    users = users.join(usr)
	    del usr
	    users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
	    print('user f', users.shape)

	    ### userAndproduct features

        #dump off priors, userAndproduct

	    print('compute userAndproduct f - please wait...')
	    priors['user_product'] = priors.product_id + priors.user_id * 100000


            #dfReorderRatioAcrossUsers=pd.DataFrame()
	    dfReorderRatioAcrossUsers=pd.DataFrame()
	    dfReorderRatioAcrossUsers=fnGetProductReorderRatioAcrossUsers(priors)
	    dfReorderRatioAcrossUsers=dfReorderRatioAcrossUsers
            #dfReorderRatioAcrossUsers =dfReorderRatioAcrossUsers


	    #priors =priors.join(dfReorderRatioAcrossUsers,on='product_id',how='left',rsuffix='_re')
            
            
	    #priors.drop(['product_id_re'],inplace=True)


	    # This was to slow !!
	    #def last_order(order_group):
	    #    ix = order_group.order_number.idxmax
	    #    return order_group.shape[0], order_group.order_id[ix],  order_group.add_to_cart_order.mean()
	    #userAndproduct = pd.DataFrame()
	    #userAndproduct['tmp'] = df.groupby('user_product').apply(last_order)

    #way too slow, moved into faster function
            #if False:
	    if False:
	        d= dict()
	        for row in priors.itertuples():
		        z = row.user_product
		        if z not in d:
			        d[z] = (1,
					        (row.order_number, row.order_id),
					        row.add_to_cart_order)
		        else:
			        d[z] = (d[z][0] + 1,
					        max(d[z][1], (row.order_number, row.order_id)),
					        d[z][2] + row.add_to_cart_order)

	        print('to dataframe (less memory)')
	        userAndproduct = pd.DataFrame.from_dict(d, orient='index')
	        del d

	    #userAndproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
            
	    #userAndproduct.nb_orders = userAndproduct.nb_orders.astype(np.int16)

	    userAndproduct =fnGetUserAndProductFeatures(priors, orders)
	    #userAndproduct.last_order_id = userAndproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
	    #userAndproduct.sum_pos_in_cart = userAndproduct.sum_pos_in_cart.astype(np.int16)
	    print('user and product f', len(userAndproduct))

	    del priors

	    ### train / test orders ###
	    print('split orders : train, test')
	    test_orders = orders[orders.eval_set == 'test']
	    train_orders = orders[orders.eval_set == 'train']

	    train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

    ### build list of candidate products to reorder, with features ###

    def features(selected_orders, labels_given=False,dfReorderRatioAcrossUsers=None):
        print('build candidate list')
        order_list = []
        product_list = []
        labels = []
        i=0
        for row in selected_orders.itertuples():
            i+=1
            if i%10000 == 0: print('order row',i)
            order_id = row.order_id
            user_id = row.user_id
            user_products = users.all_products[user_id]
            product_list += user_products
            order_list += [order_id] * len(user_products)
            if labels_given:
                labels += [(order_id, product) in train.index for product in user_products]
        
        df = pd.DataFrame({'order_id':order_list, 'product_id':product_list}, dtype=np.int32)
        labels = np.array(labels, dtype=np.int8)
        del order_list
        del product_list
    
        print('user related features')
        df['user_id'] = df.order_id.map(orders.user_id)
        df['user_total_orders'] = df.user_id.map(users.nb_orders)
        df['user_total_items'] = df.user_id.map(users.total_items)
        df['total_distinct_items'] = df.user_id.map(users.total_distinct_items)
        df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
        df['user_average_basket'] =  df.user_id.map(users.average_basket)
    
        print('order related features')
        df['dow'] = df.order_id.map(orders.order_dow)
        df['order_hour_of_day'] = df.order_id.map(orders.order_hour_of_day)
        df['days_since_prior_order'] = df.order_id.map(orders.days_since_prior_order)
        df['days_since_ratio'] = df.days_since_prior_order / df.user_average_days_between_orders
    
        print('product related features')
        df['aisle_id'] = df.product_id.map(products.aisle_id)
        df['department_id'] = df.product_id.map(products.department_id)
        df['product_orders'] = df.product_id.map(products.orders).astype(np.int32)
        df['product_reorders'] = df.product_id.map(products.reorders)
        df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)

        print('user_X_product related features')
        #merge in userAndProduct dataframe which is created faster
        df =df.merge(userAndproduct,on =['user_id','product_id'],suffixes =('_del',''))

        #df.drop(['user_id_del'],inplace=True)
        #df.drop(['product_id_del'],inplace=True)
        
        df['z'] = df.user_id * 100000 + df.product_id
        df.drop(['user_id'], axis=1, inplace=True)
        #df['UP_orders'] = df.z.map(userAndproduct.nb_orders)
        #df['UP_orders_ratio'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
        #df['UP_last_order_id'] = df.z.map(userAndproduct.last_order_id)
        #df['UP_average_pos_in_cart'] = (df.z.map(userAndproduct.sum_pos_in_cart) / df.UP_orders).astype(np.float32)
        df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders).astype(np.float32)
        df['UP_orders_since_last'] = df.user_total_orders - df.UP_last_order_id.map(orders.order_number)
        df['UP_delta_hour_vs_last'] = abs(df.order_hour_of_day - df.UP_last_order_id.map(orders.order_hour_of_day)).map(lambda x: min(x, 24-x)).astype(np.int8)
        #df['UP_same_dow_as_last_order'] = df.UP_last_order_id.map(orders.order_dow) == \
        #                                              df.order_id.map(orders.order_dow)

        #add in product id reorderatio across users
        df =df.join(dfReorderRatioAcrossUsers,on='product_id',how='left',rsuffix='_re')
        df.drop(['product_id_re'],axis=1,inplace=True)
            
        del dfReorderRatioAcrossUsers
        df.drop(['UP_last_order_id', 'z'], axis=1, inplace=True)
        print(df.dtypes)
        print(df.memory_usage())
        return (df, labels)
    

    #start here on 6.25.2017

    #test_orders.to_csv("test_orders.csv")

    if blnLoadCSVFiles:
        test_orders=pd.read_csv("test_orders.csv")
    
        df_test, _ = features(test_orders,labels_given=False,dfReorderRatioAcrossUsers=dfReorderRatioAcrossUsers)
        df_test.to_csv("df_test.csv")

    blnLoadCSVFiles =False
    if blnLoadCSVFiles:
        df_train, labels = features(train_orders, labels_given=True,dfReorderRatioAcrossUsers=dfReorderRatioAcrossUsers)
        df_test, _ = features(test_orders)

        df_train.to_csv("df_train.csv")
        
    #print ('ONLY USING A SMALL SAMPLE FOR SPEED')

    #print ('***uncomment blocking of df_train later')
    if False:
        #df_train =pd.read_csv("df_train_sample.csv")
        df_train =pd.read_csv("df_train.csv")
        print ('num records in training set ', len(df_train))
        #df_train=df_train[:400000]

        #labels =np.loadtxt("labels_train_sample.csv",delimiter=',')
        labels =np.loadtxt("labels_train.csv",delimiter=',')
        #df_train =df_train[:600000]
        #labels =labels[:600000]
    #labels=labels[:400000]
    
    #df_train =fEncodeValues(df_train)
    #df_train.to_csv("df_train.csv")

    #np.to_csv("labels_train.csv")
    #np.savetxt("labels_train.csv",labels,delimiter=",")

    f_to_use = ['user_total_orders', 'user_total_items', 'total_distinct_items',
           'user_average_days_between_orders', 'user_average_basket',
           'order_hour_of_day', 'days_since_prior_order', 'days_since_ratio',
           'aisle_id',  'product_orders', 'product_reorders', #'department_id'
           'product_reorder_rate', # 'UP_orders', #'UP_orders_ratio' -duplicate definition
           'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_orders_since_last',
           'UP_delta_hour_vs_last','ReorderRatioAcrossUsers','NumTimesInLast2Orders'] # 'dow', 'UP_same_dow_as_last_o rder'

    if False:
        df_train =df_train[f_to_use]

    

    #numpy.savetxt("foo.csv", a, delimiter=",")

    print('Training model')
    #d_train = lgb.Dataset(df_train[f_to_use],
                          #label=labels,
                          #categorical_feature=['aisle_id', 'department_id'])  # , 'order_hour_of_day', 'dow'


    ROUNDS = 100

    if False:
        lenData =int(len(df_train)*.75)
    
        dtrain = xgb.DMatrix(df_train[:lenData], labels[:lenData])
        evalSet = xgb.DMatrix(df_train[lenData: ], labels[lenData:])
    ''' from R
    params <- list(
      "objective"           = "reg:logistic",
      "eval_metric"         = "logloss",
      "eta"                 = 0.1,
      "max_depth"           = 6,
      "min_child_weight"    = 10,
      "gamma"               = 0.70,
      "subsample"           = 0.76,
      "colsample_bytree"    = 0.95,
      "alpha"               = 2e-05,
      "lambda"              = 10
    )
    '''

    #('best using ',
    xgb_params ={'n_estimators': 571, 'alpha': 0.76959019539935369, 'colsample_bytree': 0.95, 'silent': 

    0, 'min_child_weight': 10, 'subsample': 0.65339831580156815, 'eta': 0.46886915041589994, 

    'objective': 'reg:logistic', 'max_depth': 9, 'gamma': 6, 'lambda': 17.162265206257807}
    #548, 0.20421433333333336)


    #('best using ', {'n_estimators': 428, 'alpha': 0.76742554365769566, 'colsample_bytree': 0.95, 'silent': 0, 'min_child_weight': 10, 'subsample': 0.89781002806824328, 'eta': 0.27992273213099095, 'objective': 'reg:logistic', 'max_depth': 8, 'gamma': 2, 'lambda': 3.6666896943322413}, 396, 0.44789766666666669, 0.22180265612744207)
##    xgb_params = {
##        'eta': .1, #learning rate
##	    'objective': "reg:logistic",
##        'max_depth': 6, #6
##	    'min_child_weight': 10,
##	    'gamma': .7,
##	    'alpha':2e-05,
##	    'lambda': 10,
##        'subsample': .76,
##        'colsample_bytree': 0.95,
##        #'eval_metric': 'logloss',
##        'silent': 0
##    }
    num_boost_rounds =238

     # Uncomment to tune XGB `num_boost_rounds`
    #cvResult = xgb.cv(xgb_params, dtrain, num_boost_round=200,feval=xg_f1,maximize=True,nfold=3,
    #                      early_stopping_rounds=30, verbose_eval=30, as_pandas=True)
    model=None
    #cvResult=fnOptSearch(dtrain,labels,model)
    #print (cvResult)
    #len(cv_output)
    if  False:
        model = xgb.train(dict(xgb_params), dtrain, num_boost_round= num_boost_rounds,
                          feval=xg_f1,evals=[(evalSet,'test')],maximize=True,
                          verbose_eval=50,early_stopping_rounds=60)

        pickle.dump(model, open("xgbModel.pickle.dat", "wb"))



    with open(r"xgbModel.pickle.dat", "rb") as input_file:
       model = pickle.load(input_file)

    #df=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)

    #for index, row in df.iterrows():
    #    print  (row['feature'], row['importance'])
    #,early_stopping_rounds=100,evals=[(evalSet,'test')])

    #model =None
    #cvResult=fnGridSearchCV(df_train[f_to_use],labels,model)
    #cvResult=fnOptSearch(dtrain,labels,model)
    # save model to file
    #model =pickle.load("xgbModel.pickle.dat")

    #THRESHOLD =0.20723499816  # try .19 guess, should be tuned with crossvalidation on a subset of train data
    THRESHOLD =0.1923499816 
    '''
    #PREDICTION on validation set
    '''
    if False:
        # Train/Valid split
        split =int(len(df_train)*.75)
    
        xx_train, yy_train, xx_valid, yy_valid = df_train[:split], labels[:split], df_train[split:], labels[split:]

        del df_train
        dEvaltest = xgb.DMatrix(xx_valid, yy_valid)
        xx_valid = xgb.DMatrix(xx_valid)

        y_predict = model.predict(xx_valid)

    

        #["ha" if i else "Ha" for i in range(3)]
        y_predict_Thresholded=[1 if item>THRESHOLD else 0 for item in y_predict ]

        print ('f1 score ',f1_score(yy_valid, y_predict_Thresholded))



    # lgb.plot_importance(bst, figsize=(9,20))
    if False:
        del dtrain

    ### build candidates list for test ###
    blnLoadCSVFiles =False
    #need test_orders to generate prediction!
    test_orders=pd.read_csv("test_orders.csv")
    print ('comment out blnLoadCSVFiles =True after running!')
    if blnLoadCSVFiles:
        
        df_test.to_csv("df_test.csv")
    else:

        df_test =pd.read_csv("df_test.csv")

    print('xgboost predict')
    xx_test = xgb.DMatrix(df_test[f_to_use])

    preds = model.predict(xx_test)

    df_test['pred'] = preds

    df_test.sort(columns=['pred','order_id'], inplace=True,ascending =[False,True])
    #cap number of products to user avg. basket
    #otherwise could have a user who typically orders 5 products, but predict 10 or more
    # in 1 order!!
    blnOrderIdExist =False
    d = dict()
    for row in df_test.itertuples():
        if row.pred > THRESHOLD:
            #if num items < = avg basket
            
            try:
                if row.order_id in d:
                    #order id already in dict
                    items =d[row.order_id].split()
                    if len(items)<=int(round(1.2*row.user_average_basket)):
                        d[row.order_id] += ' ' + str(row.product_id)
                else:
                    #order id NOT in dict
                    #d[row.order_id] += ' ' + str(row.product_id)
                    d[row.order_id] = str(row.product_id)
            except:
                d[row.order_id] = str(row.product_id)



    for order in test_orders.order_id:
        if order not in d:
            d[order] = 'None'

    sub = pd.DataFrame.from_dict(d, orient='index')

    sub.reset_index(inplace=True)
    sub.columns = ['order_id', 'products']
    sub.to_csv('sub.csv', index=False)

    #try to find timing of a reorder by product - when a product is reordered what
    #is the avg time btween last order?

    ''' calc. probability of reorder for each product  '''
    #Add a field to calculate the sum of times an item was reordered
    #products['rsum']=df_train.groupby('product_id')['reordered'].sum()
    #Add a field to calculate the total times the item could have been reordered
    #products['rtotal']=df_train.groupby('product_id')['reordered'].count()
    #Add a field to calculate the probability that the item was reordered
    #products['prob']=products['rsum']/products['rtotal']
    #products.head()


if __name__ == '__main__':
     fnMain()
