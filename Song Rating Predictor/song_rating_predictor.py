

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

songlabel = pd.read_csv('song_labels.csv')
songs = pd.read_csv('songs.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
savelater = pd.read_csv('save_for_later.csv')

#making platform_id as key in song_labelid
idx = songlabel.groupby(['platform_id'] )['count'].transform(max) == songlabel['count']
newsonglabel = songlabel[idx]

idx1 = newsonglabel.groupby(['platform_id'] )['label_id'].transform(max) == newsonglabel['label_id']
new1songlabel = newsonglabel[idx1]
#new1songlabel.head()

# now need to merge new1songlabel and songs 
newsongs = pd.merge(songs , new1songlabel , how = 'left' , on = 'platform_id')

nwsongs1 = newsongs.drop(['platform_id'] , axis = 1)


copysong = nwsongs1
copysong['released_year'].fillna((copysong['released_year'].mean()) , inplace = True)
copysong['language'].fillna('eng',inplace = True)
copysong['number_of_comments'].fillna((copysong['number_of_comments'].mean()),inplace = True)
copysong['count'].fillna( (copysong['count'].mean()) , inplace = True )
copysong['label_id'].fillna( '30574' , inplace = True )

sflmod = pd.merge(savelater,copysong,how = 'left' , on = 'song_id')
#sflmod.head()

sflmod1 = sflmod.drop(['song_id','released_year','language','number_of_comments','count'], axis = 1)
sflmod1['count'] = 1
sflmod2 = sflmod.drop(['song_id','released_year','number_of_comments' ,'label_id','count'],axis = 1)
sflmod2['count'] = 1
sflmod3 = sflmod.drop(['song_id','number_of_comments' ,'label_id','count','language'] ,axis =1 )
sflmod3['count'] = 1
sflmod3 = sflmod3.groupby(['customer_id','released_year']).sum()
sflmod3 = sflmod3.reset_index()
sflmod1 = sflmod1.groupby(['customer_id','label_id']).sum()
sflmod1 = sflmod1.reset_index()
sflmod2 = sflmod2.groupby(['customer_id','language']).sum()
sflmod2 = sflmod2.reset_index()

idx = sflmod3.groupby(['customer_id'] )['count'].transform(max) == sflmod3['count']
newsflmod3 = sflmod3[idx]

idx1 = newsflmod3.groupby(['customer_id'] )['released_year'].transform(max) == newsflmod3['released_year']
new1sflmod3 = newsflmod3[idx1]
#new1sflmod3.head()

idx = sflmod1.groupby(['customer_id'] )['count'].transform(max) == sflmod1['count']
newsflmod1 = sflmod1[idx]

idx1 = newsflmod1.groupby(['customer_id'] )['label_id'].transform(max) == newsflmod1['label_id']
new1sflmod1 = newsflmod1[idx1]
#new1sflmod1.head()

idx = sflmod2.groupby(['customer_id'] )['count'].transform(max) == sflmod2['count']
newsflmod2 = sflmod2[idx]

idx1 = newsflmod2.groupby(['customer_id'] )['language'].transform(max) == newsflmod2['language']
new1sflmod2 = newsflmod2[idx1]
#new1sflmod2.head()

sfllabel = new1sflmod1.drop(['count'],axis = 1)
sfllang = new1sflmod2.drop(['count'] ,axis = 1)
sflyear = new1sflmod3.drop(['count'] ,axis = 1)
sfllabel = sfllabel.rename(columns={"label_id": 'mlabel'})
sflyear = sflyear.rename(columns={"released_year": "myear"})
sfllang = sfllang.rename(columns={"language": "mlang"})

# merging train and nwsongs1

anstrain = pd.merge(train , nwsongs1 , how = 'left' , on = 'song_id')

# to fill nan's
anstrain['released_year'].fillna( (nwsongs1['released_year'].mean()) , inplace = True )
anstrain['language'].fillna( 'eng' , inplace = True )
anstrain['number_of_comments'].fillna( (nwsongs1['number_of_comments'].mean()) , inplace = True )
anstrain['count'].fillna( (nwsongs1['count'].mean()) , inplace = True )
anstrain['label_id'].fillna( '30574' , inplace = True )

anstrain = pd.merge(anstrain , sflyear , how = 'left' , on = 'customer_id')
anstrain = pd.merge(anstrain , sfllabel , how = 'left' , on = 'customer_id')
anstrain = pd.merge(anstrain , sfllang , how = 'left' , on = 'customer_id')

anstrain['myear'].fillna('2012' , inplace =True)
anstrain['mlabel'].fillna('30574' , inplace = True)
anstrain['mlang'].fillna('eng' , inplace = True)





# anstrain = anstrain.drop('customer_id',axis = 1)
# anstrain = anstrain.drop('score',axis = 1)
#anstrain.head()

user_count = 0
song_count = 0
user_mapping = {}
song_mapping = {}
year_mapping = {}
language_mapping = {}
comments_mapping = {}
count_mapping = {}
label_id_mapping = {}
user_encoding = []
language_count = 0
language_encoding = []

song_encoding = []
ratinglist = []

for indx,row in anstrain.iterrows():
    # print(indx)
    if row['customer_id'] not in user_mapping.keys():
      user_mapping[row['customer_id']] = user_count
      user_count = user_count + 1

    if row['song_id'] not in song_mapping.keys():
      song_mapping[row['song_id']] = song_count
      song_count = song_count + 1
    
    if row['language'] not in language_mapping.keys():
        language_mapping[row['language']] = language_count
        language_count = language_count + 1
        
    if row['song_id'] not in count_mapping.keys():
      count_mapping[row['song_id']] = row['count']
    
    if row['song_id'] not in label_id_mapping.keys():
      label_id_mapping[row['song_id']] = row['label_id']
    
    if row['song_id'] not in comments_mapping.keys():
      comments_mapping[row['song_id']] = row['number_of_comments']
    
    if row['song_id'] not in year_mapping.keys():
      year_mapping[row['song_id']] = row['released_year']

    user_encoding.append( user_mapping[ row['customer_id'] ] )
    song_encoding.append( song_mapping[ row['song_id'] ] )
    language_encoding.append(language_mapping[row['language']])
    ratinglist.append(row['score'])

mlang_encoding = []
for indx,row in anstrain.iterrows():
    mlang_encoding.append(language_mapping[row['mlang']])

user_matrix = np.zeros([user_count,song_count])
user_matrix2 = np.zeros([user_count,song_count])

for indx,row in train.iterrows():
  # print(indx)
  i = user_mapping[row['customer_id']]
  j = song_mapping[row['song_id']]
  k = row['score']
  user_matrix2[i][j] = 1
  user_matrix[i][j] = k

remp = np.sum(user_matrix2,axis = 1)
demp = np.sum(user_matrix,axis = 1)
umeans = np.divide(demp,remp)

transmat = user_matrix.T
transmat2 = user_matrix2.T
tremp = np.sum(transmat,axis =1)
tdemp = np.sum(transmat2,axis = 1)
smeans = np.divide(tremp,tdemp)

cmean = []
somean = []
 
for indx,row in train.iterrows():
    i = user_mapping[row['customer_id']]
    j = song_mapping[row['song_id']]
    cmean.append(umeans[i])
    somean.append(smeans[j])

encoded_train = anstrain.drop(['customer_id' , 'song_id','language','mlang'] , axis = 1)
# encoded_train = anstrain.drop()
#encoded_train.head()

encoded_train['userid'] = user_encoding
encoded_train['lang_id'] = language_encoding
encoded_train['mlang'] = mlang_encoding
#encoded_train.head()

encoded_train['itemid'] = song_encoding
#encoded_train.head()

encoded_train['useravg'] = cmean
encoded_train['songavg'] = somean
#encoded_train.head()

myencode = encoded_train
myencode.label_id = myencode.label_id.astype('int64') 
myencode.mlabel = myencode.mlabel.astype('int64') 
myencode.myear = myencode.myear.astype('int64')

y_train = pd.DataFrame()
y_train = myencode['score']

Xtrain = myencode.drop(['score'],axis=1)

#Xtrain.head()

# Xtrain2 = Xtrain1.drop(['count'],axis =1)
#Xtrain.head()

model = cb.CatBoostRegressor( cat_features = [2,5,6,7,8,9])
model.fit(Xtrain, y_train)

pred = model.predict(Xtrain)

#y_train.head()

#mse = np.square(np.subtract(y_train.tolist(),pred.tolist())).mean()
#mse

#Xtrain.head()

songe = {}
# son = 0
for indx , row in anstrain.iterrows():
    if row['song_id'] not in songe.keys():
        songe[row['song_id']] = language_mapping[row['language']]
#         son  = son + 1

test1 = test
test = pd.merge(test,sflyear, how= 'left' , on = 'customer_id' )
test = pd.merge(test,sfllabel, how= 'left' , on = 'customer_id' )
test = pd.merge(test,sfllang, how= 'left' , on = 'customer_id' )
test['myear'].fillna('2012' , inplace =True)
test['mlabel'].fillna('30574' , inplace = True)
test['mlang'].fillna('eng' , inplace = True)

test_user_encoding = []
test_song_encoding = []
test_user_avg = []
test_song_avg = []
test_year = []
test_comments = []
test_count = []
test_labelid = []
test_lang = []
test_mlang = []
test_mlabel = []
test_myear = []
for indx,row in test.iterrows():
    test_user_encoding.append( user_mapping[row['customer_id']] )
    test_song_encoding.append( song_mapping[row['song_id']] )
    test_user_avg.append( umeans[user_mapping[row['customer_id']]])
    test_song_avg.append( smeans[ song_mapping[row['song_id']] ] )
    test_year.append( year_mapping[ row['song_id'] ]  )
    test_comments.append( comments_mapping[ row['song_id'] ]  )
    test_count.append( count_mapping[ row['song_id'] ]  )
    test_labelid.append( label_id_mapping[ row['song_id'] ]  )
    test_lang.append( songe[row['song_id']] )
    test_mlang.append(language_mapping[row['mlang']])
    test_mlabel.append(row['mlabel'])
    test_myear.append(row['myear'])

final_test = pd.DataFrame()
final_test['released_year'] = test_year
final_test['number_of_comments'] = test_comments
final_test['label_id'] = test_labelid
final_test['count'] = test_count
final_test['myear'] = test_myear
final_test['mlabel'] = test_mlabel
final_test['userid'] = test_user_encoding
final_test['lang_id'] = test_lang
final_test['mlang'] = test_mlang
final_test['itemid'] = test_song_encoding
final_test['useravg'] = test_user_avg
final_test['songavg'] = test_song_avg
final_test.head()

final_test.label_id = final_test.label_id.astype('int64')
final_test.mlang = final_test.mlang.astype('int64')
final_test.mlabel = final_test.mlabel.astype('int64')
final_test.myear = final_test.myear.astype('int64')
# final_test2 = final_test.drop(['count'],axis = 1)

pred = model.predict(final_test)

lst2 = pred.tolist()
lst1 = []
for i in range(len(lst2)):
    lst1.append(i)
dfsubmit = pd.DataFrame(np.array(lst1),columns=['test_row_id'])
dfsubmit['score'] = lst2
dfsubmit.to_csv('cs18b034_cs18b051.csv', index = False, header = True)