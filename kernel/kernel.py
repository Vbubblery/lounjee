
# coding: utf-8

# In[10]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from keras.layers import Dense, Input, Embedding, Reshape, Dropout, LeakyReLU, Flatten
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras import backend as Backend
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
Backend.tensorflow_backend._get_available_gpus()
import tensorflow as k
config = k.ConfigProto()
config.gpu_options.allow_growth = True

from pymongo import MongoClient
from bson.objectid import ObjectId


# In[2]:


def load_Data(url):
    client = MongoClient(url)
    db = client['lounjee']
    matches = db['matches']
    users = db['users']
    feedbacks = db['meetingFeedbackAnswers']
    dialogs = db['dialogs']
    
    result = []
    for item in users.find():
        result.append(str(item['_id']))
    df = pd.DataFrame(data=np.zeros((len(result),len(result))), index=result,dtype=np.int8, columns=result)
    
    def find_item(df,usera,userb,value=0):
        df[userb][usera] = value
        return df[userb][usera]
        
    for item in matches.find():
        try:
            if item['stateA']['type'] == 'accepted':
                find_item(df,str(item['userA']),str(item['userB']),1)
            elif item['stateA']['type'] == 'accepting':
                find_item(df,str(item['userA']),str(item['userB']),1)
            elif item['stateA']['type'] == 'postponed':
                find_item(df,str(item['userA']),str(item['userB']),1)
    #         elif item['stateA']['type'] == 'reporting':
    #             find_item(df,str(item['userA']),str(item['userB']),-5)
        except Exception:
            pass
    
    for item in users.find():
        try:
            if item['favorites']:
                for f in item['favorites']:
                    find_item(df,str(item['_id']),str(f['_id']),1)
        except Exception:
            pass
    
    df = df[(df.T != 0).any()]
    
    result=[]
    for i in df.index.values:
        for idx,val in enumerate(df.loc[i]):
            result.append({'usera':i,'userb':(df.loc[i].index)[idx],'rating':val})

    return pd.DataFrame(result)


# In[3]:


def data_Engineering(ratings,data, sampling=1):
    # Random under sampling
    if sampling is 1:
        count_class_0, count_class_1 = ratings.rating.value_counts()
        df_class_0 = ratings[ratings['rating'] == 0]
        df_class_1 = ratings[ratings['rating'] == 1]
        df_class_0_under = df_class_0.sample(count_class_1)
        ratings = pd.concat([df_class_0_under, df_class_1], axis=0)
    elif sampling is 2:
        count_class_0, count_class_1 = ratings.rating.value_counts()
        df_class_0 = ratings[ratings['rating'] == 0]
        df_class_1 = ratings[ratings['rating'] == 1]
        df_class_1_over = df_class_1.sample(count_class_0, replace=True)
        ratings = pd.concat([df_class_0, df_class_1_over], axis=0)
    
    ## reset index
    ratings.reset_index(inplace=True,drop=True)
    ratings.head()
    
    ## One hot encoding
    def dum(df,name):
        dummies = df[name].str.get_dummies(sep=',').add_prefix(name+'_')
        df.drop([name],axis=1,inplace=True)
        dummies
        df = df.join(dummies)
        return df
    arr = list(data)
    for val in arr:
        if val == 'uid':
            continue
        data = dum(data,val)
    
    result = pd.merge(ratings, data, left_on='usera',right_on='uid')
    result.drop(['usera','uid'], axis=1, inplace=True)
    result = pd.merge(result, data, left_on='userb',right_on='uid')
    result.drop(['uid','userb'], axis=1, inplace=True)
    
    del data
    del ratings
    import gc
    gc.collect()
    return result


# In[25]:


def train(data,length=743):
    def EmbeddingNetV3(n1_features,n2_features,n_latent_factors_user = 5,n_latent_factors_item = 8,n_users=743):
        model1_in = Input(shape=(n1_features,),name='userInput')
        model1_out = Embedding(input_dim = n_users+1, output_dim = n_latent_factors_user)(model1_in)
        model1_out = Flatten()(model1_out)
        model1_out = Dropout(0.2)(model1_out)

        model2_in = Input(shape=(n2_features,),name='itemInput')
        model2_out = Embedding(input_dim = n_users+1, output_dim = n_latent_factors_item)(model2_in)
        model2_out = Flatten()(model2_out)
        model2_out = Dropout(0.2)(model2_out)

        model = concatenate([model1_out, model2_out],axis=-1)
        model = LeakyReLU(alpha=0.15)(model)
        model = Dropout(0.2)(model)
        model = Dense(200)(model)
        model = LeakyReLU(alpha=0.15)(model)
        model = Dropout(0.2)(model)
        model = Dense(100)(model)
        model = LeakyReLU(alpha=0.15)(model)
        model = Dropout(0.2)(model)
        model = Dense(50)(model)
        model = LeakyReLU(alpha=0.15)(model)
        model = Dropout(0.2)(model)
        model = Dense(20)(model)
        model = LeakyReLU(alpha=0.15)(model)

        model = Dense(2, activation='softmax')(model)
        adam = Adam(lr=0.005)
        model = Model([model1_in, model2_in], model)
        model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
        return model
    
#     userb = pd.factorize(data.userb)[0]
    usera = data.iloc[:,1:525]
    userb = data.iloc[:,525:]
    n = length
    X = [usera,userb]
    y = data.rating
    
    encoder = LabelEncoder()
    encoder.fit(y)
    y = np_utils.to_categorical(encoder.transform(y), 2)    
    
    callback = [EarlyStopping(patience=2,monitor='val_acc')]
    
    model = EmbeddingNetV3(n1_features=usera.shape[1],n2_features=userb.shape[1],n_users=n)
    history = model.fit(X,y,batch_size=100,epochs=100,shuffle=True,validation_split=0.25,verbose=1,callbacks = None)
    def saveModel(model):
        model_json = model.to_json()
        open('lounjee_rs_architecture.json', 'w').write(model_json)
        model.save_weights('lounjee_rs_weights.h5', overwrite=True)
    saveModel(model)
    return (model,history,X,y)


# In[18]:


ratings = load_Data("mongodb://178.128.161.146:27017/")
data = pd.read_csv('./data1.csv')


# In[21]:


result = data_Engineering(ratings,data,sampling=1)


# In[26]:


# result.to_csv("merge_data.csv",sep=';',index=False)
# train(result)
(model,history,X,y) = train(result)


# In[27]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[8]:


from keras.models import model_from_json
def loadModel():
    model_architecture = 'lounjee_rs_architecture.json'
    model_weights = 'lounjee_rs_weights.h5'
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)
    return model
# model = loadModel()


# In[9]:


result = pd.read_csv("merge_data.csv",sep=';')
usera = result.iloc[:,1:525]
userb = result.iloc[:,525:]
n = 743
X = [usera,userb]
y = result.rating
    
encoder = LabelEncoder()
encoder.fit(y)
y = np_utils.to_categorical(encoder.transform(y), 2)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.evaluate(X,y,verbose=1)


# In[ ]:


# pred_y = model.predict(X)
# pred_y

