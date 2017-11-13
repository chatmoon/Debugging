# In[ ]: Step 0: Load The Data
# Settings and prerequisites
# Import libraries (source : https://docs.python.org/3.4/library/index.html)
import sys
import pickle
import random
from datetime import datetime as dt

import numpy as np
from numpy import newaxis

import os, sys
import PIL
from PIL import Image

import scipy as sp
from scipy import ndimage
from scipy import misc

import time

import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib import gridspec

import prettyplotlib as ppl
import brewer2mpl

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib.layers import flatten
import urllib
import urllib.request

from shutil import copyfile

from sklearn.utils import shuffle

# Directories location 
pathData = 'C:/Users/mo/home/_eSDC_/_02_WIP/_DATA_/' # input('Enter the data path directory of the project data set')
pathImg  = pathData+'images/'
pathTb   = pathData+'tboard/'
pathPro  = pathData+'preprocessed/'
pathJit  = pathData+'jittered/'
pathAdon = pathJit +'_ADDON_/'
pathFull = pathJit +'_FULL_/'
pathV1   = pathData+'v1/'
pathLog  = 'C:/Users/mo/home/_eSDC_/_02_WIP/_170515-2125_Backup_/_Coding_/logs/nn_logs/'
pathCkp  = pathLog+'170704x0054_F3000RGB_R85e-5_D67/'

# Traces activation 
tRace1, tRace2, tRace3 = False, False, False
msg0 = '....WIP'+' '+dt.now().strftime("%y%m%dx%H%M")
msg1 = '.LOADED'+' '+dt.now().strftime("%y%m%dx%H%M")
msg2 = '...DONE'+' '+dt.now().strftime("%y%m%dx%H%M")
msg3 = '..ERROR'+' '+dt.now().strftime("%y%m%dx%H%M")
msg4 = '..DEBUG'+' '+dt.now().strftime("%y%m%dx%H%M")

# Helper functions: create directory tree
def foldCheck(path):
    '''Create a folder if not present'''
    if not os.path.exists(path):
        os.makedirs(path)

def foldCrea(path, dct):
    foldCheck(path)
    for i in dct['root']:
        foldCheck(path+i)
    for i in dct[ dct['root'][1] ]:
        foldCheck(path+dct['root'][1]+'/'+i)

# create directory tree
dic0 = {}
dic0 = { 'root'     : ['images', 'jittered', 'newData', 'preprocessed', 'tboard'],         'jittered' : ['_ADDON_', '_FULL_'] }
foldCrea(pathData,dic0)
        
        
get_ipython().magic('matplotlib inline')

if tRace1: print(msg2,' | [{}] {}'.format('step0','settings and prerequisites'))


# In[ ]: Load pickled data
# TODO: Fill this in based on where you saved the training and testing data
#path = pathData
#training_file   = path+'train.p'
#validation_file = path+'valid.p'
#testing_file    = path+'test.p'
#
#try:
#    with open(training_file, mode='rb') as f:
#        train = pickle.load(f)
#    with open(validation_file, mode='rb') as f:
#        valid = pickle.load(f)
#    with open(testing_file, mode='rb') as f:
#        test = pickle.load(f)
#    
#    X_train, y_train = train['features'], train['labels']
#    X_valid, y_valid = valid['features'], valid['labels']
#    X_test, y_test = test['features'], test['labels']
#except:
#    raise IOError('the project data set should have been imported into the pathData directory before running this cell')
#
#if tRace1: print(msg1,' | [{:5}] {}'.format('step0','Load train, valid, test data'))
#
#s_train, c_train = train['sizes'], train['coords']
#s_valid, c_valid = valid['sizes'], valid['coords']
#s_test , c_test  = test['sizes'] , test['coords']
#
#if tRace1: print(msg1,' | [{:5}] {}'.format('step0','Load their related sizes and coordinates'))


# In[ ]: Step 1: Dataset Summary & Exploration
# Helper functions: Map class id, occurence, Traffic-sign names
def ocrLabel(dataLabel, tRace=False, csvFile='signnames.csv'):
    n_classes = len(np.unique(dataLabel))

    # Map class_id with Traffic-sign names. Output : dct[classId] = Traffic-sign names
    with open(csvFile, newline='', encoding="utf8") as csvFile1:
        read0 = csv.reader(csvFile1)
        dct0, dct1 = {}, {}
        for i in read0:
            try:
                dct0[int(i[0])] = i[1]
            except:
                pass

    # Occurence by class id. Output : dct[classId] = occurence
    ocr, classId = np.histogram(dataLabel, np.arange(n_classes+1))
    classId = classId[:-1].copy()
    for i,j in zip(classId,ocr):
            dct1[i] = j

    # Occurence by Traffic-sign names. Output : lt[classId] = [occurence, Traffic-sign names] 
    lt = []
    for i in classId:
            lt.append([dct1[i], dct0[i]])

    return dct0, dct1, lt

if tRace1: print(msg1,' | [{:5}] {:12} : {}'.format('step1','def ocrLabel','mapping of id/Class, occu., T-Sign names'))


# Helper functions: dict[class id] = 1D array of all related indexes
def indexClass(dataLabel, tRace=False):
    '''Output: a dictionary that has 43 keys (= class id)
               and its values are an 1D array containing all the indexes of a each class
    '''
    n_classes = len(np.unique(dataLabel))
    dct = {}
    tl  = ()

    for i in range(n_classes):
        tl     = np.where(dataLabel == i) # tuple = array of Index(class)
        dct[i] = tl[0]                  # dictionary[key=idClass] = array of Index(idClass)

    return dct
    
if tRace1: print(msg1,' | [{:5}] {:12} : {}'.format('step1','def indexClass','dict[classId] = 1D array''(''indexes'')'' '))

# Helper functions: Show xSize*ySize images
def showTrace(dataImg,title='',xSize=1, ySize=8):
    fig0, ax0 = plt.subplots(xSize, ySize, figsize=(15,6))
    fig0.subplots_adjust(hspace=0.2, wspace=0.1)
    ax0 = ax0.ravel()

    for i in range(xSize*ySize): 
        image = dataImg[i].squeeze()
        #print('[INPUT]image.shape: {}'.format(image.shape))
        
        ch = len(image.shape)
        #print('[INPUT]ch = len dataImg.shape: {}'.format(ch))
        
        if image.shape[-1] == 3:        
            cMap='rgb'
            ax0[i].imshow(image)
        elif image.shape[-1] == 32 or image.shape[-1] == 1:
            cMap='gray'
            ax0[i].imshow(image, cmap = cMap)
        else:
            raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

        #ax0[i].imshow(image, cmap = cMap)
        ax0[i].set_title(title, fontsize=8)
        ax0[i].axis('off')
        
if tRace1:print(msg1,' | [{:5}] {}: {}'.format('step1','showTrace','show xSize x ySize images'))


# In[ ]: Step 2: Design and Test a Model Architecture
# Helper functions ------------------------------------------------------------
class hFct(object):
    '''Helper functions'''
    def __init__(self,path,name,pyObj):
        self.path  = path
        self.name  = name
        self.pyObj = pyObj

    # Save python objects -----------------------------------------------------
    def serialize(self):
        """Pickle a Python object"""      
        with open(self.path+self.name, "wb") as pfile:
            pickle.dump(self.pyObj, pfile)

    # Load python objects -----------------------------------------------------
    def deserialize(self):
        """Extracts a pickled Python object and returns it"""
        with open(self.path+self.name, "rb") as pfile:
            dataSet = pickle.load(pfile)
        return dataSet

    # Load pickled data -------------------------------------------------------
    def loadValid(self):       
        dataSet   = self.deserialize()
        dataImg   = dataSet['features']
        dataLabel = dataSet['labels']
        try:
            dataSize  = dataSet['sizes']
            dataCoord = dataSet['coords']     
        except:
            dataSize  = {}
            dataCoord = {}
        finally:
            return dataImg, dataLabel, dataSize, dataCoord           
            
if tRace1: print(msg1,' | [{}] {:14} : {}, {}, {}'.format('step0','helper fct','serialize','deserialize','loadValid'))


# In[ ]: Download data
    
# Clean RAM
X_train, X_valid, X_test = None, None, None
y_train, y_valid, y_test = None, None, None


# Dictionary of data files
# note: for the next sprint > automatize by implementing the dict. creation into the jitItall function
dict1 = {}
dict1['F0INIT'] = ['train.p','valid.p','test.p']
dict1['F0RGB'] = ['train_0Rgb.p','valid_0Rgb.p','test_0Rgb.p']
dict1['F0CLAHE'] = ['train_4Clahe.p','valid_4Clahe.p','test_4Clahe.p']
dict1['F3000RGB'] = ['JIT_full_3000_train_0Rgb.p','valid_0Rgb.p','test_0Rgb.p']
dict1['F3000GRAY'] = ['JIT_full_3000_train_1Gray.p','valid_1Gray.p','test_1Gray.p']
dict1['F3000CLAHE'] = ['JIT_full_3000_train_4Clahe.p','valid_4Clahe.p','test_4Clahe.p']


# Load and shuffle data
#fxTrain, fxValid, fxTest = dict1['F0INIT'][0], dict1['F0INIT'][1], dict1['F0INIT'][2] 
#pTrain, pValid, pTest    = pathData, pathData, pathData
#fxTrain, fxValid, fxTest = dict1['F0RGB'][0], dict1['F0RGB'][1], dict1['F0RGB'][2] 
#pTrain, pValid, pTest    = pathPro, pathPro, pathPro
#fxTrain, fxValid, fxTest = dict1['F0GRAY'][0], dict1['F0GRAY'][1], dict1['F0GRAY'][2] 
#pTrain, pValid, pTest    = pathPro, pathPro, pathPro
fxTrain, fxValid, fxTest = dict1['F3000GRAY'][0], dict1['F3000GRAY'][1], dict1['F3000GRAY'][2]
pTrain, pValid, pTest    = pathFull, pathPro, pathPro
#fxTrain, fxValid, fxTest = dict1['F3000RGB'][0], dict1['F3000RGB'][1], dict1['F3000RGB'][2]
#pTrain, pValid, pTest    = pathFull, pathPro, pathPro


X_train, y_train, s_train, c_train = hFct(pTrain, fxTrain, '').loadValid()
X_valid, y_valid, s_valid, c_valid = hFct(pValid, fxValid, '').loadValid()
X_test,  y_test,  s_test,  c_test  = hFct(pTest , fxTest , '').loadValid()

try:
    X_train, y_train, s_train, c_train = shuffle(X_train, y_train, s_train, c_train)
    X_valid, y_valid, s_valid, c_valid = shuffle(X_valid, y_valid, s_valid, c_valid)
    X_test,  y_test,  s_test,  c_test  = shuffle(X_test,  y_test,  s_test,  c_test)
except:
    X_train, y_train = shuffle(X_train, y_train)
    X_valid, y_valid = shuffle(X_valid, y_valid)
    X_test,  y_test  = shuffle(X_test,  y_test)
    
    
if tRace1:print(msg1,' | [{}][{}] {:14} : {} {:8} = {:28} {}'.format('step2.5',0,'preliminaries','download', 'X_train',fxTrain,X_train.shape))
if tRace1:print(msg1,' | [{}][{}] {:14} : {} {:8} = {:28} {}'.format('step2.5',0,'preliminaries','download', 'X_valid',fxValid,X_valid.shape))
if tRace1:print(msg1,' | [{}][{}] {:14} : {} {:8} = {:28} {}'.format('step2.5',0,'preliminaries','download', 'X_test',fxTest,X_test.shape))


# In[ ]: 2.2.0. Helper functions

# ch parameter ----------------------------------------------------------------
if X_train.shape[-1] == 3:
    ch   = 3
    cMap = 'rgb'
elif X_train.shape[-1] == 1 or X_train.shape[-1] == 32:
    ch   = 1
    cMap = 'gray'
if tRace1:print()
if tRace1:print(msg1,' | [{}][{}] {:14} : {:17} = {}'.format('step2.5',0,'preliminaries','channel',ch))


# Tensorboard -----------------------------------------------------------------
class tBoard(object):
    def __init__(self):
        pass
    
    def dataSprite(dataImg, dataLabel, tRace=False):
        '''Calculate the validation dataset lenght'''
        if tRace: print('dataImg.shape: {}, dataLabel.shape: {}'.format(dataImg.shape,dataLabel.shape))
        import math
        num0 = math.ceil(len(dataImg)**0.5)
        num0 *= num0

        # TB.E-V: outImg, outLabel
        outImg, outLabel = np.empty((num0,dataImg.shape[1],dataImg.shape[2],dataImg.shape[3])), np.empty((num0))
        outImg[:dataImg.shape[0]], outLabel[:dataLabel.shape[0]] = dataImg[:].copy(), dataLabel[:].copy()
        outImg[dataImg.shape[0]:], outLabel[dataLabel.shape[0]:] = dataImg[-1], dataLabel[-1]
        if tRace: print('Image set.shape: {}, Label set.shape: {}'.format(outImg.shape,outLabel.shape))
       
        return outImg, outLabel

        
    def iNitb(X_valEV, embedding_size, embedding_input, tRace=tRace1, msg=msg2):
        # Combine all of the summary nodes into a single op -------------------
        merged = tf.summary.merge_all()
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',4,'TensorBoard','tf.summary.merge_all'))

        # Setup a 2D tensor variable that holds embedding ---------------------
        embedding  = tf.Variable(tf.zeros([len(X_valEV), embedding_size]), name="test_embedding") # 4489, embedding_size
        assignment = embedding.assign(embedding_input)
        if tRace:print(msg,' | [{}][{}] {:14}: {}, {}'.format('step2.5',4,'TensorBoard','embedding','assignment'))
        
        return merged, embedding, assignment

    def logWriter(sess, tRace=tRace1, msg=msg2):
        # Create a log writer. run 'tensorboard --logdir=./logs/nn_logs' ------
        #from datetime import datetime as dt
        now = dt.now()
        str0= now.strftime("%y%m%dx%H%M")
        str1= "./logs/nn_logs/" + str0 + "/"
        writer = tf.summary.FileWriter(str1, sess.graph) # for 0.8
        writer.add_graph(sess.graph)
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',5,'TensorBoard','create a log writer'))

        return str0, str1, writer

    # Embedding Visualization: configuration ---------------------------------- 
    def eVisu(sprImg,sprTsv,sprPath,LOGDIR,sIze,embedding,writer):
        '''TensorBoard: Embedding Visualization'''
        # Note: use the same LOG_DIR where you stored your checkpoint.
        inFileImg, inFileTvs = sprPath+sprImg, sprPath+sprTsv
        outFileImg, outFileTvs = LOGDIR+sprImg, LOGDIR+sprTsv
        
        from shutil import copyfile
        copyfile(inFileImg,outFileImg)
        copyfile(inFileTvs,outFileTvs)

        # 4. Format:
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        
        # 5. Add as much embedding as is necessary (Here we add only one)
        embedding_config = config.embeddings.add()
        embedding_config.tensor_name = embedding.name         #embedding_var.name
        embedding_config.sprite.image_path = outFileImg
        
        # 6. Link this tensor to its labels (e.g. metadata file)
        embedding_config.metadata_path = outFileTvs

        # 7. Specify the width and height of a single thumbnail.
        embedding_config.sprite.single_image_dim.extend([sIze, sIze])
        
        # 8. Saves a configuration file that TensorBoard will read during startup
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
           
        return config, embedding_config


# In[ ]:

# Classifier ------------------------------------------------------------------
class tSign(object):
    def __init__(self):
        pass
    
    # In[step2.5]: Preliminaries - initialization -----------------------------
    def iNit(cMap, n_classes, tRace=tRace1, msg=msg2):
        # Remove previous Tensors and Operations ------------------------------
        tf.reset_default_graph()
        sess = tf.InteractiveSession() # sess = tf.Session()
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',0,'initiation','remove previous tensors and operations'))

        # Setup placeholders: features and labels -----------------------------       
        if cMap =='rgb':
            ch = 3
        elif cMap =='gray':
            ch = 1
        else:
            raise ValueError('Current cMap:',cMap,'. cMap should be ''rgb'' or ''gray''')
            
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',0,'channel',ch))
        
        x = tf.placeholder(tf.float32, (None, 32, 32, ch), name='input')
        y = tf.placeholder(tf.uint8, (None), name='label') # y = tf.placeholder(tf.int32, (None, len(y_train)))
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',0,'placeholders','features and labels'))

        # One-Hot -------------------------------------------------------------
        one_hot_y = tf.one_hot(y, n_classes)
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',0,'initiation','one_hot_y'))

        # Add dropout to input and hidden layers ------------------------------
        keep_prob = tf.placeholder(tf.float32) # probability to keep units
        if tRace:print(msg,' | [{}][{}] {:14}: {}, {}'.format('step2.5',0,'placeholders','keep_prob','add dropout to input and hidden layers'))
       
        # Add image summary ---------------------------------------------------
        tf.summary.image('input', x, 8)
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',0,'TensorBoard','add image summary'))
               
        return sess, x, y, ch, one_hot_y, keep_prob

        if tRace:print(msg1,' | [{}][{}] {:14}: {}'.format('step2.5',0,'placeholders','initialization'))
            
            
    # In[step2.5]: helper functions - conv_layer, fc_layer
    # conv_layer: Build a convolutional layer ---------------------------------
    def conv_layer(input,filter_size,size_in,size_out,nAme="conv", mu=0, sigma=0.1,pAdding='VALID',maxPool=True, aCtivation='relu', leak=0.2, tRace=tRace1, msg=msg2):
        # Traces
        if tRace:a = 0

        with tf.name_scope(nAme):
            # Layer: Convolutional. Input = 32x32xsize_in. Output = 28x28xsize_out.
            shape0 = [filter_size, filter_size, size_in, size_out]
            w = tf.Variable(tf.truncated_normal(shape0, mean = mu, stddev = sigma), name=nAme+"W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name=nAme+"B")
            conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding=pAdding)
            # Traces
            if tRace:a += 1
            if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','conv.shape',conv.get_shape()))
 
            # Activation
            if aCtivation =='relu':
                act  = tf.nn.relu(tf.add(conv, b)) #act = tf.nn.relu(conv + b)
                str9 = 'RELU'
            else:
                f1  = 0.5 * (1 + leak)
                f2  = 0.5 * (1 - leak)
                act = f1 * tf.add(conv, b) + f2 * abs(tf.add(conv, b))  
                str9 = 'LEAKY RELU'
            # Traces
            if tRace:a += 1
            if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','act.shape '+str9,act.get_shape()))
            
            # Add histogram summaries for weights and biases
            tf.summary.histogram(nAme+"_weights", w)
            tf.summary.histogram(nAme+"_biases", b)
            tf.summary.histogram(nAme+"_activations", act)
            
            if maxPool:
                # Pooling. Input = 28x28xsize_out. Output = 14x14xsize_out.
                output = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=pAdding)
                # Traces
                if tRace:a += 1
                if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','max_pool.shape',output.get_shape()))
            else:
                output = act
                # Traces
                if tRace:a += 1
                if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','act.shape',output.get_shape()))

        if tRace:print(msg1,' | [{}][{}] {:14}: {}, {}'.format('step2.5',1,'helper fct','conv_layer', 'build a convolutional layer'))
            
        return output    


    # fc_layer: Build a full connected layer ----------------------------------
    def fc_layer(input, size_in, size_out, nAme="fc", act = True, drop= True, keep_prob = tf.placeholder(tf.float32), aCtivation='relu', leak=0.2, tRace=tRace1, msg=msg2):
        # Traces
        if tRace:a = 0
        if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','input.shape',input.get_shape()))
        
        with tf.name_scope(nAme):
            # Layer: Convolutional. Input = size_in. Output = size_out.
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name=nAme+"W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name=nAme+"B")
            x = tf.add(tf.matmul(input, w), b)
            # Add histogram summaries for weights and biases
            tf.summary.histogram(nAme+"_weights", w)
            tf.summary.histogram(nAme+"_biases", b)
            # Traces
            if tRace:a += 1
            if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','x.shape',x.get_shape()))
            
            
            if act: # Activation and histogram summaries:
                if aCtivation =='relu':
                    x = tf.nn.relu(x)
                    str9 = 'RELU'
                else:
                    f1  = 0.5 * (1 + leak)
                    f2  = 0.5 * (1 - leak)
                    x   = f1 * x + f2 * abs(x)  
                    str9 = 'LEAKY RELU'
                tf.summary.histogram(nAme+"_activations", x)
                # Traces
                if tRace:a += 1
                if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','act.shape '+str9,x.get_shape()))
            if drop: # Dropout
                x = tf.nn.dropout(x, keep_prob)
                # Traces
                if tRace:a += 1
                if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','dropout.shape',x.get_shape()))

            return x

        if tRace:print(msg1,' | [{}][{}][{}] {:14}: {}, {}'.format('step2.5',1,'helper fct','fc_layer','build a full connected layer'))


    # Define cost function ----------------------------------------------------
    def loss(logits, one_hot_y, rate, mod0, tRace=tRace1, msg=msg2):
        with tf.name_scope("cost"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y, name="xent")  
            #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, one_hot_y) #<-- error msg
            lossOpe = tf.reduce_mean(cross_entropy, name="loss")
            optimizer = tf.train.AdamOptimizer(learning_rate = rate, name="optAdam")
            trainingOpe = optimizer.minimize(lossOpe, name="optMin")
            # Add scalar summary for loss (cost) tensor
            tf.summary.scalar(mod0+'_loss', lossOpe)        
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',3,'tf.name_scope','cost'))
        
        return lossOpe, trainingOpe


    # Define accuracy fct -----------------------------------------------------
    def accuracy(logits, one_hot_y, mod0, tRace=tRace1, msg=msg2):
        with tf.name_scope("accuracy"):
            correctPrd = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
            accuOpe = tf.reduce_mean(tf.cast(correctPrd, tf.float32))
            # Add scalar summary for accuracy tensor
            tf.summary.scalar(mod0+'_accuracy', accuOpe)    
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',3,'tf.name_scope','accuracy'))
        
        return accuOpe


# In[ ]:

class modArc(tSign):
    def __init__(self):
        pass

    # Define architecture model1 ----------------------------------------------
    def model1(x, ch, mu, sigma, keep_prob, tRace=tRace1, msg=msg2):  # Lenet5
        if tRace:a = 0
        if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','ch',ch))
        if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','x.shape',x.get_shape()))
        
        # Layer 1: Conv{In:32x32xch;Out:28x28x6} > Activ. > mxPooling{In:28x28x6;Out:14x14x6}
        x = tSign.conv_layer(x, 5, ch, 6, 'layer1', mu, sigma, 'VALID')
        if tRace: a += 1
        if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','x.shape',x.get_shape()))

        # Layer 2: Conv{In:14x14x6;Out:10x10x16} > Activ. > mxPooling{In:10x10x16;Out:5x5x16}
        x = tSign.conv_layer(x, 5, 6, 16, 'layer2', mu, sigma, 'VALID')
        if tRace: a += 1
        if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','x.shape',x.get_shape()))
        
        # Flatten. Input = 5x5xsize_out. Output = 400.
        with tf.name_scope('flatten'):
            x = flatten(x)  # tf.reshape(x, [-1, n_input])
            if tRace: a += 1
            if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','x.shape',x.get_shape()))

        # Layer 3: Fully Connected{In:400;Out:120} > Activ. > Dropout
        x = tSign.fc_layer(x, 400, 120, 'layer3', True, True, keep_prob)
        if tRace: a += 1
        if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','x.shape',x.get_shape()))
        
        # Layer 4: Fully Connected{In:120;Out:84} > Activ. > Dropout
        x = tSign.fc_layer(x, 120, 84, 'layer4', True, True, keep_prob)
        if tRace: a += 1
        if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','x.shape',x.get_shape()))
        
        # Layer 5: Fully Connected{In:84;Out:43}
        size_in, size_out = 84, 43
        nAme = 'fcLogits'
        with tf.name_scope(nAme):
            # Layer: Convolutional. Input = size_in. Output = size_out.
            w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name=nAme+"W")
            b = tf.Variable(tf.constant(0.1, shape=[size_out]), name=nAme+"B")
            logits = tf.add(tf.matmul(x, w), b, name='logits')
            # Add histogram summaries for weights and biases
            tf.summary.histogram(nAme+"_weights", w)
            tf.summary.histogram(nAme+"_biases", b)
            # Traces
            if tRace:a += 1
            if tRace:print(msg,' | [{}][{}][{}] {:14}: {} - {} = {}'.format('step2.5','-',a,'architecture','conv_layer fct','x.shape',x.get_shape()))

        if tRace: a += 1
        if tRace:print(msg,' | [{}][{}] {:14}: {} - {} = {}'.format('step2.5',a,'architecture','define model1','x.shape',x.get_shape()))          
        # EMBBEDED VISUALIZER
        embedding_input = logits
        embedding_size  = 43

        if tRace:print(msg1,' | [{}][{}] {:14}: {}, {}'.format('step2.5',2,'architecture','define model1','EMBBEDED VISUALIZER'))
        
        return logits, embedding_input, embedding_size        


# In[ ]: 2.3. Train, Validate and Test the Model
class trainMod(object):
    def __init__(self, sess):
        self.sess = sess
    
    # Initialization fo the training of the model -----------------------------
    def initTrain(self,tRace=tRace1, msg=msg2):
        # Initiate the Saves and restores mechanisum --------------------------
        metaModel = tf.train.Saver()
        if tRace:print(msg,' | [{}][{}] {:14}: {}'.format('step2.5',5,'metaModel','initiate saves & restores metaModel'))
        
        # Initialize all variables --------------------------------------------
        self.sess.run(tf.global_variables_initializer())
        n_train = len(X_train)
        if tRace:print(msg,' | [{}][{}] {:14}: {}, {} = {}'.format('step2.5',5,'initiation','initialize all variables','len X_train',n_train))
        
        return metaModel, n_train


    def modTrain(self,ltTrain,ltValid,ltTb,EPOCHS,BATCH_SIZE,trainingOpe,x,y,keep_prob,dropout,writer,assignment,merged,lossOpe,accuOpe,str1,metaModel,mod0,tRace=tRace1, msg=msg2):
        self.sess.run(tf.global_variables_initializer())
        n_train = len(ltTrain[0])
        n_valid = len(X_valid)
        total_cost, total_accuracy = 0, 0
        
        for i0 in range(EPOCHS):
            for start, end in zip(range(0, n_train, BATCH_SIZE), range(BATCH_SIZE, n_train+1, BATCH_SIZE)):
                xBatch, yBatch = ltTrain[0][start:end], ltTrain[1][start:end]
                self.sess.run(trainingOpe, feed_dict={x: xBatch, y: yBatch, keep_prob: dropout}) # dropout})
                #sess.run(assignment, feed_dict={x: xBatch, y: yBatch, keep_prob: 0.5})
        
            # TensorBoard: Embedding Visualization ----------------------------
            config, embedding_config = tBoard.eVisu(ltTb[0],ltTb[1],ltTb[2],ltTb[3],ltTb[4],ltTb[5],writer)
            
            # ----------------------------------------------- 
            #asgnVal = self.sess.run(assignment, feed_dict={x: ltValid[0], y: ltValid[1], keep_prob: 1})
            
            for start, end in zip(range(0, n_valid, BATCH_SIZE), range(BATCH_SIZE, n_valid+1, BATCH_SIZE)):
                xBatch, yBatch = X_valid[start:end], y_valid[start:end]
                cost, accuracy = self.sess.run([lossOpe, accuOpe], feed_dict={x: xBatch, y: yBatch, keep_prob: 1})
                total_cost     += (cost     * len(xBatch))
                total_accuracy += (accuracy * len(xBatch))
            
            total_cost     = total_cost / n_valid
            total_accuracy = total_accuracy / n_valid
            
            summary = self.sess.run(merged, feed_dict={x: X_valid, y: y_valid, keep_prob: 1})
                
            #metaModel.save(sess, str1)
            if not os.path.exists(str1):
                os.makedirs(str1)
            metaModel.save(self.sess, os.path.join(str1, mod0+'.ckpt')) #, i0)
        
            # Write summary
            writer.add_summary(summary, i0)
        
            # Report the accuracy
            print('Epoch: {:3} | cost : {:.3f} | Val.accu : {:.3f}'.format(i0, total_cost, total_accuracy)) #| asgnVal : {} ... ,asgnVal))

#        validation_accuracy = tSign.evaluate(ltValid[0], ltValid[1], accuOpe, BATCH_SIZE, x, y)        
#        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
        if tRace:print("...model saved")
        if tRace:print()
        if tRace:print(msg,' | [{}] {:14}'.format('step2.5','Train the model'))

    # Measure the test accuracy: ----------------------------------------------
    def modMeasure(dataImg,dataLabel,metaModel,str1,accuOpe,x,y,keep_prob):
        with tf.Session() as sess:    
            ## Step 13 - Need to initialize all variables
            sess.run(tf.global_variables_initializer())

            ## Step 14 - Evaluate the performance of the model on the test set
            metaModel.restore(sess, tf.train.latest_checkpoint(str1)) #pathMod))

            n_data = len(dataImg)
            total_accuracy = 0

            for start, end in zip(range(0, n_data, BATCH_SIZE), range(BATCH_SIZE, n_data+1, BATCH_SIZE)):
                xBatch, yBatch = dataImg[start:end], dataLabel[start:end]
                accuracy = sess.run(accuOpe, feed_dict={x: xBatch, y: yBatch, keep_prob: 1})
                total_accuracy += (accuracy * len(xBatch))
            
            total_accuracy = total_accuracy / n_data
            
            return total_accuracy


def report(X_train, y_train, X_valid, y_valid, X_test, y_test, X_valEV, y_valEV, X_testEV, y_testEV, tRace=tRace1):
    
    # Initialization
    #if tRace1:a = input('[0] < press the keyboard to continue >')
    
    sess, x, y, ch, one_hot_y, keep_prob = tSign.iNit(cMap, n_classes)
    if tRace:print(msg2,' | [{}][{}] {:14}: {}'.format('step2.5',0,'placeholders','features and labels'))
    #if tRace1:a = input('[1] < press the keyboard to continue >')
    
    # Create model ----------------------------------
    logits, embedding_input, embedding_size = modArc.model1(x,ch,0,0.1,keep_prob)
    mod0 = 'model1'
    if tRace:print(msg2,' | [{}][{}] {:14}: {}, {}'.format('step2.5',2,'architecture','create '+mod0,'EMBBEDED VISUALIZER'))
    
    
    # In[step2.5]: train the model
    print()
    print('Training... {}: {}, {}: {}, {}: {}, {}: {}, {}: {}'.format('model', mod0,'rate',rate,'epochs',EPOCHS,'batch size',BATCH_SIZE,'keep_prob',dropout))
    print()
    
    # Create cost & accuracy function ---------------------------------------------
    lossOpe, trainingOpe = tSign.loss(logits, one_hot_y, rate, mod0)
    accuOpe = tSign.accuracy(logits, one_hot_y, mod0)
    
    # TensorBoard: initialization of the Embedding Visualization ------------------
    merged, embedding, assignment = tBoard.iNitb(X_valEV, embedding_size, embedding_input)
    
    # Initialize the training of the model
    metaModel, n_train = trainMod(sess).initTrain()
    
    # Create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    str0, str1, writer = tBoard.logWriter(sess)
    LOGDIR = str1
    
    # Define the training function ------------------------------------------------
    ltTrain = [X_train,y_train]
    ltValid = [X_valEV,y_valEV]
    ltTb    = [sprImg,sprTsv,sprPath,LOGDIR,32,embedding]
    
    # Train the model -------------------------------------------------------------
    trainMod(sess).modTrain(ltTrain,ltValid,ltTb,EPOCHS,BATCH_SIZE,trainingOpe,x,y,keep_prob,dropout,writer,assignment,merged,lossOpe,accuOpe,str1,metaModel,mod0)
    if tRace:print(msg0,' | [{}] {:14}'.format('step2.5','train the model'))
    if tRace:print()
    
    # Measure the accuracy: ---------------------------------------------------
    total_accuracy = trainMod.modMeasure(X_valid,y_valid,metaModel,str1,accuOpe,x,y,keep_prob)
    print("Validation Accuracy = {:.3f}".format(total_accuracy))
    
    total_accuracy = trainMod.modMeasure(X_test,y_test,metaModel,str1,accuOpe,x,y,keep_prob)
    print("Test Accuracy = {:.3f}".format(total_accuracy))
    
    if tRace:print(msg0,' | [{}] {:14}'.format('step2.5','measure the validation and the test accuracies'))


def showAccu(ltLog,ltAccTst,ltAccVal,ltCost, cOlor = 'r', sTr = 'model1 - variation on learning rate', ratio = 1000):
    N = len(ltLog)
    # Plot both the validation and the test accuracies with various learning rates
    ltCost = [x * ratio for x in ltCost] 
    plt.scatter(ltAccTst,ltAccVal,s=ltCost, facecolors='none', edgecolors=cOlor)
    for i in range(N):
        if ltAccTst[i] < sorted(set(ltAccTst))[-2] and ltAccTst[i] > min(ltAccTst):
            plt.annotate(ltLog[i], (ltAccTst[i],ltAccVal[i]),  xycoords='data',
                    xytext=(-30, -30), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        elif ltAccTst[i] == min(ltAccTst):
            plt.annotate(ltLog[i], (ltAccTst[i],ltAccVal[i]),  xycoords='data',
                    xytext=(20, 0), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
        else:
            plt.annotate(ltLog[i], (ltAccTst[i],ltAccVal[i]),  xycoords='data',
                    xytext=(-60, 0), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
            
    plt.xlabel('test accuracy')
    plt.ylabel('validation accuracy')
    
    # Title
    plt.title(sTr,y=1.05)
    plt.show()

# In[ ]: 2.3.1. Train, Validate and Test the Model

# Valuate some parameters
n_classes  = 43
rate       = 0.00085
mu         = 0
sigma      = 0.1
EPOCHS     = 2
BATCH_SIZE = 100 # 1 # 50 # 100
dropout    = 0.67 # 0.67 # 0.25 # 0.5 # 0.75
sprImg     = '_sp_valid_2144x2144.png' #'_sp_valid_2144x2144.png' 
sprTsv     = '_sp_valid_2144x2144.tsv' # '_sp_valid_2144x2144.tsv'
sprPath    = pathTb


# Create data for the embedding visualization ---------------------------------
try:
    X_valEV, y_valEV   = tBoard.dataSprite(X_valid, y_valid, False)
    X_testEV, y_testEV = tBoard.dataSprite(X_test, y_test, False)
    X_valEV = np.float32(X_valEV)
    X_testEV = np.float32(X_testEV)    
    if tRace1:print(msg2,' | [{}][{}] {:14}:[{}] {}'.format('step2.5',0,'preliminaries','TensorBoard','create data for Embbeded Visualizer'))
except:
    if tRace1:print(msg3,' | [{}][{}] {:14}:[{}] {}'.format('step2.5',0,'preliminaries','TensorBoard','can not create data for Embbeded Visualizer'))


# Train, Validate and Test the Model ------------------------------------------
report(X_train, y_train, X_valid, y_valid, X_test, y_test, X_valEV, y_valEV, X_testEV, y_testEV)


# In[ ]: Step 3: Test a Model on New Images

# Helper functions: Show xSize*ySize images
def showTrace2(dataImg, dataLabel, xSize=1, ySize=5):
    fig0, ax0 = plt.subplots(xSize, ySize, figsize=(15,6))
    fig0.subplots_adjust(hspace=0.2, wspace=0.1)
    ax0 = ax0.ravel()

    # Get dictionary[idClass] = tuple( index(idClass))
    dct, dct0, dct1, lt = {}, {}, {}, []
    dct0, dct1, lt = ocrLabel(dataLabel)

    for i in range(xSize*ySize): 
        image = dataImg[i].squeeze()
        
        if image.shape[-1] == 3: 
            ch = 3
            cMap='rgb'
            ax0[i].imshow(image)
        elif image.shape[-1] == 32 or image.shape[-1] == 1:
            ch = 1
            cMap='gray'
            ax0[i].imshow(image, cmap = cMap)
        else:
            raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

        #ax0[i].imshow(image, cmap = cMap)
        title = dct0[dataLabel[i]][:17]+'.'
        ax0[i].set_title(title, fontsize=8)
        ax0[i].axis('off')

        
if tRace1:print(msg1,' | [{:5}] {}: {}'.format('step3.2.0','showTrace2','show xSize x ySize images'))


# Download, resize and store images into a list and convert it into an array
#myData  = pathData+'newData/_imgOK_/'
#myData  = pathData+'newData/_new_images_0/'
#myData  = pathData+'newData/_new_images_1/'
#myData  = pathData+'newData/_new_images_3/'
myData  = pathData+'newData/_ownData_/_serie01_/'
#myData  = pathData+'newData/_ownData_/_serie02_/'
#myData  = pathData+'newData/_ownData_/_serie03_/'

import glob
import matplotlib.image as mpimg
from PIL import Image

myImage, myLabel = [], []

for i, myImg in enumerate(glob.glob(myData+'*.png')):
    myLabel.append(int(myImg[len(myData):len(myData)+2]))       # int(myImg[0:1]))  # -6:-4]))
    image = cv2.imread(myImg)
    image = cv2.resize(image,(32,32),interpolation = cv2.INTER_CUBIC)
    myImage.append(image)

myImage = np.asarray(myImage)
print('< myLabel > = {}'.format(myLabel))


# Own images - Standarize, normalize and display grayscale data
myImg2 = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in myImage ])
myImg2 = (myImg2 - np.mean(myImg2))/np.std(myImg2)
myImg2 = myImg2[..., newaxis]
showTrace2(myImg2,myLabel,xSize=1, ySize=5)

# In[ ]:3.2. Predict the Sign Type for Each Image
# Prediction function ---------------------------------------------------------
def fctPrediction(myImg, pathCkp, tRace=tRace1, msg=msg1):
    ckptMeta = 'model1.ckpt.meta'
    if tRace:print(msg,' | [{:5}] {}: {}'.format('fctPrediction','<ckptMeta>',ckptMeta))
    
    image = myImg[0].squeeze()
    if image.shape[-1] == 3:  
        Ckp  = pathLog+pathCkp[0]+'/'  # with RGB images
        cMap ='rgb'
        ch   = 3      
        #if tRace:print(msg,' | [{:5}] {}: {}'.format('step3.2.1','<RGB>',pathCkp1[len(pathLog):-1]))
    elif image.shape[-1] == 32 or image.shape[-1] == 1:
        Ckp  = pathLog+pathCkp[1]+'/' # with GRAYSCALE images
        cMap ='gray'
        ch   = 1
        #if tRace:print(msg,' | [{:5}] {}: {}'.format('step3.2.1','<GRAY>',pathCkp2[len(pathLog):-1]))
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

        
    # Prediction
    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())        
        metaModel = tf.train.import_meta_graph(Ckp+ckptMeta)
        metaModel.restore(sess, tf.train.latest_checkpoint(Ckp))
        x = tf.get_default_graph().get_tensor_by_name("input:0")
        keep_prob = tf.placeholder(tf.float32)
        logits = tf.get_default_graph().get_tensor_by_name("logits:0")
        
        myInference = tf.argmax(logits, 1) #tf.nn.softmax(
        myPrediction = sess.run(myInference, feed_dict={ x: myImg, keep_prob: 1 })

    showTrace2(myImg, myPrediction,1,5)
    if tRace:print(msg,' | [{:5}] {}: {}{}{}'.format('step3.2.1','','prediction with',cMap,' images done'))

    return myPrediction, ch

if tRace1:print(msg1,' | [{:5}] {}: {}'.format('step3.2.0','fctPrediction','model 1, standarized and normalized data'))


# In[ ]: 3.2.1. Prediction with centered, normalized and jittered images
Ckp0 = input('< repo name of the model > : ')
pathCkp = [Ckp0,Ckp0]

myPredictionGRAY, ch = fctPrediction(myImg2, pathCkp)
print(('<myPrediction> : {}').format(myPredictionGRAY))


# Calculate the accuracy for these 5 new images. 
def calAccu(myLabel,myPrediction):
    a1 = [1 if c else 0 for c in [ i1 == i2 for (i1,i2)   in zip(myLabel, myPrediction)] ]
    try:
        return (sum(a1)/len(a1))* 100 # print('score = {0:.0f}%'.format((sum(a1)/len(a1))* 100))
    except ZeroDivisionError:
        print("Can't divide by zero")

# -------------
tRace1 = True

if ch == 3: 
    if tRace1:print(msg1,' | [{:5}] {}: {}'.format('step3.2.1','<RGB>',pathCkp[0]))
    myPrediction = myPredictionRGB[:]
else: 
    if tRace1:print(msg1,' | [{:5}] {}: {}'.format('step3.2.1','<GRAY>',pathCkp[1]))
    myPrediction = myPredictionGRAY[:]

print('')
myAccu = calAccu(myLabel,myPrediction)
print('myAccu = {0:.0f}%'.format(myAccu))

tRace1 = False

a = input('<Press any key to continue> ')

# In[ ]: 3.3. Output Top 5 Softmax Probabilities For Each Image Found on the Web

def chMap(image):
    if image.shape[-1] == 3:
        cMap ='rgb'
        ch   = 3
    elif image.shape[-1] == 32 or image.shape[-1] == 1:
        cMap ='gray'
        ch   = 1
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

    return cMap, ch


def topKPrediction(myImg, pathCkp, k = 5, tRace=tRace1, msg=msg1):
    ckptMeta = 'model1.ckpt.meta'
    if tRace:print(msg,' | [{:5}] {}: {}'.format('fctPrediction','<ckptMeta>',ckptMeta))
        
    image = myImg[0].squeeze()
    if image.shape[-1] == 3:  
        Ckp  = pathLog+pathCkp[0]+'/'  # with RGB images
        cMap ='rgb'
        ch   = 3      
        if tRace:print(msg,' | [{:5}] {}: {}'.format('step3.2.1','<RGB>',Ckp[len(pathLog):-1]))
    elif image.shape[-1] == 32 or image.shape[-1] == 1:
        Ckp  = pathLog+pathCkp[1]+'/' # with GRAYSCALE images
        cMap ='gray'
        ch   = 1
        if tRace:print(msg,' | [{:5}] {}: {}'.format('step3.2.1','<GRAY>',Ckp[len(pathLog):-1]))
    else:
        raise ValueError('[ERROR] info | channel : {}, Current image.shape: {}'.format(ch,image.shape))

    # Build the graph
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, shape = (None, 32, 32, ch))
        y = tf.placeholder(tf.uint8, shape = (None), name='label')
        keep_prob = tf.placeholder(tf.float32)
        logits, embedding_input, embedding_size = modArc.model1(x,ch,0,0.1,keep_prob)
        tK0 = tf.nn.top_k(tf.nn.softmax(logits), k)
        
    # Prediction
    with tf.Session(graph = graph) as sess:
        sess.run(tf.global_variables_initializer())
        metaModel = tf.train.import_meta_graph(Ckp+ckptMeta)
        metaModel.restore(sess, tf.train.latest_checkpoint(Ckp))
        tK1 = sess.run(tK0, feed_dict={ x: myImg, keep_prob: 1 })
        
        return tK1

# -----------------
tK = topKPrediction(myImg2, pathCkp, 43)
print('Top k Softmax Probabilities : {}'.format(tK))

# -----------------
fig, ax0 = plt.subplots(5, 2, figsize=(20, 10))
fig.subplots_adjust(hspace = 0.2, wspace=0.1)
ax0 = ax0.ravel()
for i, classId, value, image, label in zip(range(0,10,2), tK.indices, tK.values, myImage, myLabel):
    ax0[i].set_title(label, fontsize=8)
    ax0[i].axis('off')    
    ax0[i].imshow(image)
    ax0[i+1].yaxis.grid(color='#eeeeee')
    ax0[i+1].bar(classId, value, color='#616161')

# Helper functions: Show xSize*ySize images
def showTrace3(dataImg, dataLabel, myImg, myLabel, top_K ,xSize=5, ySize=7):
    fig0, ax0 = plt.subplots(xSize, ySize, figsize=(15,6))
    fig0.subplots_adjust(hspace=0.2, wspace=0.1)
    ax0 = ax0.ravel()
    
    dct = indexClass(dataLabel)
    c0, c1   = 0, 0
    img0 = np.zeros([32,32,3],dtype=np.uint8)
    img0.fill(255)
    
    for i in range(xSize*ySize):
        if i in range(0,xSize*ySize,ySize):
            # myImg
            image = myImg[c0].squeeze()
            title = myLabel[c0]
            ax0[i].set_title(title, fontsize=8)
            c0 += 1
        elif i in range(1,xSize*ySize,ySize):
            # blank
            image = img0[:]
            title= '' 
            ax0[i].set_title(title, fontsize=8)
        else:
            # dataImg
            idCls = top_K.indices[c0-1][c1%(ySize-2)]
            title = top_K.values[c0-1][c1%(ySize-2)]*100
            title = title.astype(int)
            ax0[i].set_title(str(idCls)+':'+str(title)+'%', fontsize=8)
            c1 += 1
            index = random.randint(dct[idCls][0], dct[idCls][-1])
            image = dataImg[index].squeeze()
        
        cMap, ch = chMap(image)
        
        if ch == 3:
            ax0[i].imshow(image)
        else:
            ax0[i].imshow(image, cmap = cMap)
        ax0[i].axis('off')   
        
if tRace1:print(msg1,' | [{:5}] {}: {}'.format('step1','showTrace','show xSize x ySize images'))

showTrace3(X_train, y_train, myImg2, myLabel, tK ,xSize=5, ySize=35)


