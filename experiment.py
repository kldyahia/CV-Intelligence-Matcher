from dataclasses import dataclass

import numpy as np

from data import FEATURES, TRAIN_PER_CLASS, Scaler
from models import backProp

#TODO: @OLA,@Muram -> practice: rewrite full code from memory to understand it
#TODO: @Taj,@Khalid -> Solve Task 2

#TODO: @Fahim -> Do the integration work

#Message from Taj: work on your own local branches. No 'Accidental pushes.' like last time :) whatever you want to do, do on your own machine but keep straight to tthe goal. 

#Notice: Fn names must be verbs. 

@dataclass
class Result:
    data: object
    
    model: backProp
    scaler: Scaler
    
    feats: tuple
    classNames: tuple
    
    Tr_acc: float
    Ts_acc: float
    
    conf: np.ndarray

    def classifierFn(self, values):
     
        #Return predictions for one sample in order of features 
     
        row = np.array ( [ [ self.data.encoder ( feature, value ) for feature , value in zip ( self.feats, values ) ] ] )
     
        prediction = self.model.predictor( self.scaler.transformer( row ))[ 0 ]       
     
        return  self.classNames [ prediction ]



def ExperimentRunner( data, 
                   hidden_layers, 
                   eta, 
                   epochs, 
                   use_bias, 
                   activation, 
                   seed=42
                   ) :

    
# train on first 30 per species and test on 20
   
    classIDs = np.arange(len( data.class_names ))  
    TrPart , TsPart = [ ], [ ]    

    #preserve data set order within each class for the 30/20  split
    for ids in classIDs:
        
        rows = np.flatnonzero( data.labels  == ids )   
        
        if len(rows) < TRAIN_PER_CLASS + 20:
        
            raise ValueError(f"each of the classes need at least { TRAIN_PER_CLASS + 20 } samples.")
        
        TrPart.append(rows[ :TRAIN_PER_CLASS ])
        TsPart.append(rows[ TRAIN_PER_CLASS : TRAIN_PER_CLASS +  20 ])    

    Tr  = np.concatenate(TrPart)
    Ts  = np.concatenate(TsPart)

    # fit and apply functions
    data.imputer ( Tr )
    X      = data.displayMatrixFunc ( FEATURES ).astype( float )
    scaler = Scaler().fitter (X [ Tr ] ) 
    XTr = scaler.transformer (X [ Tr ] ) 
    XTs = scaler.transformer (X [ Ts ] ) 


## return object backprob with the attributes set as follows
    model = backProp (
        inCount =len ( FEATURES ) ,  
        hiddenLayers =hidden_layers ,  
        OutCount =len ( classIDs) ,  
        eta =eta ,
        epochCount =epochs ,  
        biasToggle =use_bias ,  
        activationType =activation , 
        seed =seed ,                             
    )
    model.backpropFitter( XTr , data.labels[ Tr ] )

    preds_Tr = model.predictor( XTr )  
    preds_Ts = model.predictor( XTs )  

    ## calculate accs on train and test
    acc_Tr = float( np.mean( preds_Tr ==  data.labels[ Tr ]))  
    acc_Ts = float( np.mean( preds_Ts ==  data.labels[ Ts ]))  

    ## init a confusion matrix with hte input values, store actual vs predicted
    confVals = np.zeros(( len( classIDs ), len( classIDs )) , dtype = int )  
    
    for actual, predicted in zip( data.labels[ Ts ] , preds_Ts ) :  
        confVals[actual, predicted] += 1

    # return object result with the attributes set as follows
    return Result(
        data =data , 
        model =model ,
        scaler =scaler ,
        feats =tuple ( FEATURES ) ,
        classNames = tuple( data.class_names ) , 
        Tr_acc =acc_Tr, 
        Ts_acc =acc_Ts,    
        conf =confVals,        
    )
