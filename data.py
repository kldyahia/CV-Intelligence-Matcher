from pathlib import Path

import numpy as  np
import pandas as pd


#### DO NOT Touch This!
DATASET_PATH = Path(__file__).resolve().parent / "penguins.csv"


FEATURES = [ "CulmenLength" , "CulmenDepth" , "FlipperLength" , "OriginLocation" , "BodyMass" ]

NUMERIC_FEATURES = [f for f in FEATURES if f !=  "OriginLocation" ]

TRAIN_PER_CLASS  = 30  # take tthe samples for ttrain and leave otther 20 for test (Split Requirement as is)

#TODO: @OLA,@Muram -> practice: rewrite full code from memory to understand it
#TODO: @Taj,@Khalid -> Solve Task 2

#TODO: @Fahim -> Do the integration work


#Message from Taj: work on your own local branches. No 'Accidental rewrite.' like last time :) whatever you want to do, do on your own machine but keep straight to tthe goal. 

class penguinData :

    def __init__( self, path = DATASET_PATH ) :      
        
        df  =  pd.read_csv( path )  
        
        df.columns     = df.columns.str.strip( ) 
        
        df ["Species"]  = df[ "Species" ].str.strip( )  

        for col in NUMERIC_FEATURES:
              
            df[ col ]  = pd.to_numeric(df [ col ] , errors = "coerce" )                   

        # encode categorical data to numerical values
        
        df [ "OriginLocation" ] = df [ "OriginLocation" ].str.strip( )             
        
        self.origins = sorted ( df [ "OriginLocation" ].dropna( ).unique( ) )              
        
        df [ "OriginLocation" ] = df [ "OriginLocation" ].map({o: i for i, o in enumerate(self.origins)})         

       
        self.class_names =  list( pd.unique( df[ "Species" ] ) )  # extract  three classes                
       
        self.labels = df [ "Species" ].map( { n: i for i , n in enumerate( self.class_names ) } ).to_numpy( )      
       
        self.raw    = df[ FEATURES ].astype( float )           
       
        self.table  =  self.raw         

    def imputer(self, train):
        # estimate replacements from train rows to avoid test leakage 
        training = self.raw.iloc[ train ]
        
        fillers = training [ NUMERIC_FEATURES ].mean( )
        
        fillers[ "OriginLocation" ] = training [ "OriginLocation" ].mode( ).iloc[ 0 ]
        
        self.table = self.raw.fillna ( fillers )
        
        return self

    def displayMatrixFunc( self , features ) :
        return self.table[ list( features )].to_numpy( )

    def expander(self, features, vals):
        
        vals = np.atleast_2d ( np.asarray ( vals, dtype=float ) )
        
        cols = [ ]
        
        for j, feat in enumerate(features):
            
            if feat == "OriginLocation" :
                # use one indicator per origin except the ref. origin
                cols.extend((vals[ :, j ] == level).astype( float ) for level  in range ( 1, len(self.origins )))
            else:
                cols.append(vals[ :, j] )
        return np.column_stack( cols ) 

    def matrix(self, features):
        return self.expander(features, self.displayMatrixFunc(features))

    def encoder( self, feat, val ) :
        
        val = str ( val ).strip( )
        
        if feat  ==  "OriginLocation" :
        
            if val not in self.origins:
        
                raise ValueError(f"originLocation must be one of: {', '.join( self.origins )}." )              
        
            return float ( self.origins.index( val ) )      
        
        try:
            num = float( val )        
        except ValueError:
            raise ValueError(f" { feat } must be a number") from None  
        if not np.isfinite(num):
            raise ValueError(f" { feat } must be a finite number ")   
        return num


def split_train_test( labels , class_ids , rng ) :        
    
    Tr, Ts = [ ], [ ]

    for id_c in class_ids :
        # shuffle each class that gets a  balanced sample  

        rows = rng.permutation(np.flatnonzero(labels == id_c))  
        
        Tr.extend(rows[ :TRAIN_PER_CLASS ])   
        Ts.extend(rows[ TRAIN_PER_CLASS: ])    
    
    return  np.array( Tr ), np.array( Ts )  


class Scaler :   

    def fitter( self, X ) :  
        # Storing Tr-Set stats for later scaling
        
        self.mean = X.mean( axis=0 ) 
        
        self.std = X.std( axis=0 ) 
        
        self.std [ self.std == 0 ]  = 1.0   
        
        return  self   

    def transformer( self, X ) :
        return (X - self.mean) /  self.std 

    def getOriginalUnity ( self, w, b ) : 
        return w / self.std, b - np.sum ( w * self.mean / self.std )          
