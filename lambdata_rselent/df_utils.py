
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas

TEST_DF = pandas.DataFrame( [1,2,3])

### splits ONE dataframe into train/val/test 
def TRAIN_VAL_TEST_SPLIT_ONE( df, trainSize, strat):

    dummyTrain, validate = train_test_split( df, train_size= trainSize, 
                                             stratify= df[ strat], 
                                             random_state= 16)
    train, test = train_test_split( dummyTrain, train_size= trainSize, 
                                    stratify= dummyTrain[ strat], 
                                    random_state= 16)
    return train, validate, test

### NEWER: plot_confusion_matrix from scikit-learn (no seaborn or mpl req)
def CON_MATRIX_PLOT( yTrue, yPred):
    labeled = unique_labels( yTrue)
    columned = [f'Predicted {label}' for label in labeled]
    indexed = [f'Actual {label}' for label in labeled]
    tabled = pd.DataFrame( confusion_matrix( yTrue, yPred),
                           columns= columned, index= indexed)
    return sns.heatmap( tabled, annot= True, fmt= 'd', cmap= 'viridis')