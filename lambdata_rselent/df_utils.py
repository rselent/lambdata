
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas

TEST_DF = pandas.DataFrame( [1,2,3])

class Utils:
    """
    Utilities to help with dataframe prediction management.

    trainValTest_splitOne: Splits ONE dataframe object into train, 
    validate, and test subsets.

    confusionMatrixPlot: Plots a confusion matrix to evaluate the 
    accuracy of a classification.
    """

    def __init__( self):
        pass

    def trainValTest_splitOne( self, df, trainSize= .7, strat= 'None', randomState= 16):
        """
        Splits ONE dataframe object into train, validate, and test subsets.

        Nested split of dataframe, using scikit-learn's train_test_split. 
        Splits validate subset first, at the inverse of given train size 
        value, then splits remaining 'train' dataframe into train and test 
        subsets using the same given train size.

        Parameters: 

        df: source Pandas dataframe to split
        
        trainSize: float; percent to keep for training subset dataframe 
        (default: 0.7)
        
        strat: int or string; feature/column to sort subsets by 
        (default: 'None')

        randomState: int; seed used by random number generator (default: 16)

        Returns:
        
        train: subset dataframe; of size (train = (df * trainSize) * trainSize)
        
        validate: subset dataframe; of size inverse to initial trainSize 
        (validate = df - (df * trainSize))
        
        test: subset dataframe; of size complement to train 
        (test = (df * trainSize) - ((df * trainSize) * trainSize))
        """
        dummyTrain, validate = train_test_split( df, train_size= trainSize, 
                                                stratify= df[ strat], 
                                                random_state= randomState)
        train, test = train_test_split( dummyTrain, train_size= trainSize, 
                                        stratify= dummyTrain[ strat], 
                                        random_state= randomState)
        return train, validate, test

    # NEWER: plot_confusion_matrix from scikit-learn (no seaborn or mpl req)
    def confusionMatrixPlot( self, yTrue, yPred):
        """
        Plots a confusion matrix to evaluate the accuracy of a classification.

        Parameters:

        yTrue: array, shape; correct target values.
        
        yPred: array, shape; estimated targets as returned by a classifier.

        Returns:

        Confusion matrix plot, built with Seaborn
        """
        import seaborn as sns
        labeled = unique_labels( yTrue)
        columned = [f'Predicted {label}' for label in labeled]
        indexed = [f'Actual {label}' for label in labeled]
        tabled = pd.DataFrame( confusion_matrix( yTrue, yPred),
                            columns= columned, index= indexed)
        return sns.heatmap( tabled, annot= True, fmt= 'd', cmap= 'viridis')