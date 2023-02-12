import pandas as pd 
import numpy as np 
from scipy import sparse
from sklearn.datasets import make_multilabel_classification
from os import listdir
from os.path import isfile, join
import copy 
import random 
import os 



def read_data(filepath , sparsify=False): 
    
    ''' read dataset from input file as a numpy array or sparse matrix
    
        @params: 
            filepath: path to input file (string)
            sparse: read as numpy array or sparse matrix (boolean) 
            
        returns: 
            D: dataset as numpy array or scipy sparse matrix           
            labels: numpy array of labels 
    '''
    
    # read comma separated data as panda dataframe 
    df = pd.read_csv(filepath, header=None) 
    
    #convert to standard binary form 
    D , labels = preprocessing(df)
    
    if sparsify: 
        return sparse.csr_matrix(D), sparse.csr_matrix(labels)
    else:
        return D , labels
         
 
def preprocessing(df): 
    
    '''convert categorical features in the data to binary 
       @params: 
            df: dataframe of raw data 
            
        returns: 
            D: dataset in binary form 
        
    '''
    
    # for simplicity we eclude all the columns with more than 15 distinct values 
    # change this for more toy or larger dataset s
    
    ''' only for flag dataset we do this here ''' 
    columns_to_keep = [i for i in range(len(df.nunique().values)) if (df.nunique().values[i] < 15 and df.nunique().values[i] >=3 )]
    
    label_columns = [i for i in range(len(df.nunique().values)) if df.nunique().values[i] < 3]
        
    # select columns 
    df_data = df[columns_to_keep]
    
    #select labels 
    df_labels = df[label_columns] 
    
    for col in df_labels.columns:
        df_labels[col] = df_labels[col].astype('category',copy=False)
    df_labels = pd.get_dummies(df_labels)
    
    # convert to dummy binary variables 
    # first need to convert all variables to categorical so pandas consider them in getting dummies 
    for col in df_data.columns:
        df_data[col] = df_data[col].astype('category',copy=False)
    df_data = pd.get_dummies(df_data)
    
    return df_data.to_numpy(), df_labels.to_numpy() # binary shapes 194 x 100 and 194 x 26 
    



def generate_synthetic_dataset_from_diverse_generating_rules(n_examples, n_features, n_labels, n_generating_rules, noise_proportion): 
    
    
    generating_rules_list = []
    all_covered_examples = []
    #features
    F = np.zeros((n_examples, n_features)) 
    #labelsnp.zeros((n_examples, n_features)) 
    L = np.zeros((n_examples, n_labels))
    
    example_list = [i for i in range(n_examples)]

    for i in range(n_generating_rules): 
    
        # create rule 
        #head_size = np.random.randint(low = 2, high = n_features)
        #label_size = np.random.randint(low = 1, high = n_labels)
        #head = random.sample([j for j in range(n_features)], k = head_size) 
        #tail = random.sample([h for h in range(n_labels)], k = label_size)
    
        attribute_weights = [1 for i in range(n_features)]
        label_weights = [1 for i in range(n_labels)]
        
        head_size = random.choices([x for x in range(2,n_features)], weights=[1/x for x in range(2,n_features)], k=1)[0]
        label_size = random.choices([x for x in range(2,n_labels)], weights=[1/x for x in range(2,n_labels)], k=1)[0] 
        
        head = list(set(random.choices([j for j in range(n_features)], weights=attribute_weights, k = head_size)))
        tail = list(set(random.choices([h for h in range(n_labels)], weights=label_weights,  k = label_size)))
        
        
        
        # update head and tail weights 
        for attr in head: 
            attribute_weights[attr] = attribute_weights[attr]/2 
        for lab in tail: 
            label_weights[lab] = label_weights[lab]/2 
        
        rule = ( head , tail )  
    
        generating_rules_list.append(rule) 
    
        # rule coverage 
        covered_examples = random.sample(example_list, np.random.randint(int(n_examples/n_generating_rules)))
    
        all_covered_examples.append(covered_examples)        
        
        example_list = [i for i in example_list if i not in covered_examples]
    
        for example in covered_examples: 
            for attr in rule[0]: 
                F[example][attr] = 1 
            for lab in rule[1]: 
                L[example][lab] = 1
            
            
    # finally add noise 
    # features
    F_noisy = copy.deepcopy(F)
    # labels
    L_noisy = copy.deepcopy(L)
    
    # inject noise 
    #for noise_proportion in noise_proportions: 
    to_be_swapped_features = int( np.prod(F.shape) * noise_proportion)
    to_be_swapped_labels = int(np.prod(L.shape) * noise_proportion)
    
    # inject noise into features 
    for _ in range(to_be_swapped_features): 
    
        # sample row 
        index_row = np.random.randint(F.shape[0])
        # sample feature 
        index_feature = np.random.randint(F.shape[1])
        
        # note we may sample the same (index_row, index_feature) twice and thus the swapping has no effect - this should be very unlikely for small noise proportion 
        # but it means in gneral that the noise proportion gives an upper bound on the swaps rather than the exact amount of swaps 
    
        # swap 
        F_noisy[index_row][index_feature]  = 1 - F_noisy[index_row][index_feature]
    
    # inject noise into labels 
    for _ in range(to_be_swapped_labels): 
        
        # sample row 
        index_row = np.random.randint(L.shape[0])
        # sample feature 
        index_feature = np.random.randint(L.shape[1])
        
        # swap 
        L_noisy[index_row][index_feature] = 1 - L_noisy[index_row][index_feature]

    
        
        
        
    # now generate test set of the appropriate size 
    
    n_example_test = int(n_examples * 0.4) 
    example_test_list = [i for i in range(n_example_test)]
        
         
    F_test = np.zeros((n_examples, n_features)) 
    L_test = np.zeros((n_examples, n_labels)) 
        
    for r in range(len(generating_rules_list)): 
        
        rule = generating_rules_list[r] 
    
        covered_examples = random.sample(example_test_list, np.random.randint(n_example_test/n_generating_rules))
        
        example_test_list = [i for i in example_test_list if i not in covered_examples]
    
        for example in covered_examples: 
            for attr in rule[0]: 
                F_test[example][attr] = 1 
            for lab in rule[1]: 
                L_test[example][lab] = 1
                
        
    # finally add noise 
    #features
    F_noisy_test = copy.deepcopy(F_test)
    #labels
    L_noisy_test = copy.deepcopy(L_test)
    
    # inject noise 
    #for noise_proportion in noise_proportions: 
    to_be_swapped_features = int( np.prod(F_test.shape) * noise_proportion)
    to_be_swapped_labels = int(np.prod(L_test.shape) * noise_proportion)
    
    # inject noise into features 
    for _ in range(to_be_swapped_features): 
    
        # sample row 
        index_row = np.random.randint(F_test.shape[0])
        # sample feature 
        index_feature = np.random.randint(F_test.shape[1])
       
        # swap 
        F_noisy_test[index_row][index_feature]  = 1 - F_noisy_test[index_row][index_feature]
    
    # inject noise into labels 
    for _ in range(to_be_swapped_labels): 
        
        # sample row 
        index_row = np.random.randint(L_test.shape[0])
        # sample feature 
        index_feature = np.random.randint(L_test.shape[1])
        
        # swap 
        L_noisy_test[index_row][index_feature] = 1 - L_noisy_test[index_row][index_feature]
    

    return F_noisy, L_noisy, F_noisy_test, L_noisy_test,  generating_rules_list 



def generate_synthetic_dataset_from_generating_rules(n_examples, n_features, n_labels, n_generating_rules, noise_proportion): 
    
    
    generating_rules_list = []
    all_covered_examples = []
    
    #features
    F = np.zeros((n_examples, n_features)) 
    #labelsnp.zeros((n_examples, n_features)) 
    L = np.zeros((n_examples, n_labels))
    
    example_list = [i for i in range(n_examples)]
        
    for i in range(n_generating_rules): 
        
        # create rule 
        head_size = np.random.randint(low = 2, high = n_features)
        label_size = np.random.randint(low = 1, high = n_labels)
        head = random.sample([j for j in range(n_features)], k = head_size) 
        tail = random.sample([h for h in range(n_labels)], k = label_size)
       
        rule = ( head , tail )  
       
        generating_rules_list.append(rule) 
       
        # rule coverage 
        covered_examples = random.sample(example_list, np.random.randint(n_examples/n_generating_rules))

        all_covered_examples.append(covered_examples)        
        
        for example in covered_examples: 
            for attr in rule[0]: 
                F[example][attr] = 1 
            for lab in rule[1]: 
                L[example][lab] = 1
        
        # finally add noise 
        # features
        F_noisy = copy.deepcopy(F)
        # labels
        L_noisy = copy.deepcopy(L)
        
        # inject noise 
        #for noise_proportion in noise_proportions: 
        to_be_swapped_features = int( np.prod(F.shape) * noise_proportion)
        to_be_swapped_labels = int(np.prod(L.shape) * noise_proportion)
        
        # inject noise into features 
        for _ in range(to_be_swapped_features): 
        
            # sample row 
            index_row = np.random.randint(F.shape[0])
            # sample feature 
            index_feature = np.random.randint(F.shape[1])
            
            # note we may sample the same (index_row, index_feature) twice and thus the swapping has no effect - this should be very unlikely for small noise proportion 
            # but it means in gneral that the noise proportion gives an upper bound on the swaps rather than the exact amount of swaps 
        
            # swap 
            F_noisy[index_row][index_feature]  = 1 - F_noisy[index_row][index_feature]
        
        # inject noise into labels 
        for _ in range(to_be_swapped_labels): 
            
            # sample row 
            index_row = np.random.randint(L.shape[0])
            # sample feature 
            index_feature = np.random.randint(L.shape[1])
            
            # swap 
            L_noisy[index_row][index_feature] = 1 - L_noisy[index_row][index_feature]

        
        
        
        
    # now generate test set of the appropriate size 
    
    n_example_test = int(n_examples * 0.3) 
    example_test_list = [i for i in range(n_example_test)]
        
         
    F_test = np.zeros((n_examples, n_features)) 
    L_test = np.zeros((n_examples, n_labels)) 
        
    for r in range(len(generating_rules_list)): 
        
        rule = generating_rules_list[r] 
    
        covered_examples = random.sample(example_test_list, np.random.randint(n_example_test/n_generating_rules))
    
        for example in covered_examples: 
            for attr in rule[0]: 
                F_test[example][attr] = 1 
            for lab in rule[1]: 
                L_test[example][lab] = 1
                
        
    # finally add noise 
    #features
    F_noisy_test = copy.deepcopy(F_test)
    #labels
    L_noisy_test = copy.deepcopy(L_test)
    
    # inject noise 
    #for noise_proportion in noise_proportions: 
    to_be_swapped_features = int( np.prod(F_test.shape) * noise_proportion)
    to_be_swapped_labels = int(np.prod(L_test.shape) * noise_proportion)
    
    # inject noise into features 
    for _ in range(to_be_swapped_features): 
    
        # sample row 
        index_row = np.random.randint(F_test.shape[0])
        # sample feature 
        index_feature = np.random.randint(F_test.shape[1])
       
        # swap 
        F_noisy_test[index_row][index_feature]  = 1 - F_noisy_test[index_row][index_feature]
    
    # inject noise into labels 
    for _ in range(to_be_swapped_labels): 
        
        # sample row 
        index_row = np.random.randint(L_test.shape[0])
        # sample feature 
        index_feature = np.random.randint(L_test.shape[1])
        
        # swap 
        L_noisy_test[index_row][index_feature] = 1 - L_noisy_test[index_row][index_feature]
    

    return F_noisy, L_noisy, F_noisy_test, L_noisy_test,  generating_rules_list 


def generate_synthetic_dataset(n_samples=20, n_features=10, n_classes = 10, n_labels = 5, density_features = 0.5): 
    
    '''
    generate synthetic data using sklearn (sample number of labels with Poisson and labels with Multinomial)
    
    @params: 
        n_samples: The number of samples
        n_features: The total number of features
        n_classes: The number of classes of the classification problem.
        n_labels: The average number of labels per instance

    return 
    
    '''
    
    
    
    D, labels = make_multilabel_classification(n_samples=n_samples, n_features=n_features,
                                               n_classes = n_classes, n_labels = n_labels,random_state=1)
    
    D = D.astype(int)
    
    D[D<np.quantile(D, density_features)] = 1
    D[D>=np.quantile(D, density_features)] = 0
    
    return D, labels 
    





def read_data_arff(filepath, n_labels, sparsity = 0.5, train_test_split = 0.7, sparse=False): 
    
    ''' Wrapper for reading data from MULAN repository in arff format - to be used whenever the datasets 
    contain numeric features (perform binarization of numericla features) 
            
    
        @params: 
            filepath: path to folder containing arff datasets  
            n_labels: number of labels assumed to be the last attributes
            sparsity: proportion of null values 
            train_test_split: efloat scalar specifying the proportion of 
            data records to be used for training 
    
    '''
    dataset = filepath.split("/")[-1]
    filepath_base = filepath
    files_within_directory = [f for f in listdir(filepath_base) if isfile(join(filepath_base, f))]
    
    for file in files_within_directory: 
        if "train" in file: 
            train_test_split = False # in this case we have aleady the data split 
        
    if sparse: 
        
        if not train_test_split: 
            filepath_train = filepath_base + "/" + dataset + "-train" + ".arff"
            filepath_test = filepath_base + "/" + dataset + "-test" + ".arff"
            D_train, labels_train, label_names = read_data_arff_sparse(filepath_train, n_labels) 
            D_test, labels_test, label_names = read_data_arff_sparse(filepath_test, n_labels) 
            
        else:
            
            filepath = filepath_base + "/" + dataset + ".arff"
            D, labels, label_names = read_data_arff_sparse(filepath, n_labels) 
            D_train, labels_train, D_test, labels_test = shuffle_split_data(D, labels,proportion_train=train_test_split)
    
    else:
        
        if not train_test_split: 
            
            filepath_train = filepath_base + "/" + dataset + "-train" + ".arff"
            filepath_test = filepath_base + "/" + dataset + "-test" + ".arff"            
            
            if dataset=="genbase":
                D_train, labels_train =  read_data_arff_dense_genbase(filepath_train, n_labels, sparsity) 
                D_test, labels_test =  read_data_arff_dense_genbase(filepath_test, n_labels, sparsity) 
            
            else:
                D_train, labels_train, label_names =  read_data_arff_dense(filepath_train, n_labels, sparsity) 
                D_test, labels_test, label_names =  read_data_arff_dense(filepath_test, n_labels, sparsity) 
            
        
        else:
            filepath = filepath_base + "/" + dataset + ".arff"
            D, labels, label_names = read_data_arff_dense(filepath, n_labels, sparsity) 
            D_train, labels_train, D_test, labels_test = shuffle_split_data(D, labels,proportion_train=train_test_split)
            
        
        
    return D_train, labels_train, D_test, labels_test , label_names
   
    

    
def shuffle_split_data(X, y, proportion_train):
    
    indices_split = np.random.choice(range(X.shape[0]), int(proportion_train*X.shape[0]))

    D_train = X[indices_split]
    labels_train = y[indices_split]
    D_test =  X[~indices_split]
    labels_test = y[~indices_split]

    return D_train, labels_train, D_test, labels_test



def read_data_arff_sparse(filepath, n_labels): 
    
    ''' read data from MULAN repository in arff format
        (bibtex,..)
        
        This function works for data that are stored in sparse format : bibxtex, bookmarks 
        
    @params: 
        n_labels: number of labels assumed to be the last attributes
        '''
    
    f = open(filepath, "r") 
    allines = f.readlines() 
    i = 2
    
    
    # collect set of numeric features 
    all_feats = [] 
    while allines[i].split(" ")[0] == "@attribute": 
        all_feats.append(allines[i].split(" ")[1])
        i+=1         
        
        
    label_names = all_feats[-n_labels:]
    
    
    n_cols = i - 2  
    i+=2 
    A = np.zeros((len(allines) - i, n_cols), dtype = np.ushort)
    row_counter = 0
    while i < len(allines): 
        
        line_list = allines[i][1:-1].split(",")
            
        for el in line_list: 
        
            a, b = el.split(" ") 
                    
            A[row_counter][int(a)]=1 
            
        i+=1 
        row_counter+=1 
        
    split_column = n_cols - n_labels
    
    D = A[:,:split_column]
    labels = A[:,split_column:] 
        
    return D, labels , label_names

    
def read_data_arff_dense(filepath, n_labels, sparsity = 0.5): 
    
    ''' read data from MULAN repository in arff format
        (bibtex,..)
        
        This function works for data that are stored in dense format - 
        note that labels must be binary and not categorical with more than two labels 
        
    @params: 
        n_labels: number of labels assumed to be the last attributes
        
        '''
    
    f = open(filepath, "r") 
    allines = f.readlines() 
    i = 2
    
    # collect set of numeric features 
    all_feats = []
    numeric_feats = list() 
    while allines[i].split(" ")[0] == "@attribute": 
        if allines[i].split(" ")[2]=="numeric\n": 
            numeric_feats.append(i-2) 
        all_feats.append(allines[i].split(" ")[1])
        i+=1 
      
        
    
        
    label_names = all_feats[-n_labels:]
    
    n_cols = i - 2    
        
    i+=2 
    A = np.zeros((len(allines) - i, n_cols), dtype = np.float16)
    row_counter = 0
    
    
    while i < len(allines): 

        line_list = allines[i][1:-1].split(",")
        
        for j in range(len(line_list)): 
            if len(line_list[j])==0: 
                line_list[j] = 0 
    
        A[row_counter] = list(map(float, line_list))
        
        i+=1 
        
        row_counter+=1 
    
    split_column = n_cols - n_labels
    
    D = A[:,:split_column]
    
    labels = A[:,split_column:] 
    
    split_column = n_cols - n_labels

    D = A[:,:split_column]
    labels = A[:,split_column:] 
        
    
    for i in numeric_feats:
        
        booleans = D[:,i]<=np.quantile(D[:,i], sparsity)
        
        for j in range(len(booleans)): 
            if booleans[j] == True: 
                D[j,i] = 0 
            else: 
                D[j,i] = 1 
    
    # identify categorical attributes that are not numerical, for which we will create dummy variables 
    cat_feats = [x for x in range(D.shape[1]) if x not in numeric_feats]
            
    # more than two values attributes 
    not_binary_feats = [x for x in cat_feats if len(np.unique(D[:,x])) > 2]
            # binary attributes
    binary_feats = [x for x in cat_feats if len(np.unique(D[:,x])) <= 2]
    
    # create dummy variables for them - use pandas for this 
    pd_df = pd.DataFrame(D[:, not_binary_feats])
    
    # identify categorical attributes that are not numerical, for which we will create dummy variables 
    # create dummy variables for them - use pandas for this     
    if len(not_binary_feats) > 0:     
        
        for col in pd_df.columns:    
            pd_df[col] = pd_df[col].astype('category', copy=False)
            
        dummies = pd.get_dummies(pd_df).to_numpy()
        
        D_stacked = np.hstack( ( D[:, [i for i in range(D.shape[1]) if i not in not_binary_feats]] , dummies ) )
        
        return D_stacked.astype('int'), labels.astype('int') , label_names
    
    else:
        
        
        return D.astype('int'), labels.astype('int') , label_names
    
    
    
    
    
    

def read_data_arff_dense_genbase(filepath, n_labels, sparsity = 0.5): 
    
    ''' read data from MULAN repository in arff format
        (bibtex,..)
        
        This function works for data that are stored in dense format - 
        note that labels must be binary and not categorical with more than two labels 
        
    @params: 
        n_labels: number of labels assumed to be the last attributes
        
        '''
        
        
    f = open(filepath, "r") 
    allines = f.readlines() 
    i = 3
    
    # collect set of numeric features 
    numeric_feats = list() 
    while allines[i].split(" ")[0] == "@attribute": 
        if allines[i].split(" ")[2]=="numeric\n": 
            numeric_feats.append(i-2) 
        i+=1 
       
        
    
    n_cols = i - 2    
        
    i+=2 
    A = np.zeros((len(allines) - i, n_cols-1), dtype = np.float16)
    row_counter = 0
    
    
    while i < len(allines): 

        line_list = allines[i][1:-1].split(",")[1:]
                
        coded_line_list = [] 
        for el in line_list: 
            if el=="YES": 
                coded_line_list.append(1) 
            elif el=="NO":
                coded_line_list.append(0) 
            else:
                if el=="": 
                    to_append = 0 
                else: 
                    to_append = el 
                coded_line_list.append(to_append) 
            
        
     
    
        A[row_counter] = list(map(float, coded_line_list))
        
        i+=1 
        
        row_counter+=1 
    
    split_column = n_cols - n_labels
    
    D = A[:,:split_column]
    
    labels = A[:,split_column:] 
    
    split_column = n_cols - n_labels

    D = A[:,:split_column]
    labels = A[:,split_column:] 
        
    
    
    for i in numeric_feats:
        
        booleans = D[:,i]<np.quantile(D[:,i], sparsity)
        
       
        for j in range(len(booleans)): 
            if booleans[j] == True: 
                D[j,i] = 0 
            else: 
                D[j,i] = 1 

        
    
    # identify categorical attributes that are not numerical, for which we will create dummy variables 
    cat_feats = [x for x in range(D.shape[1]) if x not in numeric_feats]
            
    # more than two values attributes 
    not_binary_feats = [x for x in cat_feats if len(np.unique(D[:,x])) > 2]
            # binary attributes
    binary_feats = [x for x in cat_feats if len(np.unique(D[:,x])) <= 2]
    
            
    # create dummy variables for them - use pandas for this 
    pd_df = pd.DataFrame(D[:, not_binary_feats])
    
    # identify categorical attributes that are not numerical, for which we will create dummy variables 
    # create dummy variables for them - use pandas for this     
    if len(not_binary_feats) > 0:     
        
        for col in pd_df.columns:    
            pd_df[col] = pd_df[col].astype('category', copy=False)
            
        dummies = pd.get_dummies(pd_df).to_numpy()
        
                
        D_stacked = np.hstack( ( D[:, [i for i in range(D.shape[1]) if i not in not_binary_feats]] , dummies ) )
        
        return D_stacked.astype('int'), labels.astype('int')
    

    else:
        
        
        return D.astype('int'), labels.astype('int') 
    
    

    
