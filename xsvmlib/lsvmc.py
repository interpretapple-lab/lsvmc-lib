from xsvmlib.xsvmc import xSVMC 
from xsvmlib.lmodels import LogicAAD, AugmentedLGrade, AugmentedLPrediction, AugmentedLGradeAggregator, LGrade, LGradeAggregator
import numpy as np
from sklearn.metrics import f1_score
import inspect

class lSVMC(xSVMC):
    """Explainable Support Vector Machine Classification
    
    This class is an implementation of the variant of the *support vector machine* (SVM)[1] classification process, 
    called *explainable SVM classification* (XSVMC), proposed in [2]. In XSVMC the most influential support vectors (MISVs) 
    are used for identifying what has been relevant to the classification. These MISVs can be used for contextualizing the 
    evaluations in such a way that the forthcoming predictions can be explained with ease.
    
    This implementation is based on Scikit-learn SVC class.
  
    Parameters:

        k: int, default=1
            Number of possible classes expected for the prediction output.

    References:
        [1] V.N.Vapnik,The Nature of Statistical Learning Theory, Springer-Verlag, New York, NY, USA, 1995.
            http://dx.doi.org/10.1007/978-1-4757-3264-1

        [2] M. Loor and G. De Tré. Contextualizing Support Vector Machine Predictions.
            International Journal of Computational Intelligence Systems, Volume 13, Issue 1, 2020,
            Pages 1483-1497,  ISSN 1875-6883, https://doi.org/10.2991/ijcis.d.200910.002

        
    """
        
    def __init__(
        self,
        *,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovo",
        break_ties=False,
        random_state=None,
        k = 1
    ):

        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

        if(not isinstance(k, int)):
            raise ValueError("K parameter must be an integer")
        elif(k < 1):
            raise ValueError("K parameter cannot lower than 0")
        self.k = k


    # ----------------------------------------
    #       Using L-grades  
    #-----------------------------------------
        

    def build_p_catalog(self):
        """ Builds a catalog that maps a tuple of classes (i, j) to the position p of a vector with shape (n*(n-1)/2,)
            
        Parameters
                None. 
        Returns
                Nothing.

        """

        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        
        n_classes = len(self.classes_)
        catalog = {}
        p = 0
        for i in range(n_classes-1):
            for j in range(i+1,n_classes):
                catalog[(i,j)] = p
                p = p+1
        self.p_catalog_ = catalog
        return
      
    def compute_M_values(self, X, y):
        """Computes the maximum absolute values of the margins for each pair of classes using the training set X. 
        
        Parameters
            X :  ndarray of shape (n_samples, n_features), or
             ndarray of shape (n_features)

            y:   ndarray of shape (n_samples)

        Returns
            M : ndarray of shape (n_classes*(n_classes-1)/2, ) consisting of the M values.

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")


        # First, sort X per class
        idx_sorted = np.argsort(y) #y[:, 0].argsort()
        y_sorted = y[idx_sorted]
        X_sorted = X[idx_sorted]

        idx_unique = np.unique(y_sorted, return_index=True)

        partitions = [[idx_unique[0][i], idx_unique[1][i], 0] for i in range(len(idx_unique[0]))]
        last_idx = len(X_sorted)
        for i in range(len(idx_unique[0])-1,-1,-1):
            partitions[i][2] = last_idx
            last_idx = partitions[i][1]

        df_sorted = self.decision_function(X_sorted)

       
        n_classes = len(self.classes_)
        n_comparisons = int(n_classes*(n_classes-1)/2)
        M = np.zeros(shape=(n_comparisons,))
        if n_classes == 2:
            M[0] = np.max(np.abs(df_sorted))
        else:
            # Computing the M values for n*(n-1)/2 binary classifiers
            p = 0
            for i in range(n_classes-1):
                for j in range(i+1,n_classes): 
                    # result of decision_function for training examples of class i
                    dfi = df_sorted[ partitions[i][1]: partitions[i][2], p]
                    Mi = np.max(np.abs(dfi))
                    # result of decision_function for training examples of class j
                    dfj = df_sorted[ partitions[j][1]: partitions[j][2], p ]
                    Mj = np.max(np.abs(dfj))
                    # M-value for i-vs.-j classifier
                    M[p] = max(Mi, Mj)
                    p += 1
        
        self.M_values_ = M
        return M
    

    def get_M_value(self, i, j):
            """Gets the M-value computed for the binary classifier ij 
            
            Parameters
            i :  index of the first class

            j:   index of the second class

            Returns
            M_value : M-value.

            """
            
            if not hasattr(self, "M_values_"):
                raise Exception("Call 'compute_M_values' with appropriate arguments before using this method.")

            if not hasattr(self, "p_catalog_"):
                # build the catalog
                self.build_p_catalog()

            p = 0
            if (i,j) not in self.p_catalog_:
                if (j,i) not in self.p_catalog_:
                    raise Exception("Invalid indices.")
                else:
                    p = self.p_catalog_[(j,i)]
            else:
                p = self.p_catalog_[(i,j)]

            return self.M_values_[p]


    

    def compute_c_grades(self, X, y, confidence_fn = f1_score):
            """Computes the confidence grades for the previously trained n*(n-1)/2 binary classifiers 
            
            Parameters
            X :  ndarray of shape (n_samples, n_features), or
                ndarray of shape (n_features). X should be the collection that is used for measuring the
                performance of the binary classifiers.

            y:   ndarray of shape (n_samples)

            confidence_fn: Function for computing the confidence values of the binary classifiers.

            Returns
            c_grades : ndarray of shape (n_classes*(n_classes-1)/2, ) consisting of the confidence grades.

            """
            
            if not hasattr(self, "classes_"):
                raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

            if not isinstance(X,np.ndarray):
                raise ValueError("X parameter must be an ndarray")


            args_fn  = inspect.signature(confidence_fn)
            has_pos_label = 'pos_label' in args_fn.parameters.keys()

            # First, sort X per class
            idx_sorted = np.argsort(y, axis=0) #y[:, 0].argsort()
            idx_sorted = idx_sorted.ravel()
            y_sorted = y[idx_sorted]
            X_sorted = X[idx_sorted]

            idx_unique = np.unique(y_sorted, return_index=True)

            partitions = [[idx_unique[0][i], idx_unique[1][i], 0] for i in range(len(idx_unique[0]))]
            last_idx = len(X_sorted)
            for i in range(len(idx_unique[0])-1,-1,-1):
                partitions[i][2] = last_idx
                last_idx = partitions[i][1]

            df_sorted = self.decision_function(X_sorted)

            # Computing confidence grades for n*(n-1)/2 binary classifiers
            cs = self.classes_
            n_classes = len(self.classes_)
            n_comparisons = int(n_classes*(n_classes-1)/2)
            c_grades = np.zeros(shape=(n_comparisons,))


            if n_classes == 2:
                y_pred_i = np.where(df_sorted>0, cs[1], cs[0])
                y_real_i = y_sorted
                c_grades[0] = confidence_fn(y_real_i, y_pred_i,pos_label=cs[0]) if has_pos_label else confidence_fn(y_real_i, y_pred_i)
            else:
                p = 0
                for i in range(n_classes-1):
                    for j in range(i+1,n_classes): 
                        # result of decision_function for test examples of class i
                        dfi = df_sorted[ partitions[i][1]: partitions[i][2], p]
                        y_real_i = y_sorted[partitions[i][1]: partitions[i][2]]
                        y_pred_i = np.where(dfi>0, cs[i], cs[j])
                        # result of decision_function for training examples of class j
                        dfj = df_sorted[ partitions[j][1]: partitions[j][2], p]
                        y_real_j = y_sorted[partitions[j][1]: partitions[j][2]]
                        y_pred_j = np.where(dfj<0, cs[j], cs[i]) # NB: the negative sign
                        # C-value for i-vs.-j classifier
                        y_pred_ij = np.concatenate((y_pred_i, y_pred_j), axis=None)
                        y_real_ij = np.concatenate((y_real_i, y_real_j), axis=None)
                        #
                        c_grades[p] = confidence_fn(y_real_ij, y_pred_ij,pos_label=cs[i]) if has_pos_label else confidence_fn(y_real_ij, y_pred_ij)
                        p += 1
            
            self.c_grades_ = c_grades
            return c_grades
    


    def get_c_grade(self, i, j):
            """Gets the confidence grade of the binary classifier ij 
            
            Parameters
            i :  index of the first class

            j:   index of the second class

            Returns
            c_grade : confidence grade.

            """
            
            if not hasattr(self, "c_grades_"):
                raise Exception("Call 'compute_c_grades' with appropriate arguments before using this method.")

            if not hasattr(self, "p_catalog_"):
                # build the catalog
                self.build_p_catalog()

            p = 0
            if (i,j) not in self.p_catalog_:
                if (j,i) not in self.p_catalog_:
                    raise Exception("Invalid indices.")
                else:
                    p = self.p_catalog_[(j,i)]
            else:
                p = self.p_catalog_[(i,j)]

            return self.c_grades_[p]
    

    def compute_s_grade(self, buoyancy, M):
        """Gets the satisfaction grade of the given buoyancy and M values
            
            Parameters
            buoyancy : buoyancy value
            M: M value 

            Returns
            s_grade : satisfaction grade.
        """

        if isinstance(buoyancy,np.ndarray):
            s = 0.5 + 0.5*buoyancy/M
            s = np.where(s<0,0,s)
            s = np.where(s>1,1.,s)
            return s 
        
        s = 0.5 + 0.5*buoyancy/M
        if s<0: s = 0.
        if s>1: s = 1.
        return s 
    
    def build_c_grades_based_weight_dict(self, base_weight = 1.0, decay_rate =0.5):
        """Builds a weight dictionary for each pair of classes based on their computed c-grades. 
           Each weight (i,j) is assigned to the Ki-vs.-Kj classifier.
           Entries for (i,j) and (j,i) will be built. 
        
        Parameters:
            base_weight: Base weight for computing the weight according to the position.  

            decay_rate: Decay rate for computing the weight according to the position. 
                        If decay_rate == 1 equals weights will be assigned to all classifiers.
                        If decay_rate > 1 the lowest weight will be assigned to the classifier with the largest c-grade.
                        If decay_rate < 1 the largest weight will be assigned to the classifier with the largest c-grade.
                  
        Returns:
            weight_dict: a dictionary consisting of the weights for each binary classifier.

        """
  
        n_classes = len(self.classes_)
        confidence_matrix = np.zeros(shape=(n_classes, n_classes))
        for i in range(n_classes-1):
            for j in range(i+1, n_classes):
                confidence_matrix[i,j] = confidence_matrix[j,i] = self.get_c_grade(i,j)
    
        idx_sorted_confidence_matrix = np.argsort(confidence_matrix, axis=1)
        idx_sorted_confidence_matrix = np.flip(idx_sorted_confidence_matrix, axis=1)

        # Computing the weights
        weight_vector = np.zeros(shape=(n_classes-1,))
        for j in range(n_classes-1):
            weight_vector[j] =  base_weight*(decay_rate**(j+1))
        weight_vector = weight_vector/np.sum(weight_vector)

        weight_dict = {}
        for i in range(n_classes):
            for j in range(n_classes-1):
                position = idx_sorted_confidence_matrix[i,j]
                weight_position = weight_vector[j]
                weight_dict[(i,position)] = weight_position

        return weight_dict
    



    def decision_function_with_s_grades(self, X):
        """Evaluates the decision function for each sample x in X.  
        Among each pair of classes A and B, the level to which each object x in X satisfies the proposition 'x IS A' is computed for each x.
         
        
        Parameters
        X :  ndarray of shape (n_samples, n_features)

        Returns
        s_grades : ndarray of shape (nsamples, n_classes*(n_classes-1)/2) consisting of the s-grades for each pair of classes.

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")

        buoyancy = self.decision_function(X)
        s_grades = self.compute_s_grade(buoyancy, self.M_values_.T)
        return s_grades
        



    def decision_function_with_s_grades_and_context(self, X):
        """Evaluates the decision function for each sample x in X.  
        Among each pair of classes A and B, the level to which each object x in X satisfies the proposition 'x IS A' is computed for each x.
        MISVs denoting the context of the evaluations are included. 

        Parameters:
            X:  ndarray of shape (n_samples, n_features)

        Returns:
            (s_grades, membership_misvs, nonmembership_misvs):   
                
                - s_grades: ndarray of shape (nsamples, n_classes*(n_classes-1)/2) consisting of the s-grades for each pair of classes.

                - membership_misvs:   ndarray of shape (nsamples, n_classes*(n_classes-1)/2) consisting of the membership MISVs for each pair of classes.

                - nonmembership_misvs:    ndarray of shape (nsamples, n_classes*(n_classes-1)/2) consisting of the nonmembership MISVs for each pair of classes.

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")
 
        memberships, nonmemberships, membership_misvs, nonmembership_misvs =  self.decision_function_with_context(X)
        buoyancy = memberships - nonmemberships
        s_grades = self.compute_s_grade(buoyancy, self.M_values_.T)
        return s_grades, membership_misvs, nonmembership_misvs
       

    def evaluate_pairwise_l_grades(self, X):
        """Evaluates the l-grade of the proposition 'X IS A' for each pair of classes A and B learned during the training process
        and for each sample in X.
        
        Parameters:
            X:  ndarray of shape (n_samples, n_features);

        Returns:
            arr : ndarray of shape (n_samples, ) consisting of dictionaries having an l-grade per each pair of classes.
        """
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

        n_classes = len(self.classes_)
        n_samples = len(X)
        s_grades, membership_misvs, nonmembership_misvs = self.decision_function_with_s_grades_and_context(X)
 
        evals_per_classifier_per_sample = np.zeros(shape=(n_samples,), dtype=dict)

        if n_classes ==2 :
            for k in range(n_samples):
                p = 0
                evals_per_classifier = {}

                # Due to binary classification
                i = 1
                j = 0
                
                # NB: Assumption that for a binary classifier with classes A and B, the s-grade of 'x is A' plus 
                #     the s-grade of 'x is B' are equal to 1, i.e., s_A = 1 - s_B
                reason_ij = {'reference_class': i, 'other_class': j, 'MISV': membership_misvs[k,p] }
                reason_ji = {'reference_class': j, 'other_class': i, 'MISV': nonmembership_misvs[k,p] }
                # s_grade_ij = LogicAAD(s_grades[k, p], membership_misvs[k,p])
                # s_grade_ji = LogicAAD(1 - s_grades[k, p], nonmembership_misvs[k,p])
                s_grade_ij = LogicAAD(s_grades[k, p], reason_ij)
                s_grade_ji = LogicAAD(1 - s_grades[k, p], reason_ji)

                reason_ij = {'reference_class': i, 'other_class': j  }
                reason_ji = {'reference_class': j, 'other_class': i }
                c_grade_ij = LogicAAD(self.get_c_grade(i,j), reason_ij)
                c_grade_ji = LogicAAD(self.get_c_grade(i,j), reason_ji)

                
                evals_per_classifier[(i,j)] = AugmentedLGrade(s_grade_ij, c_grade_ij)
                evals_per_classifier[(j,i)] = AugmentedLGrade(s_grade_ji, c_grade_ji)

                evals_per_classifier_per_sample[k] = evals_per_classifier
        else:
            for k in range(n_samples):
                p = 0
                evals_per_classifier = {}
                for i in range(n_classes-1):  
                    for j in range(i+1, n_classes):
                        
                        # NB: For a binary classifier with classes A and B, the s-grade of 'x is A' plus 
                        #     the s-grade of 'x is B' are equal to 1, i.e., s_A = 1 - s_B
                        reason_ij = {'reference_class': i, 'other_class': j, 'MISV': membership_misvs[k,p] }
                        reason_ji = {'reference_class': j, 'other_class': i, 'MISV': nonmembership_misvs[k,p] }
                        # s_grade_ij = LogicAAD(s_grades[k, p], membership_misvs[k,p])
                        # s_grade_ji = LogicAAD(1 - s_grades[k, p], nonmembership_misvs[k,p])
                        s_grade_ij = LogicAAD(s_grades[k, p], reason_ij)
                        s_grade_ji = LogicAAD(1 - s_grades[k, p], reason_ji)

                        reason_ij = {'reference_class': i, 'other_class': j  }
                        reason_ji = {'reference_class': j, 'other_class': i }
                        c_grade_ij = LogicAAD(self.get_c_grade(i,j), reason_ij)
                        c_grade_ji = LogicAAD(self.get_c_grade(i,j), reason_ji)

                        evals_per_classifier[(i,j)] = AugmentedLGrade(s_grade_ij, c_grade_ij)
                        evals_per_classifier[(j,i)] = AugmentedLGrade(s_grade_ji, c_grade_ji)

                        p += 1
                evals_per_classifier_per_sample[k] = evals_per_classifier

        return evals_per_classifier_per_sample


    def fit_for_l_grades(self, X_train, y_train, X_val=None, y_val=None, confidence_fn = f1_score):
        """Fits the SVM model according to the training data and computes the M-values to be used for building L-grades.

        
        Parameters:
            X_train: ndarray of shape (n_train_samples, n_features); or
                    ndarray of shape (n_features,) consisting of n features identified for sample X. 
                    Training vectors for fitting the estimator.

            y_train: ndarray of shape (n_train_samples,); 
                    Target values for training vectors.

            X_val: ndarray of shape (n_val_samples, n_features); or
                    ndarray of shape (n_features,) consisting of n features identified for sample X. 
                    Validation vectors for computing confidence grades

            y_val: ndarray of shape (n_train_samples,); 
                    Target values for validation vectors.

        confidence_fn: Metric like sklearn.metrics.f1_score for computing the confidence grades.
        
        Returns:
            self : object
                Fitted estimator.

        """
        
        self.fit(X_train, y_train)

        # compute the maximum absolute values of the margins for each pair of classes using the training set X.
        self.compute_M_values(X_train, y_train)

        if X_val is not None and y_val is not None:
            # compute the confidence grades
            self.compute_c_grades(X_val, y_val, confidence_fn)
            
        return self



    
    
    

    def predict_using_weighted_mean_aggregation(self, X, weight_dict= 'default'):
        """Performs a prediction based on aggregation of L-grades for each sample in X.
        Weighted mean is used for aggregating the L-grades. 

        
        Parameters:
            X: ndarray of shape (n_samples, n_features); or
           ndarray of shape (n_features,) consisting of n features identified for sample X. 

           weight_dict: A weight dictionary in which each entry a key=(i,j) and a value=w_{i,j} that
                        represents the weight of the binary classifier i-vs.-j.
        
        Returns:
            topK: ndarray of shape (n_samples, ) consisting of the class predicted for each sample in X

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        n_classes = len(self.classes_)
        n_samples = X.shape[0]
        n_classifiers = int(n_classes*(n_classes-1)) # This considers an entry for (i,j) and (j,i)

        local_weight_dict = None
        
        if isinstance(weight_dict, (dict)):
            if len(weight_dict) ==  n_classifiers:
                local_weight_dict = weight_dict
            else:
                raise Exception(f"The weight dictionary must have {n_classifiers} weights (one for each binary classifier).")
        else:
            if weight_dict != 'default':
                raise Exception(f"'weight_dict' must be a dictionary with {n_classifiers} weights, or 'default'.")
            else:
                local_weight_dict = self.build_c_grades_based_weight_dict()


        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)
        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=LGrade)

        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=LGrade)
            weight_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=float)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        l_grade = evals_per_classifier[(i,j)].flatten()
                        lgrades_mat[row, col] = l_grade
                        weight_mat[row, col] = local_weight_dict[(i,j)]
                        col += 1
                row +=1

            evals_per_class_per_sample[idx_sample,:] = LGradeAggregator.aggregate_with_weighted_mean(lgrades_mat, weight_mat)

        idx_max = np.argmax(evals_per_class_per_sample, axis=1)
        return self.classes_[idx_max]
    

    def predict_using_weighted_mean_aggregation_with_context(self, X, k = -1, weight_dict= 'default'):
        """Performs a prediction based on aggregation of L-grades for each sample in X.
        Weighted mean is used for aggregating the L-grades. 

        Parameters:
            X: ndarray of shape (n_samples, n_features); or
               ndarray of shape (n_features,) consisting of n features identified for sample X. 
            
            weight_dict: A weight dictionary in which each entry a key=(i,j) and a value=w_{i,j} that
                        represents the weight of the binary classifier i-vs.-j.
        
        Returns:
            topK: ndarray of shape (n_samples, ) consisting of the class predicted for each sample in X

        """

        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")

        cs = self.classes_
        if k<=0:
            k = self.k

        n_classes = len(self.classes_)
        if k> n_classes:
            k = n_classes

        n_samples = X.shape[0]
        n_classifiers = int(n_classes*(n_classes-1)) # This considers an entry for (i,j) and (j,i)

        local_weight_dict = None
        
        if isinstance(weight_dict, (dict)):
            if len(weight_dict) ==  n_classifiers:
                local_weight_dict = weight_dict
            else:
                raise Exception(f"The weight dictionary must have {n_classifiers} weights (one for each binary classifier).")
        else:
            if weight_dict != 'default':
                raise Exception(f"'weight_dict' must be a dictionary with {n_classifiers} weights, or 'default'.")
            else:
                local_weight_dict = self.build_c_grades_based_weight_dict()


        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)
        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=AugmentedLGrade)

        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=AugmentedLGrade)
            weight_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=float)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        augmented_l_grade = evals_per_classifier[(i,j)]
                        lgrades_mat[row, col] =augmented_l_grade
                        weight_mat[row, col] = local_weight_dict[(i,j)]
                        col += 1
                row +=1

            evals_per_class_per_sample[idx_sample,:] = AugmentedLGradeAggregator.aggregate_with_weighted_mean(lgrades_mat, weight_mat)

        idx_topk = np.argsort(evals_per_class_per_sample, axis=1)
        idx_topk = np.flip(idx_topk, axis=1)

        ret = np.full(shape=(n_samples,k),fill_value=None, dtype=AugmentedLPrediction)
        for i in range(n_samples):
            for j in range(k):
                ret[i,j] = AugmentedLPrediction(cs[idx_topk[i,j]], evals_per_class_per_sample[i][idx_topk[i,j]])
    
        return ret
    
    
    def predict_using_disjunction_aggregation(self, X):
        """Performs a prediction based on aggregation of L-grades for each sample in X.
        Disjunction is used for aggregating the L-grades. 
        
        Parameters:
            X: ndarray of shape (n_samples, n_features); or
           ndarray of shape (n_features,) consisting of n features identified for sample X. 
        
        Returns:
            topK: ndarray of shape (n_samples, ) consisting of the class predicted for each sample in X

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        n_classes = len(self.classes_)
        n_samples = len(X)

        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)

        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=LGrade)
        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=LGrade)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        l_grade = evals_per_classifier[(i,j)].flatten()
                        lgrades_mat[row, col] = l_grade
                        col += 1
                row +=1

            evals_per_class_per_sample[idx_sample,:] = LGradeAggregator.aggregate_with_disjunction(lgrades_mat)

        idx_max = np.argmax(evals_per_class_per_sample, axis=1)
        return self.classes_[idx_max]
    

    def predict_using_disjoint_aggregation_with_context(self, X, k=-1):
        """Performs a prediction based on aggregation of L-grades for each sample in X.
        Disjoint is used for aggregating the L-grades. 
        
        Parameters:
            X:  ndarray of shape (n_samples, n_features); or
                ndarray of shape (n_features,) consisting of n features identified for sample X. 
        
        Returns:
            topK: ndarray of shape (n_samples, ) consisting of the class predicted for each sample in X

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if k<=0:
            k = self.k

        n_classes = len(self.classes_)
        if k> n_classes:
            k = n_classes

        n_classes = len(self.classes_)
        n_samples = len(X)
        cs = self.classes_

        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)
       
        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=AugmentedLGrade)
        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=AugmentedLGrade)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        augmented_l_grade = evals_per_classifier[(i,j)]
                        lgrades_mat[row, col] =augmented_l_grade
                        col += 1
                row +=1

            # Although is not necessary for the aggregate_with_conjunction method,  the following line helps with a forthcoming explanation 
            lgrades_mat = np.sort(lgrades_mat, axis=1) 
            
            evals_per_class_per_sample[idx_sample,:] = AugmentedLGradeAggregator.aggregate_with_disjunction(lgrades_mat)

        idx_topk = np.argsort(evals_per_class_per_sample, axis=1)
        idx_topk = np.flip(idx_topk, axis=1)

        ret = np.full(shape=(n_samples,k),fill_value=None, dtype=AugmentedLPrediction)
        for i in range(n_samples):
            for j in range(k):
                ret[i,j] = AugmentedLPrediction(cs[idx_topk[i,j]], evals_per_class_per_sample[i][idx_topk[i,j]])

        return ret


    def predict_using_joint_aggregation(self, X):
        """Performs a prediction based on aggregation of L-grades for each sample in X.
        Joint is used for aggregating the L-grades. 
        
        Parameters:
        X: ndarray of shape (n_samples, n_features); or
           ndarray of shape (n_features,) consisting of n features identified for sample X. 
        
        Returns:
        topK: ndarray of shape (n_samples, ) consisting of the class predicted for each sample in X

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        n_classes = len(self.classes_)
        n_samples = len(X)

        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)

        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=LGrade)
        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=LGrade)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        l_grade = evals_per_classifier[(i,j)].flatten()
                        lgrades_mat[row, col] = l_grade
                        col += 1
                row +=1

            evals_per_class_per_sample[idx_sample,:] = LGradeAggregator.aggregate_with_conjunction(lgrades_mat)

        idx_max = np.argmax(evals_per_class_per_sample, axis=1)
        return self.classes_[idx_max]
    

    def predict_using_joint_aggregation_with_context(self, X, k=-1):
        """Performs a prediction based on aggregation of L-grades for each sample in X.
        Joint is used for aggregating the L-grades. 
        
        Parameters:
        X: ndarray of shape (n_samples, n_features); or
           ndarray of shape (n_features,) consisting of n features identified for sample X. 
        
        Returns:
        topK: ndarray of shape (n_samples, ) consisting of the class predicted for each sample in X

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        if k<=0:
            k = self.k

        n_classes = len(self.classes_)
        if k> n_classes:
            k = n_classes

        n_classes = len(self.classes_)
        n_samples = len(X)
        cs = self.classes_

        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)
       
        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=AugmentedLGrade)
        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=AugmentedLGrade)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        augmented_l_grade = evals_per_classifier[(i,j)]
                        lgrades_mat[row, col] =augmented_l_grade
                        col += 1
                row +=1

            # Although is not necessary for the aggregate_with_conjunction method,  the following line helps with a forthcoming explanation 
            lgrades_mat = np.sort(lgrades_mat, axis=1) 
            
            evals_per_class_per_sample[idx_sample,:] = AugmentedLGradeAggregator.aggregate_with_conjunction(lgrades_mat)

        idx_topk = np.argsort(evals_per_class_per_sample, axis=1)
        idx_topk = np.flip(idx_topk, axis=1)

        ret = np.full(shape=(n_samples,k),fill_value=None, dtype=AugmentedLPrediction)
        for i in range(n_samples):
            for j in range(k):
                ret[i,j] = AugmentedLPrediction(cs[idx_topk[i,j]], evals_per_class_per_sample[i][idx_topk[i,j]])

        return ret
        
    
    def predict_using_owa_aggregation(self, X, weight_vector='default'):
        """Performs a prediction based on aggregation of L-grades for each sample in X.
        Ordered weighted average (OWA) is used for aggregating the L-grades. 

        
        Parameters:
        X: ndarray of shape (n_samples, n_features); or
           ndarray of shape (n_features,) consisting of n features identified for sample X. 
        
        weight_vector: When 'default', the weights are computed through a geometric progression 1/(2^n), n>=1.

        Returns:
        topK: ndarray of shape (n_samples, ) consisting of the class predicted for each sample in X

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        

        n_samples = len(X)
        n_classes = len(self.classes_)
        
        weights = None
        if isinstance(weight_vector, (list, np.ndarray)):
            if len(weight_vector) ==  n_classes - 1:
                weights = weight_vector
            else:
                raise Exception(f"The weight_vector must contain {n_classes-1} weights.")
        else:
            if weight_vector != 'default':
                raise Exception(f"The weight_vector must be an ndarray, a list, or 'default'.")
            else:
                weights = np.array([1/2**i for i in range(n_classes-1, 0, -1)])

        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)
        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=LGrade)

        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=LGrade)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        l_grade = evals_per_classifier[(i,j)].flatten()
                        lgrades_mat[row, col] = l_grade
                        col += 1
                row +=1

            evals_per_class_per_sample[idx_sample,:] = LGradeAggregator.aggregate_with_owa(lgrades_mat, weights)

        idx_max = np.argmax(evals_per_class_per_sample, axis=1)
        return self.classes_[idx_max]
    

    def predict_using_owa_aggregation_with_context(self, X, k = -1, weight_vector='default'):
        """Performs an augmented prediction of the top-K classes for each sample in X using
        ordered weighted average (OWA) aggregation.

        
        Parameters:
        X: ndarray of shape (n_samples, n_features); or
           ndarray of shape (n_features,) consisting of n features identified for sample X. 
        
        weight_vector: When 'default', the weights are computed through a geometric progression 1/(2^n), n>=1.

        k: number of top predictions.

        Returns:
        topK: ndarray of shape (n_samples, k) consisting of the top-K classes predicted for each sample in X; or
              list of the top-K classes predicted for X.

        """
        
        if not hasattr(self, "classes_"):
            raise Exception("This xSVMC instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        if not isinstance(X,np.ndarray):
            raise ValueError("X parameter must be an ndarray")

        cs = self.classes_
        if k<=0:
            k = self.k

        n_classes = len(self.classes_)
        if k> n_classes:
            k = n_classes

        n_samples = X.shape[0]

        weights = None
        
        if isinstance(weight_vector, (list, np.ndarray)):
            if len(weight_vector) ==  n_classes - 1:
                weights = weight_vector
            else:
                raise Exception(f"The weight_vector must contain {n_classes-1} weights.")
        else:
            if weight_vector != 'default':
                raise Exception(f"The weight_vector must be an ndarray, a list, or 'default'.")
            else:
                weights = np.array([1/2**i for i in range(n_classes-1, 0, -1)])


        evals_per_classifier_per_sample = self.evaluate_pairwise_l_grades(X)
        evals_per_class_per_sample = np.zeros(shape=(n_samples,n_classes), dtype=AugmentedLGrade)

        for idx_sample in range(n_samples):
            evals_per_classifier = evals_per_classifier_per_sample[idx_sample]

            lgrades_mat = np.zeros(shape=(n_classes,n_classes-1), dtype=AugmentedLGrade)
            row = 0
            for i in range(n_classes):  
                col = 0
                for j in range(n_classes):
                    if (i,j) in evals_per_classifier:
                        augmented_l_grade = evals_per_classifier[(i,j)]
                        lgrades_mat[row, col] =augmented_l_grade
                        col += 1
                row +=1

            evals_per_class_per_sample[idx_sample,:] = AugmentedLGradeAggregator.aggregate_with_owa(lgrades_mat, weights)

        idx_topk = np.argsort(evals_per_class_per_sample, axis=1)
        idx_topk = np.flip(idx_topk, axis=1)

        ret = np.full(shape=(n_samples,k),fill_value=None, dtype=AugmentedLPrediction)
        for i in range(n_samples):
            for j in range(k):
                ret[i,j] = AugmentedLPrediction(cs[idx_topk[i,j]], evals_per_class_per_sample[i][idx_topk[i,j]])
    
        return ret
    



