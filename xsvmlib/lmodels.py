# ----------------------------------------
#       Using L-grades  
# ----------------------------------------

from xsvmlib.xmodels import AAD
import copy
import numpy as np
import warnings
import numbers


class LogicAAD(AAD):
    """ Logic interpretation of an Augmented Appraisal Degree 
        
        This class is an implementation of a logic interpretation of an augmented appraisal degree (L-AAD), which is a generalization of
        an augmented membership grade (AAD) [1]. An AAD denotes to which level and hints why a (membership) criterion is fulfilled [2].

        References:

        [1] M. Loor, G and De Tré, On the need for augmented appraisal degrees to handle experience-based evaluations,
            Applied Soft Computing, Volume 54, 2017, Pages 284-295, ISSN 1568-4946,
            https://doi.org/10.1016/j.asoc.2017.01.009. 

        [2] L. Zadeh, Fuzzy sets, Inf. Control 8 (3) (1965) 338-353, 
            http://dx.doi.org/10.1016/S0019-9958(65)90241-X.

    """
    def __init__(self, level: float| numbers.Number = 0.0, reason=None):
        delta = 1e-5
        if(level>1.0+delta or level<0.0):
             raise ValueError("'level' parameter must be a value in the unit interval [0,1].")
        self.level = level 
        self.reason = reason 

    def __repr__(self):
        return f"({self.level}, {self.reason})"
    
    def clone(self):
        return copy.deepcopy(self)
    
    def flatten(self):
        return self.level

    @staticmethod
    def min(laad1, laad2):
        """ Gets the minimum between two LogicalAADs.
        """
        return laad1 if laad1.level < laad2.level else laad2
    
    @staticmethod
    def aggregate_reasons(laad1, laad2):
        laad1_reason =  copy.deepcopy(laad1.reason)
        laad2_reason = copy.deepcopy(laad2.reason)
        ret = None
        if(isinstance(laad1_reason, (list))):
            if(isinstance(laad2_reason, (list))):
                laad1_reason.extend(laad2_reason)
            else:
                laad1_reason.append({
                        "before_aggregation_level": laad2.level,
                        "before_aggregation_reason": laad2_reason
                    })
            ret = laad1_reason
        else:
            if(isinstance(laad2_reason, (list))):
                laad2_reason.append({
                        "before_aggregation_level": laad1.level,
                        "before_aggregation_reason": laad1_reason
                    }
                    )
                ret = laad2_reason
            else:
                ret = [
                    {
                        "before_aggregation_level": laad1.level,
                        "before_aggregation_reason": laad1_reason
                    }, 
                    {   
                        "before_aggregation_level": laad2.level,
                        "before_aggregation_reason": laad2_reason
                    }]
        return ret

    def t_norm(laad1, laad2):
        """ 't_norm' implemented using the min function.
        """
        ret = LogicAAD.min(laad1, laad2).clone()
        ret.reason = LogicAAD.aggregate_reasons(laad1, laad2)
        return ret
    
    
    def __and__(self, other):
        """ 'and' implemented using the LogicAAD.t_norm.
        """
        return self.t_norm(other)
        
    def __iand__(self, other):
        """ 'and' implemented using the LogicAAD.t_norm.
        """
        self = self & other
        return self
    
    def max(laad1, laad2):
        """ Gets the maximum between two LogicalAADs.
        """
        return laad1 if laad1.level > laad2.level else laad2
     
    def t_conorm(laad1, laad2):
        """ 't_conorm' implemented using the max function.
        """
        ret = LogicAAD.max(laad1, laad2).clone()
        ret.reason = LogicAAD.aggregate_reasons(laad1, laad2)
        return ret
        
    def __or__(self, other):
        """ 'or' implemented using the LogicAAD.t_conorm function.
        """
        return self.t_conorm(other)
    
    def __ior__(self, other):
        """ 'or' implemented using the max function.
        """
        self = self.t_conorm(other)
        return self
    
    def __mul__(self, weight):
        if(not isinstance(weight, (float, int))):
             raise ValueError("'weight' parameter must be float or int")
        
        laad = self.clone()
        reason = laad.reason
        if reason is not None:
            reason = {"weight": weight,  "unweighted_level": laad.level, "unweighted_reason": reason}
        laad.level *= weight
        laad.reason = reason 
        return laad
    
    def __imul__(self, weight):
        if(not isinstance(weight, (float, int))):
             raise ValueError("'weight' parameter must be float or int") 
        self =  self*weight
        return self
    
    
    def __add__(self, other):
        if(not isinstance(other, (LogicAAD))):
            raise ValueError("'other' parameter must be LogicAAD")
        
        self_reason =  copy.deepcopy(self.reason)
        other_reason = copy.deepcopy(other.reason)
        if(isinstance(self_reason, (list))):
            if(isinstance(other_reason, (list))):
                self_reason.extend(other_reason)
            else:
                self_reason.append({
                        "before_aggregation_level": other.level,
                        "before_aggregation_reason": other_reason
                    })
        else:
            if(isinstance(other_reason, (list))):
                other_reason.append({
                        "before_aggregation_level": self.level,
                        "before_aggregation_reason": self_reason
                    }
                    )
            else:
                self_reason = [
                    {
                        "before_aggregation_level": self.level,
                        "before_aggregation_reason": self_reason
                    }, 
                    {   
                        "before_aggregation_level": other.level,
                        "before_aggregation_reason": other_reason
                    }]
            
        
        return LogicAAD(
            level=self.level+other.level, 
            reason=self_reason
            )
    
    
    def __iadd__(self, other):
        if(not isinstance(other, (LogicAAD))):
            raise ValueError("'other' parameter must be LogicAAD")  
        self = self + other
        return self
        
    def __lt__(self, other):
        if(not isinstance(other, (LogicAAD))):
            raise ValueError("'other' parameter must be an LogicAAD")
        
        return self.level < other.level
    
    def __le__(self, other):
        if(not isinstance(other, (LogicAAD))):
            raise ValueError("'other' parameter must be an LogicAAD")
        
        return self.level <= other.level
    
    def __gt__(self, other):
        if(not isinstance(other, (LogicAAD))):
            raise ValueError("'other' parameter must be an LogicAAD")
        
        return self.level > other.level
    
    def __ge__(self, other):
        if(not isinstance(other, (LogicAAD))):
            raise ValueError("'other' parameter must be an LogicAAD")
        
        return self.level >= other.level
    
class AugmentedLGradeConfidenceWarning(RuntimeWarning):
    pass

class AugmentedLGrade:
    """ This class puts the components of an L-grade [1], namely the satisfaction and
        the confidence grades, in context by means of augmented appraisal degrees (AADs) [2].
        In this case, the reasons behind an evaluation are expressed in such AADs through
        the (index of the) most influential support vector (MISV) [3].

        Parameters:
            s_grade: Satisfaction grade.

            c_grade: Confidence grade. 

            c_min: The minimum threshold value for confidence grades.

        References:

        [1] G. De Tré, M. Peelman, J. Dujmović, Logic reasoning under data veracity concerns,
            International Journal of Approximate Reasoning 161(2023) 108977, 
            https://doi.org/10.1016/j.ijar.2023.108977. 

        [2] M. Loor, G and De Tré, On the need for augmented appraisal degrees to handle experience-based evaluations,
            Applied Soft Computing, Volume 54, 2017, Pages 284-295, ISSN 1568-4946,
            https://doi.org/10.1016/j.asoc.2017.01.009. 

        [3] M. Loor, G. De Tré, Contextualizing support vector machine predictions, 
            International Journal of Computational Intelligence Systems 13 (2020) 1483-1497, 
            https://doi.org/10.2991/ijcis.d.200910.002.

        

    """
    def __init__(self, s_grade: LogicAAD|float|numbers.Number = None, c_grade: LogicAAD|float|numbers.Number = None, c_min: float = 0.8):
        if(isinstance(s_grade, (LogicAAD))):
            self.s_grade = s_grade
        else:
            if (isinstance(s_grade, (float, numbers.Number))):
                self.s_grade = LogicAAD(level=s_grade)
            else:
                self.s_grade = LogicAAD()

        if(isinstance(c_grade, (LogicAAD))):
            self.c_grade = c_grade
        else: 
            if (isinstance(s_grade, (float, numbers.Number))):
                self.c_grade = LogicAAD(level=c_grade)
            else:
                self.c_grade = LogicAAD()
        
        self.c_min = c_min

    def clone(self):
        return copy.deepcopy(self)
    
    def flatten(self):
        return LGrade(self.s_grade.flatten(), self.c_grade.flatten())
   
    def __and__(self, other):
        """ Basic conjunction operator based on the definition presented in [1].

        References:

        [1] G. De Tré, M. Peelman, J. Dujmović, Logic reasoning under data veracity concerns,
            International Journal of Approximate Reasoning 161(2023) 108977, 
            https://doi.org/10.1016/j.ijar.2023.108977. 

        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        s1 = self.s_grade.clone()
        c1 = self.c_grade.clone()

        s2 = other.s_grade.clone()
        c2 = other.c_grade.clone()

        ret_s = s1.t_norm(s2)
        ret_c = None

        if s1.level == 0 or s2.level == 0:
            if s1.level == 0 and s2.level == 0:
                ret_c = c1 if c1.level > c2.level else c2 # max_{s1=0, s2=0} (c1, c2)
            else:
                ret_c = c1 if s1.level == 0 else c2 # max_{s_i=0} (c1, c2)
        else: 
            ret_c = c1 if c1.level < c2.level else c2  # min(c1, c2)

        ret_c.reason = LogicAAD.aggregate_reasons(c1, c2)

        return AugmentedLGrade(ret_s, ret_c)

        
    def __iand__(self, other):
        """ 'and' implemented using the min function.
        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        self = self & other
        return self
    
    
    def __or__(self, other):
        """ Basic disjunction operator based on the definition presented in [1].

            References:

            [1] G. De Tré, M. Peelman, J. Dujmović, Logic reasoning under data veracity concerns,
                International Journal of Approximate Reasoning 161(2023) 108977, 
                https://doi.org/10.1016/j.ijar.2023.108977. 
        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        s1 = self.s_grade.clone()
        c1 = self.c_grade.clone()

        s2 = other.s_grade.clone()
        c2 = other.c_grade.clone()
        
        ret_s = s1.t_conorm(s2)
        ret_c = None

        if s1.level == 1 or s2.level == 1:
            if s1.level == 1 and s2.level == 1:
                ret_c = c1 if c1.level > c2.level else c2 # max_{s1=1, s2=1} (c1, c2)
            else:
                ret_c = c1 if s1.level == 1 else c2 # max_{s_i=1} (c1, c2)
        else: 
            ret_c = c1 if c1.level < c2.level else c2  # min(c1, c2)

        ret_c.reason = LogicAAD.aggregate_reasons(c1, c2)
        return AugmentedLGrade(ret_s, ret_c)
    
    def __ior__(self, other):
        """ 'or' implemented using the max function.
        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        self = self | other
        return self
    
    def __repr__(self):
        return f"(({self.s_grade.level:.3f}, {self.s_grade.reason}), ({self.c_grade.level:.3f}, {self.c_grade.reason})) "
    
    def __str__(self):
        return f"({self.s_grade.level:.3f}, {self.c_grade.level:.3f})"
    
    def __lt__(self, other):
        """ less-than comparison operator based on the definition presented in [1].

            References:

            [1] G. De Tré, M. Peelman, J. Dujmović, Logic reasoning under data veracity concerns,
                International Journal of Approximate Reasoning 161(2023) 108977, 
                https://doi.org/10.1016/j.ijar.2023.108977. 
        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        if self.s_grade.level == other.s_grade.level:
            return self.c_grade.level < other.c_grade.level
        else:
            c_min = max(self.c_min, other.c_min)
            if self.s_grade.level < other.s_grade.level and self.c_grade.level > other.c_grade.level and other.c_grade.level < c_min:
                warnings.warn(f"l1 < l2: while (s1={self.s_grade.level} < s2={other.s_grade.level}) holds in s_grades comparison, the expressions (c1={self.c_grade.level} > c2={other.c_grade.level}) and (c2={other.c_grade.level} < c_min={c_min}) hold in c_grades comparison.", AugmentedLGradeConfidenceWarning)
               
            return self.s_grade.level < other.s_grade.level
    
    def __le__(self, other):
        """ less-than-or-equal comparison operator based on the definition presented in [1].

            References:

            [1] G. De Tré, M. Peelman, J. Dujmović, Logic reasoning under data veracity concerns,
                International Journal of Approximate Reasoning 161(2023) 108977, 
                https://doi.org/10.1016/j.ijar.2023.108977. 7.    
        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        if self.s_grade.level == other.s_grade.level:
            return self.c_grade.level <= other.c_grade.level
        else:
            c_min = max(self.c_min, other.c_min)
            if self.s_grade.level < other.s_grade.level and self.c_grade.level > other.c_grade.level and other.c_grade.level < c_min:
                warnings.warn(f"l1 <= l2: while (s1={self.s_grade.level} < s2={other.s_grade.level}) holds in s_grades comparison, the expressions (c1={self.c_grade.level} > c2={other.c_grade.level}) and (c2={other.c_grade.level} < c_min={c_min}) hold in c_grades comparison.", AugmentedLGradeConfidenceWarning)

            return self.s_grade.level <= other.s_grade.level
    
    def __gt__(self, other):
        """ greater-than comparison operator based on the definition presented in [1].

            References:

            [1] G. De Tré, M. Peelman, J. Dujmović, Logic reasoning under data veracity concerns,
                International Journal of Approximate Reasoning 161(2023) 108977, 
                https://doi.org/10.1016/j.ijar.2023.108977.    
        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        if self.s_grade.level == other.s_grade.level:
            return self.c_grade.level > other.c_grade.level
        else:
            c_min = max(self.c_min, other.c_min)
            if self.s_grade.level > other.s_grade.level and self.c_grade.level < other.c_grade.level and self.c_grade.level < c_min:
                warnings.warn(f"l1 > l2: while (s1={self.s_grade.level} > s2={other.s_grade.level}) holds in s_grades comparison, the expressions (c1={self.c_grade.level} < c2={other.c_grade.level}) and (c1={self.c_grade.level} < c_min={self.c_min}) hold in c_grades comparison.", AugmentedLGradeConfidenceWarning)
        
            return self.s_grade.level > other.s_grade.level
    
    def __ge__(self, other):
        """ greater-than-or-equal comparison operator based on the definition presented in [1].

            References:

            [1] G. De Tré, M. Peelman, J. Dujmović, Logic reasoning under data veracity concerns,
                International Journal of Approximate Reasoning 161(2023) 108977, 
                https://doi.org/10.1016/j.ijar.2023.108977.    
        """
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be an AugmentedLGrade")
        
        if self.s_grade.level == other.s_grade.level:
            return self.c_grade.level >= other.c_grade.level
        else:
            c_min = max(self.c_min, other.c_min)
            if self.s_grade.level > other.s_grade.level and self.c_grade.level < other.c_grade.level and self.c_grade.level < c_min:
                warnings.warn(f"l1 >= l2: while (s1={self.s_grade.level} > s2={other.s_grade.level}) holds in s_grades comparison, the expressions (c1={self.c_grade.level} < c2={other.c_grade.level}) and (c1={self.c_grade.level} < c_min={self.c_min}) hold in c_grades comparison.", AugmentedLGradeConfidenceWarning)
   
            return self.s_grade.level >= other.s_grade.level


    def __mul__(self, weight):
        if(not isinstance(weight, (float, int))):
            raise ValueError("'weight' parameter must be float or int")
        
        augmented_s_grade = self.s_grade*weight
        augmented_c_grade = self.c_grade*weight

        return AugmentedLGrade(augmented_s_grade, augmented_c_grade)
    
    def __imul__(self, weight):
        if(not isinstance(weight, (float, int))):
            raise ValueError("'weight' parameter must be float or int")
        
        self.s_grade = self.s_grade*weight
        self.c_grade = self.c_grade*weight
        return self
    
    def __add__(self, other):
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be AugmentedLGrade")

        augmented_s_grade = self.s_grade + other.s_grade
        augmented_c_grade = self.c_grade + other.c_grade

        return AugmentedLGrade(augmented_s_grade, augmented_c_grade)
    
    def __iadd__(self, other):
        if(not isinstance(other, (AugmentedLGrade))):
            raise ValueError("'other' parameter must be AugmentedLGrade")
        
        self.s_grade = self.s_grade + other.s_grade
        self.c_grade = self.c_grade + other.c_grade
        return self
    


class AugmentedLPrediction: 
    """ L-grade Prediction with the Most Influential Support Vectors
    """
    def __init__(self, class_name, eval, agg_strategy= ""):
        if(not isinstance(eval, (AugmentedLGrade))):
            raise ValueError("eval parameter must be an AugmentedLGrade")
        self.class_name = class_name
        self.eval = eval 



class AugmentedLGradeAggregator:
    """ This class implements several methods for aggregating augmented L-grades.
    """

    @staticmethod
    def aggregate_with_owa(lgrades_mat, weight_vector):
        """ Aggregates the L-grades for each row in lgrades_mat using OWA.

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 

            weight_vector: ndarray of shape(n_lgrades, )
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated augmented L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
            
        if(not isinstance(weight_vector, (np.ndarray))):
            raise ValueError("'weight_vector' parameter must be an ndarray.")
        
        weights = weight_vector
        if len(weights.shape) != 1:
            raise ValueError("'weight_vector'  must be unidimensional.")
            
        if weights.shape[0] != lgrades_mat.shape[1]:
             raise ValueError("Each L-grade in 'lgrades_mat' must be associated with a weight in 'weight_vector'.")
        
        weights_sum = np.sum(weights)
        delta = 1e-5
        if weights_sum>1.+delta or weights_sum<1.-delta:
            raise ValueError(" Sum of the weights must be equal to 1.")

        # In OWA sibling operator, the ranking is based on the satisfaction grades.
        s_grades = np.apply_along_axis(lambda a_1d: [ aad.s_grade.flatten() for aad in a_1d], axis=1, arr=lgrades_mat)
        s_grades = np.reshape(s_grades, lgrades_mat.shape)  #remove extra dimension included by apply_along_axis
                
        idx_s_grades_sorted = np.argsort(s_grades, axis=1)
        # N.B.: descending order is considered in OWA
        idx_s_grades_sorted = np.flip(idx_s_grades_sorted, axis=1)

        lgrades_mat_sorted = np.take_along_axis(lgrades_mat, idx_s_grades_sorted, axis=1)
        weighted_l_grades = lgrades_mat_sorted * weights.T   
        ret = np.sum(weighted_l_grades, axis=1)

        return ret
    

    @staticmethod
    def aggregate_with_weighted_mean(lgrades_mat, weight_mat):
        """ Aggregates the augmented L-grades for each row in lgrades_mat using weighted mean..

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 

            weight_mat: ndarray of shape(n_samples, n_lgrades, )
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
             
        if(not isinstance(weight_mat, (np.ndarray))):
            raise ValueError("'weight_mat' parameter must be an ndarray.")
        
        if weight_mat.shape != lgrades_mat.shape:
             raise ValueError("Each augmented L-grade in 'lgrades_mat' must be associated with a weight in 'weight_mat'.")
        
        weights_sum = np.sum(weight_mat, axis=1)
        delta = 1e-5
        if np.all(weights_sum>1.+delta) or np.all(weights_sum<1.-delta):
            raise ValueError("Sum of the weights in each row must be equal to 1.")

        weighted_l_grades =  np.multiply(lgrades_mat,weight_mat)   
        ret = np.sum(weighted_l_grades, axis=1)

        return ret
    
    @staticmethod
    def aggregate_with_conjunction(lgrades_mat):
        """ Aggregates the L-grades for each row in lgrades_mat using the conjunction operator.

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
    
        n_row = lgrades_mat.shape[0]
        n_lgrades = lgrades_mat.shape[1]
        ret = np.full(shape=(n_row,),fill_value=None, dtype=AugmentedLPrediction)
       
        for i in range(n_row):
            agg_l_grade = None
            for j in range(n_lgrades):
                augmented_l_grade = copy.deepcopy(lgrades_mat[i,j])
                if agg_l_grade is None:
                    agg_l_grade = augmented_l_grade
                else:
                    agg_l_grade &= augmented_l_grade
            ret[i] = agg_l_grade
    

        return ret
    
    @staticmethod
    def aggregate_with_disjunction(lgrades_mat):
        """ Aggregates the L-grades for each row in lgrades_mat using the disjunction operator.

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
    
        n_row = lgrades_mat.shape[0]
        n_lgrades = lgrades_mat.shape[1]
        ret = np.full(shape=(n_row,),fill_value=None, dtype=AugmentedLPrediction)
       
        for i in range(n_row):
            agg_l_grade = None
            for j in range(n_lgrades):
                augmented_l_grade = copy.deepcopy(lgrades_mat[i,j])
                if agg_l_grade is None:
                    agg_l_grade = augmented_l_grade
                else:
                    agg_l_grade |= augmented_l_grade
            ret[i] = agg_l_grade
    

        return ret



class LGrade:
    """ This class implements the logic operations defined for an L-grade.
    """
    def __init__(self, s_grade = None, c_grade = None):
        if(isinstance(s_grade, (float))):
            self.s_grade = s_grade
        else:
            self.s_grade = 0.0

        if(isinstance(c_grade, (float))):
            self.c_grade = c_grade
        else:  
            self.c_grade = 0.0

    def clone(self):
        return LGrade(self.s_grade, self.c_grade )
   
    def __and__(self, other):
        """ Conjunction operator implemented using the min function.
        """
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        s1 = self.s_grade
        c1 = self.c_grade

        s2 = other.s_grade
        c2 = other.c_grade

        ret_s = min(s1, s2)
        ret_c = min(c1, c2)

        if s1 == 0 or s2 == 0:
            ret_c = c1 if s1 == 0 else c2
   
        return LGrade(ret_s, ret_c)

        
    def __iand__(self, other):
        """ 'and' implemented using the min function.
        """
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        self = self & other
        return self
    
    
    def __or__(self, other):
        """ 'or' implemented using the max function.
        """
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        s1 = self.s_grade
        c1 = self.c_grade

        s2 = other.s_grade
        c2 = other.c_grade
        
        ret_s = max(s1, s2)
        ret_c = max(c1, c2)

        if s1 == 1 or s2 == 1:
            ret_c = c1 if s1 == 1 else c2
        else: 
            ret_c = max(c1, c2)

        return LGrade(ret_s, ret_c)
    
    def __ior__(self, other):
        """ 'or' implemented using the max function.
        """
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        self = self | other
        return self
    
    def __repr__(self):
        return f"s-grade: {self.s_grade:.3f}; c-grade: {self.c_grade:.3f} "
    
    def __lt__(self, other):
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        if self.s_grade == other.s_grade:
            return self.c_grade < other.c_grade
        else:
            return self.s_grade < other.s_grade
    
    def __le__(self, other):
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        if self.s_grade == other.s_grade:
            return self.c_grade <= other.c_grade
        else:
            return self.s_grade <= other.s_grade
    
    def __gt__(self, other):
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        if self.s_grade == other.s_grade:
            return self.c_grade > other.c_grade
        else:
            return self.s_grade > other.s_grade
    
    def __ge__(self, other):
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be an LGrade")
        
        if self.s_grade == other.s_grade:
            return self.c_grade >= other.c_grade
        else:
            return self.s_grade >= other.s_grade


    def __mul__(self, weight):
        if(not isinstance(weight, (float, int))):
            raise ValueError("'weight' parameter must be float or int")
        
        augmented_s_grade = self.s_grade*weight
        augmented_c_grade = self.c_grade*weight

        return LGrade(augmented_s_grade, augmented_c_grade)
    
    def __imul__(self, weight):
        if(not isinstance(weight, (float, int))):
            raise ValueError("'weight' parameter must be float or int")
        
        self.s_grade = self.s_grade*weight
        self.c_grade = self.c_grade*weight
        return self
    
    def __add__(self, other):
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be LGrade")

        return LGrade(self.s_grade + other.s_grade, self.c_grade + other.c_grade)
    
    def __iadd__(self, other):
        if(not isinstance(other, (LGrade))):
            raise ValueError("'other' parameter must be LGrade")
        
        self.s_grade = self.s_grade + other.s_grade
        self.c_grade = self.c_grade + other.c_grade
        return self
    

class LGradeAggregator:
    """ This class implements several methods for aggregating L-grades.
    """

    @staticmethod
    def aggregate_with_owa(lgrades_mat, weight_vector):
        """ Aggregates the L-grades for each row in lgrades_mat using OWA.

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 

            weight_vector: ndarray of shape(n_lgrades, )
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
            
        if(not isinstance(weight_vector, (np.ndarray))):
            raise ValueError("'weight_vector' parameter must be an ndarray.")
        
        weights = weight_vector
        if len(weights.shape) != 1:
            raise ValueError("'weight_vector'  must be unidimensional.")
            
        if weights.shape[0] != lgrades_mat.shape[1]:
             raise ValueError("Each L-grade in 'lgrades_mat' must be associated with a weight in 'weight_vector'.")
        
        weights_sum = np.sum(weights)
        delta = 1e-5
        if weights_sum>1.+delta or weights_sum<1.-delta:
            raise ValueError(" Sum of the weights must be equal to 1.")


        # In OWA sibling operator, the ranking is based on the satisfaction grades.
        s_grades = np.apply_along_axis(lambda a_1d: [ aad.s_grade.flatten() for aad in a_1d], axis=1, arr=lgrades_mat)
        s_grades = np.reshape(s_grades, lgrades_mat.shape)  #remove extra dimension included by apply_along_axis
                
        idx_s_grades_sorted = np.argsort(s_grades, axis=1)
        # N.B.: descending order is considered in OWA
        idx_s_grades_sorted = np.flip(idx_s_grades_sorted, axis=1)

        lgrades_mat_sorted = np.take_along_axis(lgrades_mat, idx_s_grades_sorted, axis=1)
        weighted_l_grades = lgrades_mat_sorted * weights.T   
        ret = np.sum(weighted_l_grades, axis=1)

        return ret
    

    @staticmethod
    def aggregate_with_weighted_mean(lgrades_mat, weight_mat):
        """ Aggregates the L-grades for each row in lgrades_mat using weighted mean.

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 

            weight_mat: ndarray of shape(n_samples, n_lgrades)
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
            
        if(not isinstance(weight_mat, (np.ndarray))):
            raise ValueError("'weight_mat' parameter must be an ndarray.")
        
        if weight_mat.shape != lgrades_mat.shape:
             raise ValueError("Each L-grade in 'lgrades_mat' must be associated with a weight in 'weight_mat'.")
        
        weights_sum = np.sum(weight_mat, axis=1)
        delta = 1e-5
        if np.all(weights_sum>1.+delta) or np.all(weights_sum<1.-delta):
            raise ValueError("Sum of the weights in each row must be equal to 1.")

        weighted_l_grades =  np.multiply(lgrades_mat,weight_mat)   
        ret = np.sum(weighted_l_grades, axis=1)

        return ret
    
    @staticmethod
    def aggregate_with_conjunction(lgrades_mat):
        """ Aggregates the L-grades for each row in lgrades_mat using the conjunction operator.

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
    
        n_row = lgrades_mat.shape[0]
        n_lgrades = lgrades_mat.shape[1]
        ret = np.full(shape=(n_row,),fill_value=None, dtype=AugmentedLPrediction)
       
        for i in range(n_row):
            agg_l_grade = None
            for j in range(n_lgrades):
                augmented_l_grade = copy.deepcopy(lgrades_mat[i,j])
                if agg_l_grade is None:
                    agg_l_grade = augmented_l_grade
                else:
                    agg_l_grade &= augmented_l_grade
            ret[i] = agg_l_grade
    

        return ret
    
    @staticmethod
    def aggregate_with_disjunction(lgrades_mat):
        """ Aggregates the L-grades for each row in lgrades_mat using the disjunction operator.

        Parameters:
            lgrades_mat: ndarray of shape (n_samples, n_lgrades); 
           
        Returns:
            ret: ndarray of shape (n_samples, ) consisting of the aggregated L-grades in each row in lgrades_mat

        """

        if(not isinstance(lgrades_mat, (np.ndarray))):
            raise ValueError("'lgrades_mat' parameter must be an ndarray.")
    
        n_row = lgrades_mat.shape[0]
        n_lgrades = lgrades_mat.shape[1]
        ret = np.full(shape=(n_row,),fill_value=None, dtype=AugmentedLPrediction)
       
        for i in range(n_row):
            agg_l_grade = None
            for j in range(n_lgrades):
                augmented_l_grade = copy.deepcopy(lgrades_mat[i,j])
                if agg_l_grade is None:
                    agg_l_grade = augmented_l_grade
                else:
                    agg_l_grade |= augmented_l_grade
            ret[i] = agg_l_grade
    

        return ret