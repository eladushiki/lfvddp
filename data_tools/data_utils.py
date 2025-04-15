from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Tuple

from data_tools.detector_efficiency import shapes
import numpy as np
import numpy.typing as npt
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from fractions import Fraction
from random import choices


class DetectorEffect:
    def __init__(
            self,
            efficiency_function: str,
            error_function: str,
        ):
        self._efficiency = self._get_detector_efficiency_filter(efficiency_function)
        self._error = self._get_detector_error_inducer(error_function)

    def _get_detector_efficiency_filter(self, effect_name: Optional[str]) -> Callable[[np.ndarray], np.ndarray]:
        """
        Detector efficiency indicated the probability for each event (=row) to remain.
        """
        if not effect_name:
            return lambda x: np.ones((x.shape[0],))
        
        try:
            return getattr(shapes, effect_name)
        except AttributeError:
            raise ValueError(f"Invalid detector effect requested: {effect_name}")

    def get_detector_efficiency_compensator(self) -> Callable[[np.ndarray], np.ndarray]:
        return lambda x: np.ones((x.shape[0],)) / self._efficiency(x)

    def _get_detector_error_inducer(self, error_name: Optional[str]) -> Callable[[np.ndarray], np.ndarray]:
        """
        Detector error returns the same shape as the input.
        """
        if not error_name:
            return lambda x: x
        
        try:
            return getattr(shapes, error_name)
        except AttributeError:
            raise ValueError(f"Invalid detector error requested: {error_name}")

    def simulate_detector_effect(self, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        data_inclusion = np.random.uniform(size=(events.shape[0],)) < self._efficiency(events)
        _filtered_events = events[data_inclusion]

        errored_events = self._error(_filtered_events)
        
        return errored_events, data_inclusion


class DataSet:  # todo: convert _data to tf.data.Dataset as in https://www.tensorflow.org/api_docs/python/tf/data/Dataset
    """
    A class representing a dataset of events.

    Each row in the stored _data is a single event. The whole 2D table represents the
    collection of them.
    """
    
    def __init__(self, data: npt.NDArray):
        """
        Data has to be a 2D array
        """
        if data.ndim == 1:
            self._data = np.expand_dims(data, axis=1)
        elif data.ndim == 2:
            self._data = data
        else:
            raise ValueError(f"Data must be a 0D, 1D, or 2D array, but got {data.ndim} dimensions.")
        self._weight_mask = np.ones((self._data.shape[0],))

    def __add__(self, other) -> DataSet:
        _data = np.concatenate((self._data, other._data), axis=0)
        _weight_mask = np.concatenate((self._weight_mask, other._weight_mask), axis=0)
        
        result = DataSet(_data)
        result._weight_mask = _weight_mask
        return result

    @property
    def dim(self) -> int:
        if self._data.size == 0:
            return 0
        return self._data.shape[1]

    @property
    def n_samples(self):
        return self._data.shape[0]

    @property
    def histogram_weight_mask(self) -> np.ndarray:
        return np.expand_dims(self._weight_mask, axis=1)
    
    @property
    def corrected_total_weight(self) -> float:
        return float(np.sum(self._weight_mask))

    def __get__(self, item: int) -> np.ndarray:
        """
        Get a single event from the dataset.
        """
        return self._data[item, :] 

    def slice_along_dimension(self, dim: int) -> np.ndarray:
        """
        Get a slice of all events along a single dimension.
        """
        return self._data[:, dim]
    
    def apply_detector_effect(self, detector_effect: DetectorEffect):
        filtered_data, filter = detector_effect.simulate_detector_effect(self._data)

        self._data = filtered_data
        self._weight_mask = self._weight_mask[filter]
        
        # Accumulate compensation in weight for later use
        compensator = detector_effect.get_detector_efficiency_compensator()
        self._weight_mask *= compensator(self._data)


def resample(
        feature: np.ndarray,
        target: np.ndarray,
        background_data_str: str = "Bkg",
        label_method: str = "permute",
        method_type: str = "fixed",
        replacement: bool = True
    ):
    '''
    Creates sets of featureData and featureRef according to featureData_str and featureRef_str respectively.
    
    Parameters
    ----------
    feature: ??
    target: ??
    background_data_str: = "Ref"|"Bkg"
    label_method: = "permute"|"binomial"|"bootstrap" 
        "permute": train_test_split according to current ratio (number of events in each sample remains unchanged)
        "binomial": determine label according to binomial distribution with p_A = N_A/(N_A+N_B) - only total number of events is fixed
        "bootstrap": resample uniformly (with/wo replacement) with sizes N_A and N_B 
    method_type: = "fixed"|"binomial"|"poiss"
    replacement : for bootstrap only - whether events can be repeated. True = repeat events, False - do not repeat events.

    Returns
    ----------
    feature, target
    '''
    combined_data = feature[target[:, 0]==0]
    N_combined = combined_data.shape[0]
    if "Ref" in background_data_str:
        N_B = feature[target[:, 0]==1].shape[0]
        N_A = N_combined-N_B
    else:
        N_A = feature[target[:, 0]==1].shape[0]
        N_B = N_combined-N_A
    
    if method_type=="poiss":
        N_A = np.random.poisson(lam=N_A, size=1)[0]
        N_B = np.random.poisson(lam=N_B, size=1)[0]
    
    if label_method=="permute":
        data_A, data_B = train_test_split(combined_data,train_size=N_A,test_size = N_B,random_state=resample_seed)
    elif label_method =="binomial":
        label = choices(["A","B"],weights = [N_A/(N_A+N_B),N_B/(N_A+N_B)],k = N_combined)
        data_A = combined_data[label=="A"]
        data_B = combined_data[label=="B"]
    elif label_method=="bootstrap":
        # Note: to have no replacement random.choice should actually be replaced by random.sample
        data_A = choices(combined_data, replace = replacement, k=N_A)
        data_B = choices(combined_data, replace = replacement, k=N_B)
    else:
        raise ValueError(f"Invalid label_method for resample: {label_method}")

    if "Ref" in background_data_str:
        featureData = data_B.copy()
        print("Ref")
    else:
        featureData = data_A.copy()
        print("Bkg")

    featureRef = np.concatenate((data_A,data_B),axis=0)
    N_ref = featureRef.shape[0]
    print(N_ref)
    feature     = np.concatenate((featureData, featureRef), axis=0)
    N_R        = N_ref
    N_D        = featureData.shape[0]#N_Bkg

    ## target
    targetData  = np.ones_like(featureData)
    targetRef   = np.zeros_like(featureRef)
    weightsData = np.ones_like(featureData)
    weightsRef  = np.ones_like(featureRef)*N_D*1./N_R
    target      = np.concatenate((targetData, targetRef), axis=0)
    weights     = np.concatenate((weightsData, weightsRef), axis=0)
    target      = np.concatenate((target, weights), axis=1)

    return feature,target


def em(
        background_distribution: np.ndarray,
        signal_distribution: np.ndarray,
        train_size,
        test_size,
        sig_events,
        N_poiss=True,
        combined_portion=1,
        resolution=1,
        *args, **kwargs
    ):
    
    N_Bkg_Pois  = np.random.poisson(lam=float(Fraction(test_size))*background_distribution.shape[0]*combined_portion, size=1)[0] if N_poiss else float(Fraction(test_size))*background_distribution.shape[0]*combined_portion
    N_Ref_Pois  = np.random.poisson(lam=float(Fraction(train_size))*background_distribution.shape[0]*combined_portion, size=1)[0] if N_poiss else float(Fraction(train_size))*background_distribution.shape[0]*combined_portion
    N_Sig_Pois = np.random.poisson(lam=sig_events, size=1)[0] if N_poiss else sig_events
    
    Ref = np.random.choice(background_distribution.reshape(-1,),N_Ref_Pois,replace=True).reshape(-1,1)
    Bkg = np.random.choice(background_distribution.reshape(-1,),N_Bkg_Pois,replace=True).reshape(-1,1)
    total_Sig = np.concatenate(tuple(signal_distribution),axis=0).reshape(-1,)
    Sig = np.random.choice(total_Sig,N_Sig_Pois,replace=True).reshape(-1,1)

    # todo: I deleted here some 1e5 factors that divide all of these in the case of em and others for em_Mcoll
    # Need to go over it when I'll understand where the data comes from know the meaning of it
    return Ref,Bkg,Sig


def generate_pdf(x, size=3e6):
    '''
    Not used (as of 9/23).
    Creates the pdf of the array x.
    Returns new array of length 'size' generated from the pdf of x. 
    '''
    full_dist = x
    print('creating histogram')
    full_dist_hist = np.histogram(x,len(full_dist))
    print('creating pdf')
    full_dist_pdf = sps.rv_histogram(full_dist_hist)
    print('creating new data')
    new_data = full_dist_pdf.rvs(size=size).reshape(-1,1)
    print('we have new data!')
    
    return new_data
