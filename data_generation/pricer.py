from abc import ABC,abstractmethod
import numpy as np
import scipy.stats as stats

class Pricer(ABC):
    @abstractmethod
    def get_price(self,S,K,r,q,sigma,T,phi):
        pass
    
class ScalarPricer(Pricer):
    @abstractmethod
    def _get_price_scalar(self,S,K,r,q,sigma,T,phi):
        pass
    
    def get_price(self,S,K,r,q,sigma,T,phi):
        num_entries = max([param.shape[0] 
            for param in [S,K,r,q,sigma,T,phi]
            if not np.isscalar(param)])
        inputs = np.zeros((num_entries,7))
        for i,param in enumerate([S,K,r,q,sigma,T,phi]):
            inputs[:,i] = param
        return np.apply_along_axis(lambda row:self._get_price_scalar(*row),1,inputs)
        
class BSFormulaPricer(ScalarPricer):
    def __init__(self):
        pass
    
    @staticmethod
    def EuropeanOption(S, K, r, q, sigma, T, phi, greekCal = False):
        '''
        Calculation of Euro option price and greeks
        '''
        delta = None
        gamma = None
        theta = None
        vega = None
    
        top = np.log(S/K) + (r - q + sigma**2/2)*T
        bottom = sigma * np.sqrt(T)
        d1 = top/bottom
        d2 = d1 - sigma * np.sqrt(T)
    
        b1 = np.exp(-q*T)
        b2 = np.exp(-r*T)
    
        if greekCal:
            gamma = b1 * stats.norm.pdf(d1)/(S * bottom)
            vega = b1 * S * stats.norm.pdf(d1) * np.sqrt(T)
    
        if  phi == 1:
            nd1 = stats.norm.cdf(d1)
            nd2 = stats.norm.cdf(d2)
            price = S * b1 * nd1 - K * b2 * nd2
            if greekCal:
                delta = b1 * nd1
                theta = -b1 * S * stats.norm.pdf(d1) * sigma / (2*np.sqrt(T)) - r * K * b2 * nd2 + q * S * b1 * nd1
    
        elif phi == -1:
            nNd1 = stats.norm.cdf(-d1)
            nNd2 = stats.norm.cdf(-d2)
            price = K * b2 * nNd2 - S * b1 * nNd1
            if greekCal:
                delta = -b1 * nNd1
                theta = -b1 * S * stats.norm.pdf(d1) * sigma / (2*np.sqrt(T)) + r * K * b2 * nNd2 - q * S * b1 * nNd1
    
        return price, delta, gamma, theta, vega
    
    def _get_price_scalar(self,S,K,r,q,sigma,T,phi):
        return BSFormulaPricer.EuropeanOption(S,K,r,q,sigma,T,phi)[0] 