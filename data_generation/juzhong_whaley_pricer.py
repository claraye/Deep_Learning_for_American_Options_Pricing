from pricer import ScalarPricer
from whaley_pricer import WhaleyPricer
from juzhong_pricer import JuZhongPricer

class JuZhongWhaleyPricer(ScalarPricer):
    def __init__(self):
        self.juzhong_pricer = JuZhongPricer()
        self.whaley_pricer = WhaleyPricer()
        
    def _get_price_scalar(self, S, K, r, q, sig, T, phi):
        if T > 0.5:
            return self.juzhong_pricer.JuZhongPrice(S,K,r,q,sig,T,phi)[0]
        else:
            return self.whaley_pricer.WhaleyPrice(S,K,r,q,sig,T,phi)
        