import numpy as np
from pricer import ScalarPricer,BSFormulaPricer
import scipy.stats as stats
import math

class WhaleyPricer(ScalarPricer):
    def __init__(self):
        pass
    
    @staticmethod
    def findSx_Whaley(Sx, K, r, q, v, T, PutCall):
        '''
        Finds the critical stock price, Sx, above/below which it becomes optimal
        to exercise American options.
        '''
        n = 2*(r-q)/v**2
        k = 2*r/v**2/(1-np.exp(-r*T))
    
        c1 = np.log(Sx/K)
        c2 = (r-q+v**2/2)
        c3 = v*np.sqrt(T)
    
        d1 = (c1+c2)/c3
        d2 = d1 - v*np.sqrt(T)
        
        if Sx < 0:
            y = 1e100
        elif PutCall == 'C':
            #res = BSFormulaPricer.EuropeanOption(Sx, K, r, q, v, T, 1)
            cSx = Sx*np.exp(-q*T)*stats.norm.cdf(d1) - K*np.exp(-r*T)*stats.norm.cdf(d2)
            q2 = ( 1-n  + np.sqrt((n-1)**2+4*k) )/2
            #y = (res[0] + (1 - np.exp(-q*T)*stats.norm.cdf( (np.log(Sx/K) + (r-q+v**2/2))/v/np.sqrt(T)) )*Sx/q2 - Sx + K)**2;
            y =  (Sx - K  - cSx - (Sx/q2)*(1 - np.exp(-q*T)*stats.norm.cdf(d1)))**2;
        else:
            res = BSFormulaPricer.EuropeanOption(Sx, K, r, q, v, T, -1)
            q1 = ( 1-n - np.sqrt((n-1)**2+4*k) )/2
            y = (res[0] - (1-np.exp(-q*T)*stats.norm.cdf(-d1))*Sx/q1 + Sx - K)**2
        
        return y
    
    @staticmethod
    def findSxViaNewton_Whaley(Sx, K, b1, b2, c2, denomD1, T, qI, phi):
        '''
        Finds the critical stock price, Sx, above/below which it becomes optimal
        to exercise American options.
        '''
        
        tmp2 = Sx / qI
        c1 = np.log(Sx/K)
        d1 = (c1 + c2*T) / denomD1
        d2 = d1 - denomD1
    
        if (phi == 1):
    
            nd1 = stats.norm.cdf(d1)
            nd2 = stats.norm.cdf(d2)
            premium = Sx*b1*nd1 - K*b2*nd2
            delta = b1*nd1
    
            a1 = (c1 + c2*T)/denomD1
            da1 = 1.0/ (Sx * denomD1)
    
            tmp1 = 1.0-b1*stats.norm.cdf(a1)
            tmp3 = stats.norm.pdf(a1)
    
            gSx    = premium + tmp1*tmp2 - Sx + K
            d1gSx  = delta + tmp1/qI - b1 * tmp3 *da1 * tmp2 - 1
            #d2gSx  = gamma - (b1/qI)*tmp3*(2.0*daI + Sx*(aI*(daI**2)-d2aI))
    
            #d1y = 2*gSx*d1gSx
            #d2y = 2*(d1gSx**2)+2*gSx*d2gSx
    
        elif (phi == -1):
    
            nNd1 = stats.norm.cdf(-d1)
            nNd2 = stats.norm.cdf(-d2)
            premium = K*b2*nNd2 - Sx*b1*nNd1
            delta = -b1*nNd2
    
            a1 = -(c1 + c2*T)/denomD1
            da1 = -1.0/ (Sx * denomD1)
    
            tmp1 = 1.0-b1*stats.norm.cdf(a1)
            tmp3 = stats.norm.pdf(a1)
    
            gSx    = premium - tmp1*tmp2 + Sx - K
            d1gSx  = delta - tmp1/qI + b1 * tmp3 *da1 * tmp2 + 1
            #d2gSx  = gamma + (b1/qI)*tmp3*(2.0*daI + Sx*(d2aI - aI*(daI**2)))
    
            #d1y = 2*gSx*d1gSx
            #d2y = 2*(d1gSx**2)+2*gSx*d2gSx
    
        #Sx = Sx - d1y/d2y
        Sx = Sx - gSx/d1gSx
        return Sx
    
    @staticmethod
    def findGSxWhaley(Sx, K, b1, b2, c2, denomD1, T, qI, phi):
        '''
        Finds the critical stock price, Sx, above/below which it becomes optimal
        to exercise American options.
        '''
        
        tmp2 = Sx / qI
        c1 = math.log(Sx/K)
        d1 = (c1 + c2*T) / denomD1
        d2 = d1 - denomD1

        if (phi == 1):

            nd1 = stats.norm.cdf(d1)
            nd2 = stats.norm.cdf(d2)
            premium = Sx*b1*nd1 - K*b2*nd2
            delta = b1*nd1

            a1 = (c1 + c2*T)/denomD1
            da1 = 1.0/ (Sx * denomD1)

            tmp1 = 1.0-b1*stats.norm.cdf(a1)
            tmp3 = stats.norm.pdf(a1)

            gSx    = premium + tmp1*tmp2 - Sx + K
            d1gSx  = delta + tmp1/qI - b1 * tmp3 *da1 * tmp2 - 1

        elif (phi == -1):

            nNd1 = stats.norm.cdf(-d1)
            nNd2 = stats.norm.cdf(-d2)
            premium = K*b2*nNd2 - Sx*b1*nNd1
            delta = -b1*nNd2

            a1 = -(c1 + c2*T)/denomD1
            da1 = -1.0/ (Sx * denomD1)

            tmp1 = 1.0-b1*stats.norm.cdf(a1)
            tmp3 = stats.norm.pdf(a1)

            gSx    = premium - tmp1*tmp2 + Sx - K
            d1gSx  = delta - tmp1/qI + b1 * tmp3 *da1 * tmp2 + 1

        return gSx, d1gSx

    @staticmethod
    def whaley(S, K, r, q, sig, T, denomD1, discountedRate, discountedDividend, 
           europeanPremium, deltaE, gammaE, vegaE, PutCall, maxIterNewton, tol):
        '''
        Barone-Adesi and Whaley quadratic approximation for American vanilla options
        Finds the American prices as the European prices + premium
        Premium is based on Sx, the critical stock price above or below
        which it becomes optimal to exercise the American options
        S = Spot price
        K = Strike price
        r = Risk free rate
        q = Dividend yield
        sig = Volatility
        T = Maturity
        PutCall = 'C'all or 'P'ut
        '''
        
        c2 = r-q+sig**2/2
    
        n = 2*(r-q)/sig**2
        k = 2*r/sig**2/(1-discountedRate)
    
        dNdSig = -2*n/sig
        dKdSig = -2*k/sig
    
        tmp = (n-1)**2+4*k
    
        q1 = (1-n-np.sqrt(tmp))/2
        dQ1dSig = 0.5*( -dNdSig - 0.5 *(2*(n-1)*dNdSig + 4*dKdSig) / np.sqrt(tmp))
    
        q2 = (1-n+np.sqrt(tmp))/2
        dQ2dSig = 0.5*( -dNdSig + 0.5 *(2*(n-1)*dNdSig + 4*dKdSig) / np.sqrt(tmp))
    
        #d1 = (np.log(S/K)+(r-q+sig**2/2)*T)/(sig*np.sqrt(T))
        #vegaE = S*discountedDividend*np.sqrt(T)*stats.norm.pdf(d1)/100
        #gammaE = discountedDividend*stats.norm.pdf(d1)/(S*denomD1)
    
        europeanStyle = 0
    
        # Quadratic approximation
        if PutCall == 'C':
    
                phi = 1
    
                Sx1 = K
                gSx1, d1gSx1 = WhaleyPricer.findGSxWhaley(Sx1, K, discountedDividend, discountedRate, c2, denomD1, T, q2, phi)
                if (S <= K):
                    Sx2 = 5.0*K
                elif (S > K):
                    Sx2 = 5.0*S
    
                gSx2, d1gSx2 = WhaleyPricer.findGSxWhaley(Sx2, K, discountedDividend, discountedRate, c2, denomD1, T, q2, phi)
    
                if np.isnan(gSx1) or np.isnan(gSx2) or (gSx1*gSx2>=0):
                    europeanStyle = 1
    
                if (europeanStyle == 1):
    
                    AmerPrice = europeanPremium
                    AmerDelta = None
                    AmerVega = vegaE
                    AmerGamma = gammaE
    
                    Sx = np.nan
    
                else:
    
                    if abs(d1gSx2) > abs(d1gSx1):
                        sNew = Sx2
                    else:
                        sNew = Sx1
    
                    sPrevious = sNew
    
                    counter = 0
                    flag = 1
                    while (flag == 1):
                        counter = counter+1
                        sNew = WhaleyPricer.findSxViaNewton_Whaley(sNew, K, discountedDividend, discountedRate, c2, denomD1, T, q2, phi)
                        if abs(sNew - sPrevious)<tol:
                            #print('counter =', num2str(counter), ' Newton-Raphson Converged')
                            flag = 0
                            break
    
                        if ((counter > maxIterNewton) or (sNew < 0)):
                            #print('counter =', num2str(counter), ' break & switch to Bisection Method')
                            break
                            
                        sPrevious = sNew
    
                    counter = 0
                    while (flag == 1):
                        counter = counter+1
                        Sx_m = (Sx1+Sx2)/2
                        gSx_m, _ = WhaleyPricer.findGSxWhaley(Sx_m, K, discountedDividend, discountedRate, c2, denomD1, T, q2, phi)
                        check1 = gSx_m*gSx1
                        if (check1 > 0):
                            Sx1 = Sx_m
                            gSx1 = gSx_m
                        elif (check1 < 0):
                            Sx2 = Sx_m
                            #gSx2 = gSx_m
    
                        if (abs(Sx1-Sx2) < tol):
                            flag = 0
                            sNew = Sx_m
                            #print('counter =', num2str(counter), ' Bisection Method Converged')
                            
                    Sx = sNew
    
                    a2 = (np.log(Sx/K) + c2*T)/denomD1
                    A2 = Sx*(1-discountedDividend*stats.norm.cdf(a2))/q2
    
                    if S<Sx:
    
                        AmerPrice = europeanPremium + A2*(S/Sx)**q2
    
                        d1 = (np.log(Sx/K) + c2*T)/denomD1
                        dD1dSig = np.sqrt(T) - d1/sig
                        dA2dSig = Sx*(-np.exp(-q*T)*stats.norm.pdf(d1)*dD1dSig*q2 - (1-discountedDividend*stats.norm.cdf(d1))*dQ2dSig)/q2**2
                        AmerVega = vegaE + (dA2dSig *(S/Sx)**q2 + A2 *(S/Sx)**q2 * np.log(S/Sx) * dQ2dSig)
    
                        AmerDelta = deltaE + A2*(S/Sx)**q2 * q2/S
                        AmerGamma = gammaE + A2*(S/Sx)**q2 * q2*(q2-1)/S**2
                    else:
    
                        AmerPrice = np.maximum(S - K,0)
                        AmerDelta = 1
                        AmerVega = 0
                        AmerGamma = 0
                        
    
        if PutCall == 'P':
    
                phi = -1
    
                if (S >= K):
                    Sx1 = K/10.0
                elif (S < K):
                    Sx1 = S/10.0
    
                gSx1, d1gSx1 = WhaleyPricer.findGSxWhaley(Sx1, K, discountedDividend, discountedRate, c2, denomD1, T, q1, phi)
    
                Sx2 = K
                gSx2, d1gSx2 = WhaleyPricer.findGSxWhaley(Sx2, K, discountedDividend, discountedRate, c2, denomD1, T, q1, phi)
    
                if np.isnan(gSx1) or np.isnan(gSx2) or (gSx1*gSx2>=0):
                    europeanStyle = 1
    
                if (europeanStyle == 1):
    
                    AmerPrice = europeanPremium
                    AmerDelta = None
                    AmerVega = vegaE
                    AmerGamma = gammaE
    
                    Sx = np.nan
    
                else:
    
                    if abs(d1gSx2) > abs(d1gSx1):
                        sNew = Sx2
                    else:
                        sNew = Sx1
                    
                    sPrevious = sNew
    
                    flag = 1
                    counter = 0
                    while (flag == 1):
                        counter = counter+1
                        sNew = WhaleyPricer.findSxViaNewton_Whaley(sNew, K, discountedDividend, discountedRate, c2, denomD1, T, q1, phi)
                        if abs(sNew - sPrevious)<tol:
                            flag = 0
                            break
                        
                        if ((counter > maxIterNewton) or (sNew < 0)):
                            #print('counter =', num2str(counter), ' break & switch to Bisection Method')
                            break
    
                        sPrevious = sNew
    
                    counter = 0
                    while (flag == 1):
                        counter = counter+1
                        Sx_m = (Sx1+Sx2)/2
                        gSx_m, _ = WhaleyPricer.findGSxWhaley(Sx_m, K, discountedDividend, discountedRate, c2, denomD1, T, q1, phi)
                        check1 = gSx_m*gSx1
                        if (check1 >0):
                            Sx1 = Sx_m
                            gSx1 = gSx_m
                        elif (check1 <0):
                            Sx2 = Sx_m
                            #gSx2 = gSx_m
                        
                        if (abs(Sx1-Sx2) < tol):
                            flag = 0
                            sNew = Sx_m
                            #print('counter =', num2str(counter), ' Bisection Method Converged')
    
                    Sx = sNew
                    a1 = -(np.log(Sx/K) + c2*T)/denomD1
                    A1 = -Sx*(1-discountedDividend*stats.norm.cdf(a1))/q1
    
                    if S>Sx:
    
                        AmerPrice = europeanPremium + A1*(S/Sx)**q1
    
                        #print(AmerPrice, europeanPremium, A1, S, Sx, q1)
                        d1 = (np.log(Sx/K) + c2*T)/denomD1
                        dD1dSig = np.sqrt(T) - d1/sig
                        dA1dSig = -Sx * ( np.exp(-q*T)*stats.norm.pdf(-d1)*dD1dSig*q1 - (1-np.exp(-q*T)*stats.norm.cdf(-d1))*dQ1dSig )/q1**2
                        AmerVega = vegaE + ( dA1dSig *(S/Sx)**q1 + A1 *(S/Sx)**q1 * np.log(S/Sx) * dQ1dSig )
    
                        AmerDelta = deltaE + A1*(S/Sx)**q1 * q1/S
                        AmerGamma = gammaE + A1*(S/Sx)**q1 * q1*(q1-1)/S**2
    
                    else:
    
                        AmerPrice = np.maximum(K - S, 0)
                        AmerDelta = -1
                        AmerVega = 0
                        AmerGamma = 0
    
        #vega2 =  vegaWhaley(phi, K, Sx, S, q, r, sig, T)
    
        return AmerPrice, AmerDelta, AmerGamma, AmerVega, Sx
    
    @staticmethod
    def WhaleyPrice(S, K, r, q, sig, T, phi):
        if (phi == 1):
            callPut = 'C'
        elif (phi == -1):
            callPut = 'P'
    
        maxIterNewton = 5
        tol = 0.001
    
        denomD1 = sig*np.sqrt(T)
        discountedRate = np.exp(-r*T);
        discountedDividend = np.exp(-q*T)
    
        res = BSFormulaPricer.EuropeanOption(S, K, r, q, sig, T, phi)
        europeanP = res[0]
    
        d1 = (np.log(S/K)+(r-q+sig**2/2)*T)/(sig*np.sqrt(T))
    
        if phi == 1:
            deltaE = discountedDividend*stats.norm.cdf(d1)
        elif phi == -1:
            deltaE = -discountedDividend*stats.norm.cdf(-d1)
    
        vegaE = S*discountedDividend*np.sqrt(T)*stats.norm.pdf(d1)
        gammaE = discountedDividend*stats.norm.pdf(d1)/(S*denomD1)
    
        price, delta, gamma, vega, Sx = WhaleyPricer.whaley(S, K, r, q, sig, T, denomD1, discountedRate, discountedDividend, 
                                                 europeanP, deltaE, gammaE, vegaE, callPut, maxIterNewton, tol)
        return price
    
    def _get_price_scalar(self, S, K, r, q, sig, T, phi):
        return WhaleyPricer.WhaleyPrice(S,K,r,q,sig,T,phi)