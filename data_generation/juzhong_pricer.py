import numpy as np
from pricer import ScalarPricer,BSFormulaPricer
import scipy.stats as stats

class JuZhongPricer(ScalarPricer):
    def __init__(self):
        pass
    
    @staticmethod
    def findSx(initialGuess, K, r, q, sigma, T, phi, lambdaH):
        finish = False
        countCycle = 0
        while (not finish):
            d1 = (np.log(initialGuess / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            euroPrice, delta = BSFormulaPricer.EuropeanOption(initialGuess, K, r, q, sigma, T, phi=phi, greekCal=True)[0:2]
            leftSide = phi * initialGuess - lambdaH * (phi * (initialGuess - K))
            rightSide = phi * initialGuess * np.exp(-q*T) * stats.norm.cdf(phi*d1) - euroPrice * lambdaH
            # difference = phi * exp(-q*T) * norm.cdf(phi*d1) + lambdaH * (phi*(initialGuess-K) - euroPrice)/initialGuess - phi
            if abs(leftSide - rightSide) / K < 0.0000001:
                finish = True
            else:
                slopeBi = np.exp(-q*T) * stats.norm.pdf(phi*d1) /(sigma*np.sqrt(T)) + (1 - lambdaH) * delta
                initialGuess = (lambdaH * K * phi + initialGuess * slopeBi - rightSide)/(slopeBi - phi * (1-lambdaH))
                countCycle += 1
        return initialGuess
    
    @staticmethod
    def greeksAnalysis(S, K, r, q, sigma, T, phi, Sx, A, lambdaH, beta, alpha, h, b, c, amerPrice):
        euroPriceS, deltaS, gammaS, thetaS, vegaS = BSFormulaPricer.EuropeanOption(S, K, r, q, sigma, T, phi, greekCal=True)
        if phi * (Sx - S) > 0:
            euroPriceSx, deltaSx, gammaSx, thetaSx, vegaSx = BSFormulaPricer.EuropeanOption(Sx, K, r, q, sigma, T, phi, greekCal=True)
            d1Sx = (np.log(Sx / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            d2Sx = d1Sx - sigma * np.sqrt(T)
            d1S = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            chi = b * (np.log(S/Sx))**2 + c * np.log(S/Sx)
            chiPS = (2 * b / S) * np.log(S/Sx) + c / S
            chiPSS = (2 * b / (S**2)) * (1 - np.log(S/Sx)) - c / (S**2)
            amerDelta = deltaS + (lambdaH/(S*(1-chi)) + chiPS/((1-chi)**2)) * (phi * (Sx - K) - euroPriceSx) * ((S/Sx)**lambdaH)
            amerGamma = np.exp(-q*T) * stats.norm.pdf(phi*d1S) / (S * sigma * np.sqrt(T)) + (2 * lambdaH * chiPS / (S*(1-chi)**2) + 2 * chiPS**2 / ((1-chi)**3) + chiPSS / (1-chi)**2 + (lambdaH**2 - lambdaH)/(S**2 * (1-chi))) * (phi * (Sx - K) - euroPriceSx) * ((S/Sx)**lambdaH)
            amerTheta = r * amerPrice - (sigma*S)**2 * amerGamma / 2 - (r-q) * S * amerDelta
            paramLambda = (beta - 1) ** 2 + 4 * alpha / h
            alphaP = - 2 * alpha / sigma
            betaP = - 2 * beta / sigma
            lambdaP = 0.5 * (-betaP + 0.5 * phi * (2 * (beta - 1) * betaP + 4 * alphaP / h) / np.sqrt(paramLambda))
            lambdaPh = - phi * alpha / (h**2 * np.sqrt(paramLambda))
            lambdaPhP = - (phi / ((h**2) * np.sqrt(paramLambda))) * (alphaP - alpha * (2 * (beta - 1) * betaP + 4 * alphaP / h) / (2 * paramLambda))
            bP = (1-h) * (lambdaPh * alphaP + alpha * lambdaPhP - alpha * lambdaPh * (2 * lambdaP + betaP) / (2 * lambdaH + beta - 1)) / (2 * (2 * lambdaH + beta - 1))
            numeratorForSxP = Sx * np.exp(-q*T) * stats.norm.pdf(d1Sx) * (np.sqrt(T) - d1Sx / sigma) + (phi * (Sx - K) - euroPriceSx) * lambdaP - lambdaH * vegaSx
            denominatorSxP = (1 - lambdaH) * (phi - deltaSx) - np.exp(-q*T) * stats.norm.pdf(d1Sx) / (sigma * np.sqrt(T))
            SxP = numeratorForSxP / denominatorSxP
            d1SxP = SxP / (Sx * sigma * np.sqrt(T)) + (np.sqrt(T) - d1Sx / sigma)
            d2SxP = d1SxP - np.sqrt(T)
            AP = (SxP * (phi - deltaSx) - vegaSx) / h
            thetaP = - np.exp(-q*T) * stats.norm.pdf(d1Sx) * (Sx + sigma * SxP - Sx * sigma * d1Sx * d1SxP) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * stats.norm.pdf(d2Sx) * d2SxP + q * Sx * np.exp(-q*T) * stats.norm.pdf(d1Sx) * d1SxP + q * deltaSx * SxP
            cP = - ((1-h) * alpha / (2 * lambdaH + beta - 1)) * (-(thetaP / A - thetaSx * AP / (A**2)) / (h * r * np.exp(-r*T)) + lambdaPhP / (2 * lambdaH + beta - 1) - lambdaPh * (2 * lambdaP + betaP) / ((2 * lambdaH + beta - 1)**2)) - (1-h) * (-thetaSx / (h * r * np.exp(-r*T) * A) + 1 / h + lambdaPh / (2 * lambdaH + beta - 1)) * (alphaP / (2 * lambdaH + beta - 1) - alpha * (2 * lambdaP + betaP) / ((2 * lambdaH + beta - 1)**2))
            chiP = bP * ((np.log(S/Sx))**2) - (2 * b * (np.log(S/Sx)) + c) * SxP / Sx + cP * (np.log(S/Sx))
            amerVega = vegaS + (h * ((S/Sx)**lambdaH) / (1 - chi)) * (AP + A * (lambdaP * (np.log(S/Sx)) - lambdaH * SxP / Sx)) + h * A * ((S/Sx)**lambdaH) *chiP / ((1 - chi)**2)
            
        else:
            amerDelta = phi
            amerGamma = 0
            amerTheta = r * amerPrice - (sigma * S) ** 2 * amerGamma / 2 - (r - q) * S * amerDelta
            amerVega = 0
        return amerDelta, amerGamma, amerTheta, amerVega

    
    @staticmethod
    def JuZhongPrice(S, K, r, q, sigma, T, phi):
        '''
        American option calculation based on Ju-Zhong paper
        '''
        if phi == 1 and q == 0:
            return BSFormulaPricer.EuropeanOption(S, K, r, q, sigma, T, phi, greekCal = True)
        alpha = 2 * r/(sigma**2)
        beta = 2 * (r-q)/(sigma**2)
        hTau = 1 - np.exp(-r*T)
        lambdaH = (-(beta-1) + phi * np.sqrt((beta-1)**2 + 4 * alpha/hTau))/2
    
        qInfty =  (1 - beta + phi * np.sqrt((beta - 1)**2 + 4*alpha))/2
        sInfty = K/(1 - 1/qInfty)
        hi = (-phi*(r-q)*T - 2*sigma*np.sqrt(T)) * K / (phi * (sInfty - K))
        initialGuess = sInfty + (K - sInfty) * np.exp(hi)
        Sx = JuZhongPricer.findSx(initialGuess, K, r, q, sigma, T, phi, lambdaH)
        ah = (phi * (Sx - K) - BSFormulaPricer.EuropeanOption(Sx, K, r, q, sigma, T, phi=phi)[0])/hTau
    
        theta = BSFormulaPricer.EuropeanOption(Sx, K, r, q, sigma, T, phi=phi, greekCal=True)[-2]
        lambdaHDerivation = -phi * alpha / (hTau**2 * np.sqrt((beta -1)**2 + 4*alpha/hTau))
        b = (1 - hTau) * alpha * lambdaHDerivation/(2*(2 * lambdaH + beta - 1))
        c = - (1 - hTau) * alpha / (2 * lambdaH + beta - 1) * (-theta/(hTau * ah * r * np.exp(-r*T)) + 1/hTau + lambdaHDerivation/(2 * lambdaH + beta - 1))
        euroPrice = BSFormulaPricer.EuropeanOption(S, K, r, q, sigma, T, phi=phi)[0]
        if phi * (Sx - S) > 0:
            amerPrice = euroPrice + (hTau * ah * (S/Sx)**lambdaH)/(1 - b * (np.log(S/Sx))**2 - c * np.log(S/Sx))
        else:
            amerPrice = phi * (S - K)
        amerDelta, amerGamma, amerTheta, amerVega = JuZhongPricer.greeksAnalysis(S, K, r, q, sigma, T, phi, Sx, ah, lambdaH, beta, alpha, hTau, b, c, amerPrice)
        return amerPrice, amerDelta, amerGamma, amerTheta, amerVega, amerPrice - euroPrice
    
    def _get_price_scalar(self,S,K,r,q,sigma,T,phi):
        return JuZhongPricer.JuZhongPrice(S,K,r,q,sigma,T,phi)[0]