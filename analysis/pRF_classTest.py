# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:05:29 2017

@author: marian
"""

import numpy as np


class Fit(object):
    def __init__(self, betaSw, varNumMtnDrctns, varNumVoxChnk):
        # get objects
        self.betaSw = betaSw
        self.varNumMtnDrctns = varNumMtnDrctns
        self.varNumVoxChnk = varNumVoxChnk
        # prepare array for best residuals
        self.vecBstRes = np.zeros(varNumVoxChnk, dtype='float32')
        self.vecBstRes[:] = np.inf
        if type(self.betaSw) is np.ndarray and betaSw.dtype == 'bool':
            self.aryEstimMtnCrvTrn = np.zeros((varNumVoxChnk, varNumMtnDrctns),
                                              dtype='float32')
            self.aryEstimMtnCrvTst = np.zeros((varNumVoxChnk, varNumMtnDrctns),
                                              dtype='float32')
            self.resTrn = np.zeros((varNumVoxChnk), dtype='float32')
            self.resTst = np.zeros((varNumVoxChnk), dtype='float32')
            self.aryErrorTrn = np.zeros((varNumVoxChnk), dtype='float32')
            self.aryErrorTst = np.zeros((varNumVoxChnk), dtype='float32')
            self.contrast = np.eye(varNumMtnDrctns)
            self.denomTrn = np.zeros((varNumVoxChnk, len(self.contrast)),
                                     dtype='float32')
            self.denomTst = np.zeros((varNumVoxChnk, len(self.contrast)),
                                     dtype='float32')
    
        else:
            self.aryEstimMtnCrv = np.zeros((varNumVoxChnk, varNumMtnDrctns),
                                           dtype='float32')

    def fit(self, aryDsgnTmp, aryFuncChnk, lgcTemp=slice(None)):
        aryTmpPrmEst, aryTmpRes = np.linalg.lstsq(
            aryDsgnTmp, aryFuncChnk[:, lgcTemp])[0:2]
        return aryTmpPrmEst.T, aryTmpRes

    def predict(self, aryDsgnTmp, betas, lgcTemp):
        # calculate prediction
        return np.dot(aryDsgnTmp, betas[lgcTemp, :].T)
    
    def tstFit(self, aryDsgnTmp, aryFuncChnk, lgcTemp=slice(None)):
        # get beta weights for axis of motion tuning curves
        betas = self.fit(aryDsgnTmp, aryFuncChnk,
            lgcTemp)[0].T
        # calculate prediction
        aryPredTc = self.predict(self, aryDsgnTmp, betas, lgcTemp)
        # Sum of squares:
        vecBstRes[lgcTemp] = np.sum((aryFuncChnk[:, lgcTemp] -
                                     aryPredTc) ** 2,  axis=0)
                                    

    


