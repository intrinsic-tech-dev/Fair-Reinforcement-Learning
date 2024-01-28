import json, sys, os
from os import path
from time import strftime, gmtime
import csv
import torch
from DataStruct import Obs, Action, CbData, Reward, State

from AgentBase import AgentBase

from collections import OrderedDict

from DataLogger import DataLogger

from Config import Config
from Path import Path

import numpy as np

import cvxpy as cp

#import qcqp 

import math

import gurobipy as gp 
from gurobipy import GRB

from Constant import * 

class GroupAgentTrain(object):
    def __init__(self):

        self.numScores = None
        self.minScore = None

        self.rewardEst = None

        self.scoreIsQualScoreLag1IsQualLag1DecLag1Freq = None
        self.rewardMultScoreIsQualDecFreq = None
        self.groupFreq = None

        self.estTransProb = None
        self.estReward = None

        self.scoreIndex = None
        self.decision = None
        self.isQual = None       
        
        self.occuMeasNpForUpdate = None
        self.occuMeasCvx = None

        self.status = None

    def init(self, groupIndex, evalType, numScores, minScore, optRoundEps, numNnTrainSteps, status):

        self.groupIndex = groupIndex
        self.evalType = evalType

        self.numScores = numScores
        self.minScore = minScore
        self.optRoundEps = optRoundEps
        self.numNnTrainSteps = numNnTrainSteps
        self.initScoreIsQualFreq = np.ones(shape = (self.numScores, 2)) * optRoundEps
        self.scoreIsQualScoreLag1IsQualLag1DecLag1Freq = np.ones(shape = (self.numScores, 2, self.numScores, 2, 2)) * optRoundEps
        self.rewardMultScoreIsQualDecFreq = np.zeros(shape = (self.numScores, 2, 2))
        self.groupFreq = 0

        self.estInitProb = np.ones(shape = (self.numScores, 2)) * optRoundEps
        self.estTransProb = np.ones(shape = (self.numScores, 2, self.numScores, 2, 2)) * optRoundEps
        self.estReward = np.ndarray(shape = (self.numScores, 2, 2))

        self.occuMeasNpForUpdate = np.ndarray(shape = (self.numScores, 2, 2, self.numNnTrainSteps), dtype = float)
        for timeStep in range(self.numNnTrainSteps):
            occuMeasTmNpUnNrm = np.random.rand(self.numScores, 2, 2) 
            occuMeasTmNpNrm = occuMeasTmNpUnNrm / np.sum(occuMeasTmNpUnNrm) 
            self.occuMeasNpForUpdate[:,:,:,timeStep] = occuMeasTmNpNrm

        self.occuMeasNpForUpdateEval = np.ndarray(shape = (self.numScores, 2, 2, self.numNnTrainSteps), dtype = float)
        for timeStep in range(self.numNnTrainSteps):
            occuMeasTmNpUnNrm = np.random.rand(self.numScores, 2, 2) 
            occuMeasTmNpNrm = occuMeasTmNpUnNrm / np.sum(occuMeasTmNpUnNrm) 
            self.occuMeasNpForUpdateEval[:,:,:,timeStep] = occuMeasTmNpNrm

        self.occuMeasNpForEval = np.ndarray(shape = (self.numScores, 2, 2, self.numNnTrainSteps), dtype = float)
        for timeStep in range(self.numNnTrainSteps):
            occuMeasTmNpUnNrm = np.random.rand(self.numScores, 2, 2) 
            occuMeasTmNpNrm = occuMeasTmNpUnNrm / np.sum(occuMeasTmNpUnNrm) 
            self.occuMeasNpForEval[:,:,:,timeStep] = occuMeasTmNpNrm

        self.occuMeasCvx = []
        for score in range(numScores):
            occuMeasScore = []
            for isQual in range(2):
                occuMeasScore.append(cp.Variable(shape = (2, self.numNnTrainSteps)))
            self.occuMeasCvx.append(occuMeasScore)

        self.status = status
    def comp_policy(self, scoreIndex, nnTrainStepIndex):

        occuMeas = self.occuMeasNpForUpdateEval[scoreIndex,:,:,nnTrainStepIndex]


        dec = np.sum(occuMeas, axis = 0)#.reshape(2) 
        policyProb = dec * 1.0 / np.sum(dec) #np.sum(probScoreDec, axis = 0)

        return policyProb

    def update_initScoreIsQual_freq(self, scoreIndex, isQual):
        self.initScoreIsQualFreq[scoreIndex][isQual] += 1

    #update stats
    def update_scoreIsQualScoreLag1IsQualLag1DecLag1_freq(self, scoreIndex, isQual):
        if self.scoreIndex != None and self.decision != None:
            self.scoreIsQualScoreLag1IsQualLag1DecLag1Freq[scoreIndex][isQual][self.scoreIndex][self.isQual][self.decision] += 1

    #update stats
    def update_rewardMultScoreIsQualDec_freq(self, reward, scoreIndex, isQual, dec):
        self.rewardMultScoreIsQualDecFreq[scoreIndex][isQual][dec] += reward

    def update_group_freq(self):
        self.groupFreq += 1

    def update_estInitProb(self):
        for scoreIndex in range(self.numScores):
            for isQualIndex in range(2):
                self.estInitProb[scoreIndex][isQualIndex] = self.initScoreIsQualFreq[scoreIndex][isQualIndex] * 1.0 / np.sum(self.initScoreIsQualFreq)

    def update_estTransProb(self):
        for scoreIndex in range(self.numScores):
            for isQualIndex in range(2):
                for scoreLag1Index in range(self.numScores):
                    for isQualLag1Index in range(2):
                        for decLag1Index in range(2):
                            self.estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index] = self.scoreIsQualScoreLag1IsQualLag1DecLag1Freq[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index] * 1.0 / np.sum(self.scoreIsQualScoreLag1IsQualLag1DecLag1Freq, axis = (0,1))[scoreLag1Index][isQualLag1Index][decLag1Index]  #max(self.scoreIsQualDecFreq[scoreLag1Index][isQualLag1Index][decLag1Index], 1)

    def update_estReward(self):
        for scoreIndex in range(self.numScores):
            for isQualIndex in range(2):
                for decIndex in range(2):
                    self.estReward[scoreIndex][isQualIndex][decIndex] = self.rewardMultScoreIsQualDecFreq[scoreIndex][isQualIndex][decIndex] * 1.0 / np.sum(self.scoreIsQualScoreLag1IsQualLag1DecLag1Freq, axis = (0,1))[scoreIndex][isQualIndex][decIndex]#max(self.scoreIsQualDecFreq[scoreIndex][isQualIndex][decIndex], 1)# + bonus[[self.score2ScoreIndex(score)][isQual][dec]

    #update sample
    def update_scoreDec(self, scoreIndex, decision):
        self.scoreIndex = scoreIndex
        self.decision = decision

    #update sample
    def update_isQual(self, isQual):
        self.isQual = isQual

class AgentTrain(AgentBase):#AgentMbLnrFwdGatFwdLnrBwdGatBwd
    def __init__(self):
        self.path = None
        self.useConfigMode = None
        self.device = None
        self.nnModel = None

        self.nnInputDim = None
        self.nnHiddenDim = None
        self.nnTargetDim = None

        self.gamma = None

        self.nnLr = None

        self.logStartTrainEpisodeIndex = None

        self.objValEval = None
        self.objValUpdate = None

        self.status = None

    def read_config(self, identifier, rootDir, exprimentHyperConfigDir, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, curLocalTimeInMicro, device, numTrainEpisode, numInterPolicyUpdateEpisodes, numInterPolicyUpdateEvalEpisodes, numInterPolicyEvalEpisodes, numNnTrainSteps, isTrainPhase, evalType, envType, groupProbList, trialTrainIndex, useTrueModel):#nnBatchSize, nnInputDim, nnHiddenDim, nnTargetDim, nnLayerDim):#, nStates, nActions):
        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootDir, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        self.exprimentHyperConfigDir = exprimentHyperConfigDir
        self.curLocalTimeInMicro = curLocalTimeInMicro

        if self.curLocalTimeInMicro != None:
            with open(self.exprimentHyperConfigDir + self.__class__.__name__ + "Hyper" + str(self.curLocalTimeInMicro) + ".json") as json_file:
                config = json.load(json_file)
                self.algoType = config["algoType"]
                self.methodType = config["methodType"]
                self.methodConst = config["methodConst"]
                self.paretoType = config["paretoType"]

        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)

            self.numScores = config["numScores"]
            self.minScore = config["minScore"]
            self.numGroups = config["numGroups"]
            self.mypDec0Prob = config["mypDec0Prob"]

            self.timeLimitDpOrig = config["timeLimitDpOrig"]
            self.timeLimitDpLag = config["timeLimitDpLag"]
            self.timeLimitEqOptOrig = config["timeLimitEqOptOrig"]
            self.timeLimitEqOptLag = config["timeLimitEqOptLag"]
            self.timeLimitAggOrig = config["timeLimitAggOrig"]
            self.timeLimitAggLag = config["timeLimitAggLag"]
            self.timeLimitDpCvOrig = config["timeLimitDpCvOrig"]
            self.timeLimitDpCvLag = config["timeLimitDpCvLag"]
            self.timeLimitEqOptCvOrig = config["timeLimitEqOptCvOrig"]
            self.timeLimitEqOptCvLag = config["timeLimitEqOptCvLag"]

            self.feasEps = config["feasEps"]
            self.optValEps = config["optValEps"]
            self.optRoundEps = config["optRoundEps"]
            self.optRoundEpsTransProb = config["optRoundEpsTransProb"]
            self.mipGap = config["mipGap"]

            self.algoLoggerCapacity = config["algoLoggerCapacity"]
            self.optObjConstrForUpdateLoggerFileName = config["optObjConstrForUpdateLoggerFileName"]           

            self.hasTransProbConstr = config["hasTransProbConstr"]
            self.isComputeEval = config["isComputeEval"]
            self.numInterPolicyUpdateEpisodes = numInterPolicyUpdateEpisodes
            self.numInterPolicyUpdateEvalEpisodes = numInterPolicyUpdateEvalEpisodes
            self.numInterPolicyEvalEpisodes = numInterPolicyEvalEpisodes
            
            self.envType = envType
            self.evalType = evalType

            self.device = device

            self.identifier = identifier

            self.numTrainEpisode = numTrainEpisode
            self.numNnTrainSteps = numNnTrainSteps

            self.isTrainPhase = isTrainPhase

            self.groupProbList = groupProbList

            self.trialTrainIndex = trialTrainIndex

            self.useTrueModel = useTrueModel

            self.evalOptIndex = 0
            self.updateOptIndex = 0

            self.status = AGENT_UPDATE
            self.groupList = []
            for groupIndex in range(self.numGroups):
                group = GroupAgentTrain()
                group.init(groupIndex, self.evalType, self.numScores, self.minScore, self.optRoundEps, self.numNnTrainSteps, self.status)
                self.groupList.append(group)

            numPolicyUpdates = int(self.numTrainEpisode / (self.numInterPolicyUpdateEpisodes + self.numInterPolicyUpdateEvalEpisodes))
            if self.methodType == "orig":
                self.estConstrForUpdate = [self.methodConst for policyUpdateIndex in range(numPolicyUpdates)]
            self.algoEvalLogDictDict = OrderedDict()
            self.algoUpdateLogDictDict = OrderedDict()

            optObjConstrForUpdateHeader = ["evalTypeEnvTypeParetoType", "paretoType", "methodName", "methodConst", "methodSampleSize", "curLocalTimeInMicro", "trialTrainIndex", "trainEpisodeIndex", "optObjVal", "optConstrVal", "optGap"]

            self.optObjConstrForUpdateLogger = DataLogger(self.algoLoggerCapacity, self.optDataCurDir + self.optObjConstrForUpdateLoggerFileName, optObjConstrForUpdateHeader) 

            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")

            self.algoDir = self.optDataCurDir + "algoFromTrain/"
            if not os.path.exists(self.algoDir):
                os.makedirs(self.algoDir)

    def set_initProbList_transProbList_rewardList(self, initProbList, transProbList, rewardList):
        self.initProbList = initProbList
        self.transProbList = transProbList
        self.rewardList = rewardList

    def set_methodName(self, methodName):
        self.methodName = methodName

    def on_cbData(self, cbData, trainEpisodeIndex, nnTrainStepIndex):
        obs = self.preprocess(cbData, trainEpisodeIndex, nnTrainStepIndex)
        action = self.select_action(obs, trainEpisodeIndex, nnTrainStepIndex)
        self.postprocess(obs, action, trainEpisodeIndex, nnTrainStepIndex)


        return action


    def preprocess(self, cbData, trainEpisodeIndex, nnTrainStepIndex):#, gateActNet, stepSizeScaleActNet, state):

        obs = Obs()
        obs.scoreIndexList = cbData.scoreIndexList

        return obs

    def select_action(self, obs, trainEpisodeIndex, nnTrainStepIndex):
        policyProbList = []
        for groupIndex in range(self.numGroups):
            policyProb = self.groupList[groupIndex].comp_policy(obs.scoreIndexList[groupIndex], nnTrainStepIndex)
            policyProbList.append(policyProb)
        action = Action()
        action.decisionList = []
        for groupIndex in range(self.numGroups):
            decision = np.argmax(np.random.multinomial(1, policyProbList[groupIndex]))
            action.decisionList.append(decision)
        return action#.detach()

    def postprocess(self, obs, action, trainEpisodeIndex, nnTrainStepIndex):
        pass
    def on_state_action_reward(self, state, action, reward, trainEpisodeIndex, nnTrainStepIndex):
        if self.status == AGENT_UPDATE:
            for groupIndex in range(self.numGroups):
                if nnTrainStepIndex == 0:
                    self.groupList[groupIndex].update_initScoreIsQual_freq(state.scoreIndexList[groupIndex], state.isQualList[groupIndex])
                self.groupList[groupIndex].update_scoreIsQualScoreLag1IsQualLag1DecLag1_freq(state.scoreIndexList[groupIndex], state.isQualList[groupIndex])
                self.groupList[groupIndex].update_rewardMultScoreIsQualDec_freq(reward.rewardValList[groupIndex], state.scoreIndexList[groupIndex], state.isQualList[groupIndex], action.decisionList[groupIndex])
                self.groupList[groupIndex].update_scoreDec(state.scoreIndexList[groupIndex], action.decisionList[groupIndex])
                self.groupList[groupIndex].update_isQual(state.isQualList[groupIndex])
        
        if (trainEpisodeIndex + 1) % (self.numInterPolicyUpdateEpisodes + self.numInterPolicyUpdateEvalEpisodes) == self.numInterPolicyUpdateEpisodes and (nnTrainStepIndex+1) % self.numNnTrainSteps == 0:
            policyUpdateIndex = int((trainEpisodeIndex + 1) / (self.numInterPolicyUpdateEpisodes + self.numInterPolicyUpdateEvalEpisodes))
            self.compute_occuMeas_update(trainEpisodeIndex, policyUpdateIndex)   
            print("the index of the optimization for updateOpt: ", self.updateOptIndex)
            self.updateOptIndex += 1

    def compute_occuMeas_eval(self, trainEpisodeIndex, policyEvalIndex):#, trainEpisodeIndex, nnTrainStepIndex):
        
        #2. get estimates for both groups
        for groupIndex in range(self.numGroups):
            if self.useTrueModel:
                self.groupList[groupIndex].estInitProb = self.initProbList[groupIndex]
                self.groupList[groupIndex].estTransProb = self.transProbList[groupIndex]
                self.groupList[groupIndex].estReward = self.rewardList[groupIndex]
            else:
                self.groupList[groupIndex].update_estInitProb()
                self.groupList[groupIndex].update_estTransProb()
                self.groupList[groupIndex].update_estReward()

        if self.algoType == "dpOrigCvx":
            self.solve_dpOrigCvx_opt_eval(trainEpisodeIndex, policyEvalIndex)#estGroupProb, estDpConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr)
        elif self.algoType == "dpLagCvx":
            self.solve_dpLagCvx_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "dpOrigGp":
            self.solve_dpOrigGp_opt_eval(trainEpisodeIndex, policyEvalIndex)#estGroupProb, estDpConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr)
        elif self.algoType == "dpLagGp":
            self.solve_dpLagGp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "eqOptOrigGp":
            self.solve_eqOptOrigGp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "eqOptLagGp":#eqOptLag
            self.solve_eqOptLagGp_opt_eval(trainEpisodeIndex, policyEvalIndex)#estGroupProb, estEqOptConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr)
        elif self.algoType == "eqOptSdp":
            self.solve_eqOptSdp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "aggOrigGp":
            self.solve_aggOrigGp_opt_eval(trainEpisodeIndex, policyEvalIndex)
        elif self.algoType == "aggLagGp":
            self.solve_aggLagGp_opt_eval(trainEpisodeIndex, policyEvalIndex)

    def compute_occuMeas_update(self, trainEpisodeIndex, policyUpdateIndex):#, trainEpisodeIndex, nnTrainStepIndex):
        
        #2. get estimates for both groups
        for groupIndex in range(self.numGroups):
            if self.useTrueModel:
                self.groupList[groupIndex].estInitProb = self.initProbList[groupIndex]
                self.groupList[groupIndex].estTransProb = self.transProbList[groupIndex]
                self.groupList[groupIndex].estReward = self.rewardList[groupIndex]
            else:
                self.groupList[groupIndex].update_estInitProb()
                self.groupList[groupIndex].update_estTransProb()
                self.groupList[groupIndex].update_estReward()

        if self.algoType == "dpOrigCvx":
            self.solve_dpOrigCvx_opt_update(trainEpisodeIndex, policyUpdateIndex)#estGroupProb, estDpConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr)
        elif self.algoType == "dpLagCvx":
            self.solve_dpLagCvx_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "dpOrigGp":
            self.solve_dpOrigGp_opt_update(trainEpisodeIndex, policyUpdateIndex)#estGroupProb, estDpConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr)
        elif self.algoType == "dpLagGp":
            self.solve_dpLagGp_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "eqOptOrigGp":
            self.solve_eqOptOrigGp_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "eqOptLagGp":#eqOptLag
            self.solve_eqOptLagGp_opt_update(trainEpisodeIndex, policyUpdateIndex)#estGroupProb, estEqOptConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr)
        elif self.algoType == "eqOptSdp":
            self.solve_eqOptSdp_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "aggOrigGp":
            self.solve_aggOrigGp_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "aggLagGp":
            self.solve_aggLagGp_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "dpCvOrigGp":
            self.solve_dpCvOrigGp_opt_update(trainEpisodeIndex, policyUpdateIndex)#estGroupProb, estDpConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr)
        elif self.algoType == "dpCvLagGp":
            self.solve_dpCvLagGp_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "eqOptCvOrigGp":
            self.solve_eqOptCvOrigGp_opt_update(trainEpisodeIndex, policyUpdateIndex)
        elif self.algoType == "eqOptCvLagGp":#eqOptLag
            self.solve_eqOptCvLagGp_opt_update(trainEpisodeIndex, policyUpdateIndex)#estGroupProb, estEqOptConstr)#, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr)
        elif self.algoType == "myp":
            self.solve_myp_opt_update(trainEpisodeIndex, policyUpdateIndex)

    def solve_dpOrigCvx_opt(self, trainEpisodeIndex, policyEvalIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1
        #objective
        objSum = 0
        for groupIndex in range(self.numGroups):
            expr = 0
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):
                            expr = expr + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
            objSum = objSum + self.groupProbList[groupIndex] * expr 
        objective = cp.Maximize(objSum)

        #constraints
        constraintList = []
        #fairness constr
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                expr = 0
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        expr = expr + self.groupList[firstGroupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep]
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        expr = expr - self.groupList[secondGroupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep]
                constraintList.append((expr <= self.estConstrForEval[policyEvalIndex]))
                constraintList.append((expr >= - self.estConstrForEval[policyEvalIndex]))

        #trans dyn constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                        exprLeft = 0
                        for decLag1Index in range(2):
                            exprLeft = exprLeft + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decLag1Index][timeStep]
                        exprRight = 0
                        for scoreLag1Index in range(self.numScores):
                            for isQualLag1Index in range(2):
                                for decLag1Index in range(2):
                                    exprRight = exprRight + self.groupList[groupIndex].occuMeasCvx[scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                        constraintList.append((exprLeft == exprRight))#self.optRoundEps))

        #prob box constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                            constraintList.append((self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] >= 0))## - self.optRoundEps))
                            constraintList.append((self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] <= 1))## + self.optRoundEps))
                            

        #prob sum 1 constr
        for groupIndex in range(self.numGroups):
            for timeStep in range(self.numNnTrainSteps):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            expr = expr + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep]
                constraintList.append((expr <= 1 + self.optRoundEps))
                constraintList.append((expr >= 1 - self.optRoundEps))
       
        #problem
        problem = cp.Problem(objective, constraintList)
        
        problem.solve(verbose = True)

        print("status: ", problem.status)
        print("optimal value: ", problem.value)

        if problem.status == "optimal":
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep].value
                                if value < self.optRoundEps:
                                    self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] = self.optRoundEps
                                else:
                                    self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] = value

        else:
            print("algoType, methodConst, envType: ", self.algoType, self.methodConst, self.envType)
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
#                print(self.optDataCurDir + "infeasibilityErrorCode.csv")
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)
            

    def solve_dpLagCvx_opt(self, trainEpisodeIndex, policyEvalIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1
        #objective
        objSum = 0
        for groupIndex in range(self.numGroups):
            expr = 0
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):
                            expr = expr + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
            objSum = objSum + self.groupProbList[groupIndex] * expr 
        objective = cp.Maximize(objSum)

        #constraints
        constraintList = []
        #fairness constr
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                expr = 0
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        expr = expr + self.groupList[firstGroupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep]
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        expr = expr - self.groupList[secondGroupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep]

                objSum = objSum - self.methodConst * (self.estConstrForEval[policyEvalIndex] - expr) + self.methodConst * (self.estConstrForEval[policyEvalIndex] + expr)

        #trans dyn constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                        exprLeft = 0
                        for decLag1Index in range(2):
                            exprLeft = exprLeft + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decLag1Index][timeStep]
                        exprRight = 0
                        for scoreLag1Index in range(self.numScores):
                            for isQualLag1Index in range(2):
                                for decLag1Index in range(2):
                                    exprRight = exprRight + self.groupList[groupIndex].occuMeasCvx[scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                        constraintList.append((exprLeft == exprRight))#self.optRoundEps))

        #prob box constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                            constraintList.append((self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] >= 0))## - self.optRoundEps))
                            constraintList.append((self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] <= 1))## + self.optRoundEps))
                            

        #prob sum 1 constr
        for groupIndex in range(self.numGroups):
            for timeStep in range(self.numNnTrainSteps):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            expr = expr + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep]
                constraintList.append((expr <= 1 + self.optRoundEps))
                constraintList.append((expr >= 1 - self.optRoundEps))
       
        #problem
        problem = cp.Problem(objective, constraintList)
        
        problem.solve(verbose = True)

        print("status: ", problem.status)
        print("optimal value: ", problem.value)

        if problem.status == "optimal":
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep].value
                                if value < self.optRoundEps:
                                    self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] = self.optRoundEps
                                else:
                                    self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] = value
        else:
            print("algoType, methodConst, methodConst, envType: ", self.algoType, self.methodConst, self.methodConst, self.envType)
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
#                print(self.optDataCurDir + "infeasibilityErrorCode.csv")
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)
    
    def solve_dpOrigGp_opt_eval(self, trainEpisodeIndex, policyEvalIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr 
            qcqp.setObjective(objSum, GRB.MAXIMIZE)
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr + occuMeas[firstGroupIndex][scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr - occuMeas[secondGroupIndex][scoreIndex][isQualIndex][1][timeStep]
                qcqp.addConstr(expr <= self.estConstrForEval[policyEvalIndex])
                qcqp.addConstr(expr >= - self.estConstrForEval[policyEvalIndex])

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            if self.hasTransProbConstr:
            #trans dyn constr
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                                exprLeft = 0
                                for decLag1Index in range(2):
                                    exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                                exprRight = 0
                                for scoreLag1Index in range(self.numScores):
                                    for isQualLag1Index in range(2):
                                        for decLag1Index in range(2):
                                            exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                                qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                                qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)
            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)
            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitDpOrig)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status

        if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
            count = 0
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = sol[count].X
                                count += 1
                                self.groupList[groupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value
            self.algoEvalLogDictDict[trainEpisodeIndex] = OrderedDict()
            for groupIndex in range(self.numGroups):
                self.algoEvalLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForEval

            obj = qcqp.getObjective()
            objValEval = obj.getValue()
            
            avgFairConstrVal = 0
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][1][timeStep]
                avgFairConstrVal += abs(fairConstrVal)
            avgFairConstrVal /= self.numNnTrainSteps
        else:
            print("algoType, methodConst, envType: ", self.algoType, self.methodConst, self.envType) 
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)
            

    def solve_dpLagGp_opt_eval(self, trainEpisodeIndex, policyEvalIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr 
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr + occuMeas[firstGroupIndex][scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr - occuMeas[secondGroupIndex][scoreIndex][isQualIndex][1][timeStep]

                objSum = objSum - self.methodConst * expr * expr
            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            if self.hasTransProbConstr:
            #trans dyn constr
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                                exprLeft = 0
                                for decLag1Index in range(2):
                                    exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                                exprRight = 0
                                for scoreLag1Index in range(self.numScores):
                                    for isQualLag1Index in range(2):
                                        for decLag1Index in range(2):
                                            exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]

                                qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                                qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)
                #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)
                                
            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]

                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitDpOrig)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status
        if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
            count = 0
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = sol[count].X
                                count += 1 
                                self.groupList[groupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value
            self.algoEvalLogDictDict[trainEpisodeIndex] = OrderedDict()
            for groupIndex in range(self.numGroups):
                self.algoEvalLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForEval

            obj = qcqp.getObjective()
            objValEval = obj.getValue()
            
            avgFairConstrVal = 0
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][1][timeStep]
                avgFairConstrVal += abs(fairConstrVal)
            avgFairConstrVal /= self.numNnTrainSteps

        else:
            print("algoType, methodConst, methodConst, envType: ", self.algoType, self.methodConst, self.methodConst, self.envType) 
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)

    def solve_eqOptOrigGp_opt_eval(self, trainEpisodeIndex, policyEvalIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            qcqp.setObjective(objSum, GRB.MAXIMIZE)
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                exprLhs = 0

                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprLhs = exprLhs + exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1 - exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1

                exprRhs = self.estConstrForEval[policyEvalIndex] * (exprSumScoreDecFirstGroupIsQual1 * exprSumScoreDecSecondGroupIsQual1)

                qcqp.addConstr(exprLhs <= exprRhs)
                qcqp.addConstr(exprLhs >= -exprRhs)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)



            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):

                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptOrig)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()
            sol = qcqp.getVars()
            status = qcqp.Status

        try:
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X ###value#"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]

            obj = qcqp.getObjective()
            objValEval = obj.getValue()
        except AttributeError:
            objValEval = self.objValEval
        self.algoEvalLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoEvalLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForEval

        
        
        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
        
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps


    def solve_eqOptLagGp_opt_eval(self, trainEpisodeIndex, policyEvalIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))


            prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]
            prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]

             
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "gamma_" + str(timeStep))
                prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "eta_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr


            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                exprLhs = 0

                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                qcqp.addConstr(prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] == exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1)
                qcqp.addConstr(prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] == exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)



            for timeStep in range(self.numNnTrainSteps):
                exprLhs = 0
                exprLhs = exprLhs + prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] - prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep]
                objSum = objSum - self.methodConst * exprLhs * exprLhs ###+ self.methodConst * (exprRhsUnscale)

            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

             #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)      
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptLag)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status

        try:
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value #"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = qcqp.getObjective()
            objValEval = obj.getValue()
        except AttributeError:
            objValEval = self.objValEval
        self.algoEvalLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoEvalLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForEval

        
        
        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
        
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps

    def solve_aggOrigGp_opt_eval(self, trainEpisodeIndex, policyEvalIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1
        if True:

            agg = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = agg.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            agg.setObjective(objSum, GRB.MAXIMIZE)

            exprSumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

            exprSumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps): 
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

            exprLhs = ((exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1) - (exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1))### * self.numNnTrainSteps
            exprRhs = self.estConstrForEval[policyEvalIndex] * (exprSumScoreDecFirstGroupIsQual1 * exprSumScoreDecSecondGroupIsQual1)
            agg.addConstr(exprLhs <= exprRhs)
            agg.addConstr(exprLhs >= -exprRhs)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        agg.addConstr(expr <= self.optRoundEps)
                        agg.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            agg.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            agg.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):

                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    agg.addConstr(expr <= 1 + self.optRoundEps)
                    agg.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        agg.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        agg.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            agg.setParam(GRB.param.Method, 2)
            agg.setParam(GRB.param.OptimalityTol, self.optValEps)
            agg.setParam(GRB.param.FeasibilityTol, self.feasEps)
            agg.setParam(GRB.param.NumericFocus, 3)
            agg.setParam(GRB.param.NonConvex, 2)
            agg.setParam(GRB.param.MIPGap, self.mipGap)
            agg.setParam(GRB.param.TimeLimit, self.timeLimitAggOrig)
            agg.setParam(GRB.param.DualReductions, 0)
            agg.optimize()
            sol = agg.getVars()

            status = agg.Status
        try:
            if agg.Status == GRB.OPTIMAL or agg.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X#value#"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = agg.getObjective()
            objValEval = obj.getValue()
        except AttributeError:
            objValEval = self.objValEval
        self.algoEvalLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoEvalLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForEval

        
        
        avgFairConstrVal = 0
        sumScoreFirstGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

        sumScoreDecFirstGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

        sumScoreSecondGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

        sumScoreDecSecondGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

        fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1) * self.numNnTrainSteps
        avgFairConstrVal = fairConstrVal#/= self.numNnTrainSteps


    def solve_aggLagGp_opt_eval(self, trainEpisodeIndex, policyEvalIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:

            agg = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = agg.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 = agg.addVar(vtype = GRB.CONTINUOUS, name = "gamma")
            prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 = agg.addVar(vtype = GRB.CONTINUOUS, name = "eta")

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            exprSumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

            exprSumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps): 
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

            agg.addConstr(prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 == exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1)
            agg.addConstr(prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 == exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)

            exprLhs = prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 - prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1

            objSum = objSum - self.methodConst * exprLhs * exprLhs ###+ self.methodConst * (exprRhsUnscale)
            agg.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        agg.addConstr(expr <= self.optRoundEps)
                        agg.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            agg.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            agg.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):

                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    agg.addConstr(expr <= 1 + self.optRoundEps)
                    agg.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        agg.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        agg.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)
            agg.setParam(GRB.param.Method, 2)
            agg.setParam(GRB.param.OptimalityTol, self.optValEps)
            agg.setParam(GRB.param.FeasibilityTol, self.feasEps)
            agg.setParam(GRB.param.NumericFocus, 3)
            agg.setParam(GRB.param.NonConvex, 2)
            agg.setParam(GRB.param.MIPGap, self.mipGap)
            agg.setParam(GRB.param.TimeLimit, self.timeLimitAggLag)
            agg.setParam(GRB.param.DualReductions, 0)
            agg.optimize()
            sol = agg.getVars()

            status = agg.Status
        try:
            if agg.Status == GRB.OPTIMAL or agg.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X#value#"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = agg.getObjective()
            objValEval = obj.getValue()
        except AttributeError:
            objValEval = self.objValEval
        self.algoEvalLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoEvalLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForEval
        
        avgFairConstrVal = 0
        sumScoreFirstGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

        sumScoreDecFirstGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

        sumScoreSecondGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][1][timeStep]

        sumScoreDecSecondGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForEval[scoreIndex][1][decIndex][timeStep]

        fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1) * self.numNnTrainSteps
        avgFairConstrVal = fairConstrVal#/= self.numNnTrainSteps


    def solve_dpOrigGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))


            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr 
            qcqp.setObjective(objSum, GRB.MAXIMIZE)
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr + occuMeas[firstGroupIndex][scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr - occuMeas[secondGroupIndex][scoreIndex][isQualIndex][1][timeStep]

                qcqp.addConstr(expr <= self.estConstrForUpdate[policyUpdateIndex])
                qcqp.addConstr(expr >= - self.estConstrForUpdate[policyUpdateIndex])
    

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            if self.hasTransProbConstr:
            #trans dyn constr
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                                exprLeft = 0
                                for decLag1Index in range(2):
                                    exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                                exprRight = 0
                                for scoreLag1Index in range(self.numScores):
                                    for isQualLag1Index in range(2):
                                        for decLag1Index in range(2):
                                            exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
#                                qcqp.addConstr(exprLeft == exprRight)

                                
                                qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                                qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)


            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)
            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitDpOrig)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status

        if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
            count = 0
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = sol[count].X
                                count += 1
                                self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value

            #fairness constr
            for timeStep in range(self.numNnTrainSteps):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
            for groupIndex in range(self.numGroups):
                occuMeasScoreDecTm = np.sum(self.groupList[groupIndex].occuMeasNpForUpdateEval, axis = 1).transpose(2, 0, 1)
            self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
            for groupIndex in range(self.numGroups):
                self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
            
            avgFairConstrVal = 0
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                avgFairConstrVal += abs(fairConstrVal)
            avgFairConstrVal /= self.numNnTrainSteps

            optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
            self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
            if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
                print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)
        else:
            print("algoType, methodConst, envType: ", self.algoType, self.methodConst, self.envType) 
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)
            

    def solve_dpLagGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1
        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr 
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr + occuMeas[firstGroupIndex][scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr - occuMeas[secondGroupIndex][scoreIndex][isQualIndex][1][timeStep]

                objSum = objSum - self.methodConst * expr * expr


            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            if self.hasTransProbConstr:
                #trans dyn constr
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                                exprLeft = 0
                                for decLag1Index in range(2):
                                    exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                                exprRight = 0
                                for scoreLag1Index in range(self.numScores):
                                    for isQualLag1Index in range(2):
                                        for decLag1Index in range(2):
                                            exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                                qcqp.addConstr(exprLeft == exprRight)
            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)
                                
            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]

                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitDpLag)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status
        if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
            count = 0
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = sol[count].X
                                count += 1 
                                self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value
            self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
            for groupIndex in range(self.numGroups):
                self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
            
            avgFairConstrVal = 0
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                avgFairConstrVal += abs(fairConstrVal)
            avgFairConstrVal /= self.numNnTrainSteps

            optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
            self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
            if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
                print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)
        else:
            print("algoType, methodConst, methodConst, envType: ", self.algoType, self.methodConst, self.methodConst, self.envType) 
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)

    def solve_eqOptOrigGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            qcqp.setObjective(objSum, GRB.MAXIMIZE)
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):

                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprLhs = (exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1) - (exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)
                exprRhs = self.estConstrForUpdate[policyUpdateIndex] * (exprSumScoreDecFirstGroupIsQual1 * exprSumScoreDecSecondGroupIsQual1)
                qcqp.addConstr(exprLhs <= exprRhs)
                qcqp.addConstr(exprLhs >= -exprRhs)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)


            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):

                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)
            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptOrig)
            qcqp.setParam(GRB.param.DualReductions, 0)
            qcqp.optimize()
            sol = qcqp.getVars()

            status = qcqp.Status
        try:
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value#occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X#value#"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
        except AttributeError:
            objValUpdate = self.objValUpdate
        self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

        
        
        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps

        optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
        self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
        if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
            print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)

        self.objValUpdate = objValUpdate

    def solve_eqOptLagGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1
        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]
            prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]
             
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "gamma_" + str(timeStep))
                prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "eta_" + str(timeStep))
            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                exprLhs = 0

                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                qcqp.addConstr(prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] == exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1)
                qcqp.addConstr(prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] == exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)


            for timeStep in range(self.numNnTrainSteps):
                exprLhs = 0
                exprLhs = exprLhs + prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] - prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep]


                objSum = objSum - self.methodConst * exprLhs * exprLhs ###+ self.methodConst * exprRhsUnscale

            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)
            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptLag)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()
            sol = qcqp.getVars()

            status = qcqp.Status

        try:
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value #"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
        except AttributeError:
            objValUpdate = self.objValUpdate
        
        self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
        
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps

        optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
        self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
        if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
            print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)


        self.objValUpdate = objValUpdate
    def solve_eqOptSdp_opt(self, trainEpisodeIndex, policyEvalIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1


        #objective
        objSum = 0
        for groupIndex in range(self.numGroups):
            expr = 0
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):
                            expr = expr + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
            objSum = objSum + self.groupProbList[groupIndex] * expr
        objective = cp.Maximize(objSum)

        #constraints
        constraintList = []
        #fairness constr
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            exprLhs = 0

            exprSumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasCvx[scoreIndex][1][1][timeStep]

            exprSumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasCvx[scoreIndex][1][decIndex][timeStep]

            exprSumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasCvx[scoreIndex][1][1][timeStep]

            exprSumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasCvx[scoreIndex][1][decIndex][timeStep]

            exprLhs = exprLhs + self.groupProbList[firstGroupIndex] * exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1 - self.groupProbList[secondGroupIndex] * exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1

            exprRhs = self.estConstrForEval[policyUpdateIndex] * (exprSumScoreDecFirstGroupIsQual1 * exprSumScoreDecSecondGroupIsQual1)

            constraintList.append((exprLhs == exprRhs))

        #trans dyn constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                    exprLeft = 0
                    for isQualLag1Index in range(2):
                        for decLag1Index in range(2):
                            exprLeft = exprLeft + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualLag1Index][decLag1Index][timeStep]
                    exprRight = 0
                    for scoreLag1Index in range(self.numScores):
                        for isQualLag1Index in range(2):
                            for decLag1Index in range(2):
                                exprRight = exprRight + self.groupList[groupIndex].occuMeasCvx[scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][scoreLag1Index][decLag1Index]
                    constraintList.append((exprLeft <= exprRight))# + self.optRoundEps))
                    constraintList.append((exprLeft >= exprRight))# - self.optRoundEps))

        #prob box constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                            constraintList.append((self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] >= 0))##-self.optRoundEps))
                            constraintList.append((self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep] <= 1))## + self.optRoundEps))
                            

        #prob sum 1 constr
        for groupIndex in range(self.numGroups):
            for timeStep in range(self.numNnTrainSteps):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            expr = expr + self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep]
                constraintList.append((expr <= 1 + self.optRoundEps))
                constraintList.append((expr >= 1 - self.optRoundEps))



        #problem
        problem = cp.Problem(objective, constraintList)

        problem.solve()

        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):
                            value = self.groupList[groupIndex].occuMeasCvx[scoreIndex][isQualIndex][decIndex][timeStep].value

    def solve_aggOrigGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:

            agg = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = agg.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))
            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            agg.setObjective(objSum, GRB.MAXIMIZE)

            exprSumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

            exprSumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps): 
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

            exprLhs = ((exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1) - (exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1))### * self.numNnTrainSteps
            exprRhs = self.estConstrForUpdate[policyUpdateIndex] * (exprSumScoreDecFirstGroupIsQual1 * exprSumScoreDecSecondGroupIsQual1)
            agg.addConstr(exprLhs <= exprRhs)
            agg.addConstr(exprLhs >= -exprRhs)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        agg.addConstr(expr <= self.optRoundEps)
                        agg.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            agg.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            agg.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)


            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):

                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    agg.addConstr(expr <= 1 + self.optRoundEps)
                    agg.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        agg.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        agg.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            agg.setParam(GRB.param.Method, 2)
            agg.setParam(GRB.param.OptimalityTol, self.optValEps)
            agg.setParam(GRB.param.FeasibilityTol, self.feasEps)
            agg.setParam(GRB.param.NumericFocus, 3)
            agg.setParam(GRB.param.NonConvex, 2)
            agg.setParam(GRB.param.MIPGap, self.mipGap)
            agg.setParam(GRB.param.TimeLimit, self.timeLimitAggOrig)
            agg.setParam(GRB.param.DualReductions, 0)
            agg.optimize()
            sol = agg.getVars()

            status = agg.Status
        try:
            if agg.Status == GRB.OPTIMAL or agg.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X#value#"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = agg.getObjective()
            objValUpdate = obj.getValue()
        except AttributeError:
            objValUpdate = self.objValUpdate
        self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

        
        
        avgFairConstrVal = 0
        sumScoreFirstGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

        sumScoreDecFirstGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

        sumScoreSecondGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

        sumScoreDecSecondGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

        fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1) * self.numNnTrainSteps
        avgFairConstrVal = fairConstrVal#/= self.numNnTrainSteps

        #trans dyn constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                        expr = 0
                        for decLag1Index in range(2):
                            expr += self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decLag1Index][timeStep]
                        for scoreLag1Index in range(self.numScores):
                            for isQualLag1Index in range(2):
                                for decLag1Index in range(2):
                                    expr -= self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
#                        print("groupIndex, scoreIndex, isQualIndex, timeStep, transProb constr: ", abs(expr))

        #prob atom
        for groupIndex in range(self.numGroups):
            for timeStep in range(self.numNnTrainSteps):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            expr += self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep]

        optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
        self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
        if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
            print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)

        self.objValUpdate = objValUpdate

    def solve_aggLagGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:

            agg = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = agg.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 = agg.addVar(vtype = GRB.CONTINUOUS, name = "gamma")
            prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 = agg.addVar(vtype = GRB.CONTINUOUS, name = "eta")


            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr

            exprSumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

            exprSumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                for timeStep in range(self.numNnTrainSteps):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

            exprSumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps): 
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

            agg.addConstr(prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 == exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1)
            agg.addConstr(prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 == exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)

            exprLhs = prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 - prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1

            objSum = objSum - self.methodConst * exprLhs * exprLhs ### + self.methodConst * (exprRhsUnscale)

            agg.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        agg.addConstr(expr <= self.optRoundEps)
                        agg.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            agg.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            agg.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)


            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):

                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                agg.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    agg.addConstr(expr <= 1 + self.optRoundEps)
                    agg.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        agg.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        agg.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            agg.setParam(GRB.param.Method, 2)
            agg.setParam(GRB.param.OptimalityTol, self.optValEps)
            agg.setParam(GRB.param.FeasibilityTol, self.feasEps)
            agg.setParam(GRB.param.NumericFocus, 3)
            agg.setParam(GRB.param.NonConvex, 2)
            agg.setParam(GRB.param.MIPGap, self.mipGap)
            agg.setParam(GRB.param.TimeLimit, self.timeLimitAggLag)
            agg.setParam(GRB.param.DualReductions, 0)

            agg.optimize()

            sol = agg.getVars()

            status = agg.Status
        try:
            if agg.Status == GRB.OPTIMAL or agg.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X#value#"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = agg.getObjective()
            objValUpdate = obj.getValue()
        except AttributeError:
            objValUpdate = self.objValUpdate
        self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

        
        
        avgFairConstrVal = 0
        sumScoreFirstGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

        sumScoreDecFirstGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

        sumScoreSecondGroupIsQual1Dec1 = 0
        for scoreIndex in range(self.numScores):
            for timeStep in range(self.numNnTrainSteps):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

        sumScoreDecSecondGroupIsQual1 = 0
        for scoreIndex in range(self.numScores):
            for decIndex in range(2):
                for timeStep in range(self.numNnTrainSteps):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

        fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1) * self.numNnTrainSteps
        avgFairConstrVal = fairConstrVal#/= self.numNnTrainSteps

        #trans dyn constr
        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                        expr = 0
                        for decLag1Index in range(2):
                            expr += self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decLag1Index][timeStep]
                        for scoreLag1Index in range(self.numScores):
                            for isQualLag1Index in range(2):
                                for decLag1Index in range(2):
                                    expr -= self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
        #prob atom
        for groupIndex in range(self.numGroups):
            for timeStep in range(self.numNnTrainSteps):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            expr += self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep]

        optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
        self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
        if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
            print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)

        self.objValUpdate = objValUpdate

    def solve_dpCvOrigGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1


        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))


            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr 
            objSum = objSum * objSum
            qcqp.setObjective(objSum, GRB.MAXIMIZE)
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr + occuMeas[firstGroupIndex][scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr - occuMeas[secondGroupIndex][scoreIndex][isQualIndex][1][timeStep]


                qcqp.addConstr(expr <= self.estConstrForUpdate[policyUpdateIndex])
                qcqp.addConstr(expr >= - self.estConstrForUpdate[policyUpdateIndex])
    
            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            if self.hasTransProbConstr:
            #trans dyn constr
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                                exprLeft = 0
                                for decLag1Index in range(2):
                                    exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                                exprRight = 0
                                for scoreLag1Index in range(self.numScores):
                                    for isQualLag1Index in range(2):
                                        for decLag1Index in range(2):
                                            exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]

                                
                                qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                                qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)
            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)
                                

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)
            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)


            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitDpOrig)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status

        if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
            count = 0
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = sol[count].X
                                count += 1
                                self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value

            #fairness constr
            for timeStep in range(self.numNnTrainSteps):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
            for groupIndex in range(self.numGroups):
                occuMeasScoreDecTm = np.sum(self.groupList[groupIndex].occuMeasNpForUpdateEval, axis = 1).transpose(2, 0, 1)
            self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
            for groupIndex in range(self.numGroups):
                self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
            
            avgFairConstrVal = 0
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                avgFairConstrVal += abs(fairConstrVal)
            avgFairConstrVal /= self.numNnTrainSteps

            optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
            self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
            if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
                print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)
        else:
            print("algoType, methodConst, envType: ", self.algoType, self.methodConst, self.envType) 
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)
            

    def solve_dpCvLagGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estDpCosntr):#estGroupProb, estDpConstr):#estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estDpConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1


        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr 
            objSum = objSum * objSum
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr + occuMeas[firstGroupIndex][scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = expr - occuMeas[secondGroupIndex][scoreIndex][isQualIndex][1][timeStep]

                objSum = objSum - self.methodConst * expr * expr

            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            if self.hasTransProbConstr:
                #trans dyn constr
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                                exprLeft = 0
                                for decLag1Index in range(2):
                                    exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                                exprRight = 0
                                for scoreLag1Index in range(self.numScores):
                                    for isQualLag1Index in range(2):
                                        for decLag1Index in range(2):
                                            exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                                qcqp.addConstr(exprLeft == exprRight)
            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)
                                
            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]

                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitDpLag)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()

            sol = qcqp.getVars()
            status = qcqp.Status

        if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
            count = 0
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                value = sol[count].X
                                count += 1 
                                self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value
            self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
            for groupIndex in range(self.numGroups):
                self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
            
            avgFairConstrVal = 0
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                fairConstrVal = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        fairConstrVal = fairConstrVal - self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][1][timeStep]
                avgFairConstrVal += abs(fairConstrVal)
            avgFairConstrVal /= self.numNnTrainSteps

            optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
            self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
            if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
                print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)
        else:
            print("algoType, methodConst, methodConst, envType: ", self.algoType, self.methodConst, self.methodConst, self.envType) 
            with open(self.optDataCurDir + "infeasibilityErrorCode.csv","w") as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow([1])
            assert(False)

    def solve_eqOptCvOrigGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1


        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))


            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            objSum = objSum * objSum
            qcqp.setObjective(objSum, GRB.MAXIMIZE)
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):

                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprLhs = (exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1) - (exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)
                exprRhs = self.estConstrForUpdate[policyUpdateIndex] * (exprSumScoreDecFirstGroupIsQual1 * exprSumScoreDecSecondGroupIsQual1)
                qcqp.addConstr(exprLhs <= exprRhs)
                qcqp.addConstr(exprLhs >= -exprRhs)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):

                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptOrig)
            qcqp.setParam(GRB.param.DualReductions, 0)
            qcqp.optimize()
            sol = qcqp.getVars()

            status = qcqp.Status
        try:
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value#occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X#value#"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
        except AttributeError:
            objValUpdate = self.objValUpdate
        self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

        
        
        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps
        optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
        self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
        if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
            print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)

        self.objValUpdate = objValUpdate

    def solve_eqOptCvLagGp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):
        firstGroupIndex = 0
        secondGroupIndex = 1
        if True:

            qcqp = gp.Model("qcqp")

            #variables
            occuMeas = [[[[[None for timeStep in range(self.numNnTrainSteps)] for decIndex in range(2)] for isQualIndex in range(2)] for scoreIndex in range(self.numScores)] for groupIndex in range(self.numGroups)]
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep))

            prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]
            prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1 = [None for timeStep in range(self.numNnTrainSteps)]

             
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "gamma_" + str(timeStep))
                prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] = qcqp.addVar(vtype = GRB.CONTINUOUS, name = "eta_" + str(timeStep))


            #objective
            objSum = 0
            for groupIndex in range(self.numGroups):
                expr = 0
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] * self.groupList[groupIndex].estReward[scoreIndex][isQualIndex][decIndex]
                objSum = objSum + self.groupProbList[groupIndex] * expr
            objSum = objSum * objSum
            #fairness constr
            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                exprLhs = 0

                exprSumScoreFirstGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreFirstGroupIsQual1Dec1 = exprSumScoreFirstGroupIsQual1Dec1 + occuMeas[firstGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecFirstGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecFirstGroupIsQual1 = exprSumScoreDecFirstGroupIsQual1 + occuMeas[firstGroupIndex][scoreIndex][1][decIndex][timeStep]

                exprSumScoreSecondGroupIsQual1Dec1 = 0
                for scoreIndex in range(self.numScores):
                    exprSumScoreSecondGroupIsQual1Dec1 = exprSumScoreSecondGroupIsQual1Dec1 + occuMeas[secondGroupIndex][scoreIndex][1][1][timeStep]

                exprSumScoreDecSecondGroupIsQual1 = 0
                for scoreIndex in range(self.numScores):
                    for decIndex in range(2):
                        exprSumScoreDecSecondGroupIsQual1 = exprSumScoreDecSecondGroupIsQual1 + occuMeas[secondGroupIndex][scoreIndex][1][decIndex][timeStep]

                qcqp.addConstr(prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] == exprSumScoreFirstGroupIsQual1Dec1 * exprSumScoreDecSecondGroupIsQual1)
                qcqp.addConstr(prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep] == exprSumScoreSecondGroupIsQual1Dec1 * exprSumScoreDecFirstGroupIsQual1)


            for timeStep in range(self.numNnTrainSteps):
                exprLhs = 0
                exprLhs = exprLhs + prod_exprSumScoreFirstGroupIsQual1Dec1_exprSumScoreDecSecondGroupIsQual1[timeStep] - prod_exprSumScoreSecondGroupIsQual1Dec1_exprSumScoreDecFirstGroupIsQual1[timeStep]

                objSum = objSum - self.methodConst * exprLhs * exprLhs ###+ self.methodConst * exprRhsUnscale

            qcqp.setObjective(objSum, GRB.MAXIMIZE)

            #conditional independence constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for timeStep in range(self.numNnTrainSteps):#self.numNnTrainSteps-1, -1, -1):
                        exprScoreXIsQual1Dec1 = occuMeas[groupIndex][scoreIndex][1][1][timeStep]

                        exprSumDecScoreXIsQual1 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual1 = exprSumDecScoreXIsQual1 + occuMeas[groupIndex][scoreIndex][1][decIndex][timeStep]

                        exprScoreXIsQual0Dec1 = occuMeas[groupIndex][scoreIndex][0][1][timeStep]

                        exprSumDecScoreXIsQual0 = 0
                        for decIndex in range(2):
                            exprSumDecScoreXIsQual0 = exprSumDecScoreXIsQual0 + occuMeas[groupIndex][scoreIndex][0][decIndex][timeStep]

                        expr = exprScoreXIsQual1Dec1 * exprSumDecScoreXIsQual0 - exprScoreXIsQual0Dec1 * exprSumDecScoreXIsQual1


                        qcqp.addConstr(expr <= self.optRoundEps)
                        qcqp.addConstr(expr >= - self.optRoundEps)

            #trans dyn constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps-1, 0, -1):
                            exprLeft = 0
                            for decLag1Index in range(2):
                                exprLeft = exprLeft + occuMeas[groupIndex][scoreIndex][isQualIndex][decLag1Index][timeStep]
                            exprRight = 0
                            for scoreLag1Index in range(self.numScores):
                                for isQualLag1Index in range(2):
                                    for decLag1Index in range(2):
                                        exprRight = exprRight + occuMeas[groupIndex][scoreLag1Index][isQualLag1Index][decLag1Index][timeStep-1] * self.groupList[groupIndex].estTransProb[scoreIndex][isQualIndex][scoreLag1Index][isQualLag1Index][decLag1Index]
                            qcqp.addConstr(exprLeft <= exprRight + self.optRoundEpsTransProb)
                            qcqp.addConstr(exprLeft >= exprRight - self.optRoundEpsTransProb)

            #prob box constr
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        for decIndex in range(2):
                            for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] >= self.optRoundEps)
                                qcqp.addConstr(occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep] <= 1)

            #prob sum 1 constr
            for groupIndex in range(self.numGroups):
                for timeStep in range(self.numNnTrainSteps):
                    expr = 0
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep]
                    qcqp.addConstr(expr <= 1 + self.optRoundEps)
                    qcqp.addConstr(expr >= 1 - self.optRoundEps)

            #init prob constraint
            for groupIndex in range(self.numGroups):
                for scoreIndex in range(self.numScores):
                    for isQualIndex in range(2):
                        expr = 0
                        for decIndex in range(2):
                            expr = expr + occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][0]
                        qcqp.addConstr(expr <= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] + self.optRoundEps)
                        qcqp.addConstr(expr >= self.groupList[groupIndex].estInitProb[scoreIndex][isQualIndex] - self.optRoundEps)

            qcqp.setParam(GRB.param.Method, 2)
            qcqp.setParam(GRB.param.OptimalityTol, self.optValEps)
            qcqp.setParam(GRB.param.FeasibilityTol, self.feasEps)
            qcqp.setParam(GRB.param.NumericFocus, 3)
            qcqp.setParam(GRB.param.NonConvex, 2)
            qcqp.setParam(GRB.param.MIPGap, self.mipGap)
            qcqp.setParam(GRB.param.TimeLimit, self.timeLimitEqOptLag)
            qcqp.setParam(GRB.param.DualReductions, 0)

            qcqp.optimize()
            sol = qcqp.getVars()

            status = qcqp.Status

        try:
#            print(sol[0].X)
            if qcqp.Status == GRB.OPTIMAL or qcqp.Status == GRB.TIME_LIMIT:
                count = 0
                for groupIndex in range(self.numGroups):
                    for scoreIndex in range(self.numScores):
                        for isQualIndex in range(2):
                            for decIndex in range(2):
                                for timeStep in range(self.numNnTrainSteps):
                                    value = sol[count].X
                                    count += 1
                                    self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = occuMeas[groupIndex][scoreIndex][isQualIndex][decIndex][timeStep].X###value #"rho_" + str(groupIndex) + "_" + str(scoreIndex) + "_" + str(isQualIndex) + "_" + str(decIndex) + "_" + str(timeStep)]
            obj = qcqp.getObjective()
            objValUpdate = obj.getValue()
        except AttributeError:
            objValUpdate = self.objValUpdate
        
        self.algoUpdateLogDictDict[trainEpisodeIndex] = OrderedDict()
        for groupIndex in range(self.numGroups):
            self.algoUpdateLogDictDict[trainEpisodeIndex][groupIndex] = self.groupList[groupIndex].occuMeasNpForUpdateEval

        avgFairConstrVal = 0
        for timeStep in range(self.numNnTrainSteps):#-1, -1, -1):
            sumScoreFirstGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreFirstGroupIsQual1Dec1 = sumScoreFirstGroupIsQual1Dec1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecFirstGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecFirstGroupIsQual1 = sumScoreDecFirstGroupIsQual1 + self.groupList[firstGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            sumScoreSecondGroupIsQual1Dec1 = 0
            for scoreIndex in range(self.numScores):
                sumScoreSecondGroupIsQual1Dec1 = sumScoreSecondGroupIsQual1Dec1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][1][timeStep]

            sumScoreDecSecondGroupIsQual1 = 0
            for scoreIndex in range(self.numScores):
                for decIndex in range(2):
                    sumScoreDecSecondGroupIsQual1 = sumScoreDecSecondGroupIsQual1 + self.groupList[secondGroupIndex].occuMeasNpForUpdateEval[scoreIndex][1][decIndex][timeStep]

            fairConstrVal = abs(sumScoreFirstGroupIsQual1Dec1 / sumScoreDecFirstGroupIsQual1 - sumScoreSecondGroupIsQual1Dec1 / sumScoreDecSecondGroupIsQual1)
        
            avgFairConstrVal += abs(fairConstrVal)
        avgFairConstrVal /= self.numNnTrainSteps

        optObjConstrForUpdateList = [self.evalType + self.envType + self.paretoType, self.paretoType, self.methodName, self.methodConst, self.numInterPolicyUpdateEpisodes, self.curLocalTimeInMicro, self.trialTrainIndex, trainEpisodeIndex, objValUpdate, avgFairConstrVal, qcqp.MIPGap]
        self.optObjConstrForUpdateLogger.write_log_to_file(False, optObjConstrForUpdateList)
        if (trainEpisodeIndex + 1) % self.numInterPolicyUpdateEpisodes == 0:
            print("objValUpdate, avgFairConstrVal: ", objValUpdate, avgFairConstrVal)


        self.objValUpdate = objValUpdate

    def solve_myp_opt_update(self, trainEpisodeIndex, policyUpdateIndex):#, estEqOptConstr):#estGroupProb, estTransProbFirstGroup, estTransProbSecondGroup, estRewardFirstGroup, estRewardSecondGroup, estEqOptConstr):

        for groupIndex in range(self.numGroups):
            for scoreIndex in range(self.numScores):
                for isQualIndex in range(2):
                    for decIndex in range(2):
                        for timeStep in range(self.numNnTrainSteps):
                            if decIndex == 0:
                                self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = 1.0 / self.numScores / self.numNnTrainSteps * self.mypDec0Prob
                            elif decIndex == 1 and isQualIndex == 1:
                                self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = 1.0 / self.numScores / self.numNnTrainSteps * (1 - self.mypDec0Prob) * self.methodConst 
                            elif decIndex == 1 and isQualIndex == 0:
                                self.groupList[groupIndex].occuMeasNpForUpdateEval[scoreIndex][isQualIndex][decIndex][timeStep] = 1.0 / self.numScores / self.numNnTrainSteps * (1 - self.mypDec0Prob) * (1 - self.methodConst)

    def begin_est(self):
        self.status = AGENT_UPDATE

    def end_est(self):
        self.status = AGENT_UPDATE_EVAL


