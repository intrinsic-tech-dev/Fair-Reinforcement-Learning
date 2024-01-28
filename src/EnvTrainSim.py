import json, sys, os
from os import path
from time import strftime, gmtime

import torch

from DataStruct import CbData, Reward, State

from Config import Config
from Path import Path

from torch.profiler import profile, record_function, ProfilerActivity

import random

import numpy as np

class GroupEnv(object):
    def __init__(self):
        self.minScore = None
        self.numScores = None

        self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList = None
        self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList = None
        self.transProbIsQualCondDecLag1Val0ListList = None

        self.transProbScoreCondScoreLag1IsQualVal1ListList = None
        self.transProbScoreCondScoreLag1IsQualVal0ListList = None

        self.rewardCoefDec1IsQual1 = None
        self.rewardCoefDec1IsQual0 = None
        self.rewardCoefDec0 = None

        self.sumActIsQualPerScoreTm = None
        self.sumScoreAct1IsQual1PerTm = None
        self.sumScoreActIsQual1PerTm = None

        self.scoreIndex = None
        self.isQual = None

    def init(self, numScores, minScore, numNnTrainSteps, isQualDnDecLag1Val1IsQualLag1Val1WgtDisList, isQualEqDecLag1Val1IsQualLag1Val1WgtDis, isQualUpDecLag1Val1IsQualLag1Val1WgtDisList, isQualDnDecLag1Val1IsQualLag1Val0WgtDisList, isQualEqDecLag1Val1IsQualLag1Val0WgtDis, isQualUpDecLag1Val1IsQualLag1Val0WgtDisList, isQualDnDecLag1Val0WgtDisList, isQualEqDecLag1Val0WgtDis, isQualUpDecLag1Val0WgtDisList, scoreScoreLag1IsQualVal1DnWgtDisList, scoreScoreLag1IsQualVal1EqWgtDis, scoreScoreLag1IsQualVal1UpWgtDisList, scoreScoreLag1IsQualVal0DnWgtDisList, scoreScoreLag1IsQualVal0EqWgtDis, scoreScoreLag1IsQualVal0UpWgtDisList, rewardCoefDec1IsQual1, rewardCoefDec1IsQual0, rewardCoefDec0):

        self.minScore = minScore
        self.numScores = numScores
        self.numNnTrainSteps = numNnTrainSteps

        self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList = []
        for i in range(2):
            probWgtList = isQualDnDecLag1Val1IsQualLag1Val1WgtDisList[1:(i+1)][::-1] + [isQualEqDecLag1Val1IsQualLag1Val1WgtDis] + isQualUpDecLag1Val1IsQualLag1Val1WgtDisList[1:(2-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList.append(nrmProbWgtList)

        self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList = []
        for i in range(2):
            probWgtList = isQualDnDecLag1Val1IsQualLag1Val0WgtDisList[1:(i+1)][::-1] + [isQualEqDecLag1Val1IsQualLag1Val0WgtDis] + isQualUpDecLag1Val1IsQualLag1Val0WgtDisList[1:(2-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList.append(nrmProbWgtList)


        self.transProbIsQualCondDecLag1Val0ListList = []
        for i in range(2):
            probWgtList = isQualDnDecLag1Val0WgtDisList[1:(i+1)][::-1] + [isQualEqDecLag1Val0WgtDis] + isQualUpDecLag1Val0WgtDisList[1:(2-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbIsQualCondDecLag1Val0ListList.append(nrmProbWgtList)


        self.transProbScoreCondScoreLag1IsQualVal1ListList = []
        for i in range(self.numScores):
            probWgtList = scoreScoreLag1IsQualVal1DnWgtDisList[1:(i+1)][::-1] + [scoreScoreLag1IsQualVal1EqWgtDis] + scoreScoreLag1IsQualVal1UpWgtDisList[1:(self.numScores-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbScoreCondScoreLag1IsQualVal1ListList.append(nrmProbWgtList)


        self.transProbScoreCondScoreLag1IsQualVal0ListList = []
        for i in range(self.numScores):
            probWgtList = scoreScoreLag1IsQualVal0DnWgtDisList[1:(i+1)][::-1] + [scoreScoreLag1IsQualVal0EqWgtDis] + scoreScoreLag1IsQualVal0UpWgtDisList[1:(self.numScores-i)]
            nrmProbWgtList = [probWgt * 1.0 / sum(probWgtList) for probWgt in probWgtList]
            self.transProbScoreCondScoreLag1IsQualVal0ListList.append(nrmProbWgtList)

        
        self.rewardCoefDec1IsQual1 = rewardCoefDec1IsQual1#np.random.random_sample()
        self.rewardCoefDec1IsQual0 = rewardCoefDec1IsQual0#np.random.random_sample()
        self.rewardCoefDec0 = rewardCoefDec0


        self.sumActIsQualPerScoreTm = np.zeros(shape = (self.numScores, self.numNnTrainSteps))
        self.sumScoreAct1IsQual1PerTm = np.zeros(shape = (self.numNnTrainSteps))
        self.sumScoreActIsQual1PerTm = np.zeros(shape = (self.numNnTrainSteps))

    def epiInit(self):
        self.isQual = np.argmax(np.random.multinomial(1, [0.5, 0.5]))
        initScoreProbList = [1.0/self.numScores] * self.numScores
        self.scoreIndex = np.argmax(np.random.multinomial(1, initScoreProbList))

    def create_isQual(self, decision):
        if decision == 1:
            if self.isQual == 1:
                isQual = np.argmax(np.random.multinomial(1, self.transProbIsQualCondIsQualLag1Val1DecLag1Val1ListList[self.isQual]))
            else:
                isQual = np.argmax(np.random.multinomial(1, self.transProbIsQualCondIsQualLag1Val0DecLag1Val1ListList[self.isQual]))
        else:
            isQual = np.argmax(np.random.multinomial(1, self.transProbIsQualCondDecLag1Val0ListList[self.isQual]))
        return isQual

    def create_scoreIndex(self, isQual):
        if isQual == 1:
            scoreIndex = np.argmax(np.random.multinomial(1, self.transProbScoreCondScoreLag1IsQualVal1ListList[self.scoreIndex]))
        else:
            scoreIndex = np.argmax(np.random.multinomial(1, self.transProbScoreCondScoreLag1IsQualVal0ListList[self.scoreIndex]))
        return scoreIndex

    def create_rewardVal(self, decision):#how the decision impacting the distribution
        if decision == 1:
            if self.isQual == 1:
                rewardVal = self.rewardCoefDec1IsQual1 #* self.scoreIndex2Score(self.scoreIndex) + self.rewardCoefDecNeg
            else:
                rewardVal = self.rewardCoefDec1IsQual0 #self.rewardCoefDecPos * self.score + self.rewardCoefDecNeg
        else:
            rewardVal = self.rewardCoefDec0
        return rewardVal

    def update_sumActIsQualPerScoreTm(self, timeIndex):
        self.sumActIsQualPerScoreTm[self.scoreIndex][timeIndex] += 1

    def update_sumScoreAct1IsQual1PerTm(self, decision, timeIndex):
        if self.isQual == 1:
            if decision == 1:
                self.sumScoreAct1IsQual1PerTm[timeIndex] += 1

    def update_sumScoreActIsQual1PerTm(self, timeIndex):
        if self.isQual == 1:
            self.sumScoreActIsQual1PerTm[timeIndex] += 1

    def set_isQual(self, isQual):
        self.isQual = isQual

    def set_scoreIndex(self, scoreIndex):
        self.scoreIndex = scoreIndex

    def scoreIndex2Score(self, scoreIndex):
        return scoreIndex + self.minScore

class EnvTrainSim(object):
    def __init__(self):
        self.path = None
        self.useConfigMode = None
        self.identifier = None
        self.device = None

        self.nnModel = None

        self.nnInputDim = None
        self.nnHiddenDim = None
        self.nnTargetDim = None

        self.nStates = None
        self.nActions = None

        self.numNnTrainSteps = None

        self.stepIndex = None

        #rl mem var
        self.initState = None
        self.state = None

        self.rewardFunc = None

        self.initDataInputState = None
        self.initDataTargetState = None
        self.initDataInputReward = None
        self.initDataTargetReward = None

        self.fbDataInputState = None
        self.fbDataTargetState = None
        self.fbDataInputReward = None
        self.fbDataTargetReward = None

        self.groupList = None
        self.groupProbList = None

    def read_config(self, identifier, rootDir, optConfigCurDir, optDataCurDir, iptConfigHistDir, iptDataHistDir, device, numEpisode, numInterPolicyUpdateEvalEpisodes, numNnTrainSteps, groupProbList):#, nnBatchSize, nnInputDim, nnHiddenDim, nnTargetDim, nnLayerDim, numNnTrainSteps):
        iptConfigFileName, self.optConfigCurDir, self.iptConfigHistDir, self.optDataCurDir = Path.make_path(rootDir, self.__class__.__name__, identifier, optConfigCurDir, iptConfigHistDir, optDataCurDir)
        self.iptDataHistDir = iptDataHistDir

        
        with open(iptConfigFileName) as json_file:
            config = json.load(json_file)
 
            nnModelName = config["nnModelName"]

            self.rewardFuncName = config["rewardFuncName"]

            self.numScores = config["numScores"]
            self.minScore = config["minScore"]
            self.isQualDnDecLag1Val1IsQualLag1Val1WgtDisList = config["isQualDnDecLag1Val1IsQualLag1Val1WgtDisList"]
            self.isQualEqDecLag1Val1IsQualLag1Val1WgtDis = config["isQualEqDecLag1Val1IsQualLag1Val1WgtDis"]
            self.isQualUpDecLag1Val1IsQualLag1Val1WgtDisList = config["isQualUpDecLag1Val1IsQualLag1Val1WgtDisList"]
            self.isQualDnDecLag1Val1IsQualLag1Val0WgtDisList = config["isQualDnDecLag1Val1IsQualLag1Val0WgtDisList"]
            self.isQualEqDecLag1Val1IsQualLag1Val0WgtDis = config["isQualEqDecLag1Val1IsQualLag1Val0WgtDis"]
            self.isQualUpDecLag1Val1IsQualLag1Val0WgtDisList = config["isQualUpDecLag1Val1IsQualLag1Val0WgtDisList"]
            self.isQualDnDecLag1Val0WgtDisList = config["isQualDnDecLag1Val0WgtDisList"]
            self.isQualEqDecLag1Val0WgtDis = config["isQualEqDecLag1Val0WgtDis"]
            self.isQualUpDecLag1Val0WgtDisList = config["isQualUpDecLag1Val0WgtDisList"]
            self.scoreScoreLag1IsQualVal1DnWgtDisList = config["scoreScoreLag1IsQualVal1DnWgtDisList"]
            self.scoreScoreLag1IsQualVal1EqWgtDis = config["scoreScoreLag1IsQualVal1EqWgtDis"]
            self.scoreScoreLag1IsQualVal1UpWgtDisList = config["scoreScoreLag1IsQualVal1UpWgtDisList"]
            self.scoreScoreLag1IsQualVal0DnWgtDisList = config["scoreScoreLag1IsQualVal0DnWgtDisList"]
            self.scoreScoreLag1IsQualVal0EqWgtDis = config["scoreScoreLag1IsQualVal0EqWgtDis"]
            self.scoreScoreLag1IsQualVal0UpWgtDisList = config["scoreScoreLag1IsQualVal0UpWgtDisList"]
 
            self.rewardCoefDec1IsQual1List = config["rewardCoefDec1IsQual1List"]
            self.rewardCoefDec1IsQual0List = config["rewardCoefDec1IsQual0List"]
            self.rewardCoefDec0List = config["rewardCoefDec0List"]

          

            self.identifier = identifier
            self.device = device

            self.numEpisode = numEpisode

            self.numInterPolicyUpdateEvalEpisodes = numInterPolicyUpdateEvalEpisodes
            self.numNnTrainSteps = numNnTrainSteps
            
            self.stepIndexCrossEpisode = 0

            self.groupProbList = groupProbList

            self.groupList = []
            self.numGroups = len(self.groupProbList)
            for i in range(self.numGroups):
                group = GroupEnv()
                group.init(self.numScores, self.minScore, self.numNnTrainSteps, self.isQualDnDecLag1Val1IsQualLag1Val1WgtDisList, self.isQualEqDecLag1Val1IsQualLag1Val1WgtDis, self.isQualUpDecLag1Val1IsQualLag1Val1WgtDisList, self.isQualDnDecLag1Val1IsQualLag1Val0WgtDisList, self.isQualEqDecLag1Val1IsQualLag1Val0WgtDis, self.isQualUpDecLag1Val1IsQualLag1Val0WgtDisList, self.isQualDnDecLag1Val0WgtDisList, self.isQualEqDecLag1Val0WgtDis, self.isQualUpDecLag1Val0WgtDisList, self.scoreScoreLag1IsQualVal1DnWgtDisList, self.scoreScoreLag1IsQualVal1EqWgtDis, self.scoreScoreLag1IsQualVal1UpWgtDisList, self.scoreScoreLag1IsQualVal0DnWgtDisList, self.scoreScoreLag1IsQualVal0EqWgtDis, self.scoreScoreLag1IsQualVal0UpWgtDisList, self.rewardCoefDec1IsQual1List[i], self.rewardCoefDec1IsQual0List[i], self.rewardCoefDec0List[i])
                self.groupList.append(group)


            Config.write_config(config, self.optConfigCurDir + self.__class__.__name__ + self.identifier + ".json")


    def load_init_data(self, dataInputState, dataTargetState, dataInputReward, dataTargetReward):
        self.initDataInputState = dataInputState
        self.initDataTargetState = dataTargetState
        self.initDataInputReward = dataInputReward
        self.initDataTargetReward = dataTargetReward

    def load_fb_data(self, dataInputState, dataTargetState, dataInputReward, dataTargetReward):
        self.fbDataInputState = dataInputState
        self.fbDataTargetState = dataTargetState
        self.fbDataInputReward = dataInputReward
        self.fbDataTargetReward = dataTargetReward

    def epiInit(self):
        for groupIndex in range(self.numGroups):
            self.groupList[groupIndex].epiInit()

        cbData = CbData()
    
        scoreIndexList = []
        for groupIndex in range(self.numGroups):
            scoreIndexList.append(self.groupList[groupIndex].scoreIndex)
        cbData.set_scoreIndexList(scoreIndexList)

        self.stepIndex = 0

        self.cbData = cbData

        return cbData#.detach()


    def getCbData(self):
        return self.cbData

    def feedback(self, action):
        #part 1: forward dynamics for the group of the action, update this group stats
        state = State()

        memScoreIndexList = []
        memIsQualList = []
        for groupIndex in range(self.numGroups):
            memScoreIndexList.append(self.groupList[groupIndex].scoreIndex)
            memIsQualList.append(self.groupList[groupIndex].isQual)
        state.set(memScoreIndexList, memIsQualList)
        
        memRewardValList = []
        for groupIndex in range(self.numGroups):
            memRewardVal = self.groupList[groupIndex].create_rewardVal(action.decisionList[groupIndex])#.groupActionList[groupIndex])
            memRewardValList.append(memRewardVal)

        reward = Reward()
        reward.set_groupProb(self.groupProbList)
        reward.set_rewardVal(memRewardValList)

        isQualList = []
        scoreIndexList = []
        for groupIndex in range(self.numGroups):
            isQual = self.groupList[groupIndex].create_isQual(action.decisionList[groupIndex])

            scoreIndex = self.groupList[groupIndex].create_scoreIndex(isQual)

            isQualList.append(isQual)
            scoreIndexList.append(scoreIndex)

        cbData = CbData()
        cbData.set_scoreIndexList(scoreIndexList)
        self.cbData = cbData

        for groupIndex in range(self.numGroups):

            self.groupList[groupIndex].update_sumActIsQualPerScoreTm(self.stepIndex)

            self.groupList[groupIndex].update_sumScoreAct1IsQual1PerTm(action.decisionList[groupIndex], self.stepIndex)
     
            self.groupList[groupIndex].update_sumScoreActIsQual1PerTm(self.stepIndex)

            self.groupList[groupIndex].set_isQual(isQualList[groupIndex])
            self.groupList[groupIndex].set_scoreIndex(scoreIndexList[groupIndex])

        self.stepIndex += 1
        self.stepIndexCrossEpisode += 1


        return state, reward

    def est_dp(self):#, trainEpisodeIndex):
        firstGroupIndex = 0
        secondGroupIndex = 1
        if True:
            estDp = 0
            for scoreIndex in range(self.numScores):
                for timeIndex in range(self.numNnTrainSteps):
                    estDp += abs(self.groupList[firstGroupIndex].sumActIsQualPerScoreTm[scoreIndex][timeIndex] - self.groupList[secondGroupIndex].sumActIsQualPerScoreTm[scoreIndex][timeIndex])
            estDp /= self.numNnTrainSteps * self.numInterPolicyUpdateEvalEpisodes * 1.0

            self.groupList[firstGroupIndex].sumActIsQualPerScoreTm.fill(0)
            self.groupList[secondGroupIndex].sumActIsQualPerScoreTm.fill(0)

            return estDp

    def est_eqOpt(self):#, trainEpisodeIndex):
        firstGroupIndex = 0
        secondGroupIndex = 1

        if True:
            estEqOpt = 0
            for timeIndex in range(self.numNnTrainSteps):
                estEqOpt += abs(self.groupList[firstGroupIndex].sumScoreAct1IsQual1PerTm[timeIndex] * 1.0 / max(self.groupList[firstGroupIndex].sumScoreActIsQual1PerTm[timeIndex], 1) - self.groupList[secondGroupIndex].sumScoreAct1IsQual1PerTm[timeIndex] * 1.0 / max(self.groupList[secondGroupIndex].sumScoreActIsQual1PerTm[timeIndex], 1))
            estEqOpt /= self.numNnTrainSteps * 1.0 #* self.numInterPolicyUpdateEvalEpisodes * 1.0

            self.groupList[firstGroupIndex].sumScoreAct1IsQual1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreAct1IsQual1PerTm.fill(0)
            self.groupList[firstGroupIndex].sumScoreActIsQual1PerTm.fill(0)
            self.groupList[secondGroupIndex].sumScoreActIsQual1PerTm.fill(0)
            return estEqOpt






