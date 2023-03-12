import csv
import sys
import torch
import torch.nn.functional as F
import numpy as np
from random import shuffle


class DataLoader:
    
    def __init__(self, num_P=180, num_NCT=247, num_I=361, num_O=629):

        self.num_P = num_P
        self.num_NCT = num_NCT
        self.num_I = num_I
        self.num_O = num_O

        self.NCTP, self.NCTI, self.NCTO, self.Problem, self.Intervention, self.Outcome = self._loadRawData_('./ClinicalTrails.csv')
        self.testNCTP, self.testProblem, self.trainNCTP, self.trainProblem = self._dataPartition_(self.NCTP, self.Problem)

    def get_num(self):
        return {'num_P': self.num_P,
                'num_NCT': self.num_NCT,
                'num_I': self.num_I,
                'num_O': self.num_O}

    def _loadRawData_(self, path):

        with open(path) as f:
            reader = csv.reader(f)
            data = list(reader)
        NCTP = []
        NCTI = []
        NCTO = []
        Problem = []
        Intervention = []
        Outcome = []
        for triad in data:
            if triad[1] == 'hasProblem':
                NCTP.append(int(triad[0]) - self.num_P)
                Problem.append(int(triad[2]))
            elif triad[1] == 'hasIntervention':
                NCTI.append(int(triad[0]) - self.num_P)
                Intervention.append(int(triad[2]) - self.num_P - self.num_NCT)
            elif triad[1] == 'hasOutcome':
                NCTO.append(int(triad[0]) - self.num_P)
                Outcome.append(int(triad[2]) - self.num_P - self.num_NCT - self.num_I)
            else:
                sys.exit('Invalid meta path: ', triad)
        f.close()
        return NCTP, NCTI, NCTO, Problem, Intervention, Outcome

    def _dataPartition_(self, NCTP, Problem):
        testSize = int(len(NCTP)*0.2)
        num_Problem = len(Problem)
        testProblem = list()
        testNCTP = list()
        trainProblem = list()
        trainNCTP = list()
        pairs = list(zip(Problem, NCTP))
        shuffle(pairs)
        # print(pairs)
        num_test = 0
        num = 0
        while num < num_Problem:
            if num_test <= testSize and Problem.count(pairs[num][0]) != 1 and NCTP.count(pairs[num][1]) != 1:
                testProblem.append(pairs[num][0])
                testNCTP.append(pairs[num][1])
                num_test = num_test+1
                num = num+1
            else:
                Problem.remove(pairs[num][0])
                NCTP.remove(pairs[num][1])
                trainProblem.append(pairs[num][0])
                trainNCTP.append(pairs[num][1])
                num = num+1

        return testNCTP, testProblem, trainNCTP, trainProblem

    def generateRawTrainGraph(self):
        return {'NCTP': self.trainNCTP, 
                'NCTI': self.NCTI, 
                'NCTO': self.NCTO, 
                'Problem': self.trainProblem, 
                'Intervention': self.Intervention, 
                'Outcome': self.Outcome}

    def generateTestData(self):
        return {'testProblem': self.testProblem, 
                'testNCTP': self.testNCTP}

    def _similarNCT_(self, path, similarity):

        head = []
        tail = []
        distanceList = []
        attributeEmbedding = torch.load(path)
        NCTAttributeEmb = attributeEmbedding[self.num_P:self.num_P+self.num_NCT]
        for i in range(len(NCTAttributeEmb)):
            for j in range(len(NCTAttributeEmb)):
                distance = F.pairwise_distance(NCTAttributeEmb[i], NCTAttributeEmb[j], p=2).tolist()  # type为float, 不是list, 因为只有一个
                if i != j:
                    head.append(i)
                    tail.append(j)
                    distanceList.append(distance)
        distanceList = np.array(distanceList)
        similarityList = 1 - (distanceList - np.min(distanceList)) / (np.max(distanceList) - np.min(distanceList))
        similarityList = similarityList.tolist()
        finalHead = []
        finalTail = []
        counter = 0
        for i in range(len(NCTAttributeEmb)):
            finalHead.append(i)
            finalTail.append(i)
        for i in range(len(similarityList)):
            if similarityList[i] > similarity:
                finalHead.append(head[i])
                finalTail.append(tail[i])
                counter = counter+1
        return finalHead, finalTail, counter

    def generateTrailGraph(self, similarity):
        head, tail, counter = self._similarNCT_('./AttributeEmbedding.pt', similarity)
        return {'sourceNCT': head,
                'targetNCT': tail}

    def loadAttributeEmb(self, path):
        loadTensor = torch.load(path)
        ProblemFeature = loadTensor[0 : self.num_P]
        NCTFeature = loadTensor[self.num_P : self.num_P+self.num_NCT]
        InterventionFeature = loadTensor[self.num_P+self.num_NCT : self.num_P+self.num_NCT+self.num_I]
        OutcomeFeature = loadTensor[self.num_P+self.num_NCT+self.num_I:]
        return {'ProblemFeature': ProblemFeature, 
                'NCTFeature': NCTFeature, 
                'InterventionFeature': InterventionFeature, 
                'OutcomeFeature': OutcomeFeature}
