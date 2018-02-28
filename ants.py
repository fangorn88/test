import numpy
import random
import sys
import copy
import time

# cost function for the distance
def cost_default(weights, distance, decay_factor=0):        
    value = weights * numpy.exp(-decay_factor * numpy.power(distance,2))
    return value

# use an ant-based algorithm
"""
1. select one node, and choose the best edge that does not close the tour, and get score for the path;
2. apply (1) using all nodes as a starting position;
3. based on the path scores, score edges and starting node;
4. launch a batch of n ants;
5. select starting position of the ant based either on constraints, or randomly based on the node score;
6. select edge based on constraints or randomly based on the edge score;
7. return to 3.
"""

class Path:
    def __init__(self, 
                 distance_matrix = None, 
                 distance_matrix_file = None,
                 cost=cost_default, 
                 fixed_position = numpy.array([]), 
                 fixed_connect = numpy.array([]),
                 connectivity_degree = 1,
                 score_momentum = 0.9,
                 accuracy = 0.01,
                 batchSize = None,
                 maximum_cycles = None,
                 accuracy_patience = None,
                 random_seed = 12345,
                 storedBestScores = 5,
                 output_file ="output_path.csv"):
 
        """

        :type distance_matrix: numpy.array
        :param distance_matrix: the distance matrix for all nodes
    
        :type distance_matrix_file: string
        :param distance_matrix_file: the path to the distance matrix csv file for all nodes (no header). Only used if distance_matrix is not given

        :type cost: function returning a numpy.array
        :param cost: calculates de cost function. Default is cost_default, the sum of all consecutive scores.

        :type fixed_position: numpy.array
        :param fixed_position: an array with predetermined positions for particular nodes. Only first node implemented. Usage example: fixed_position = numpy.array([[0,3]])

        :type fixed_connect: numpy.array
        :param fixed_connect: an array with pairs of nodes that must be consecutive. Not implemented.

        :type connectivity_degree: int
        :param connectivity_degree: the connectivity degree for the cost function. Default value is 1

        :type score_momentum: float
        :param score_momentum: the weight of the previous probability distribution for the transitions. Default = 0.9

        :type accuracy: float
        :param accuracy: minimum accuracy necessary to reset accuracy_patience counter. Default = 0.01

        :type batchSize: int
        :param batchSize: number of ants sent per batch. Default = 5 * n_nodes

        :type maximum_cycles: int
        :param maximum_cycles: maximum number of cycles before stopping regardless of accuracy. Default = 20000 * n_nodes
    
        :type accuracy_patience: int 
        :param accuracy_patience: number of cycles without minimum score improvement before stopping. Default = 30 * n_nodes
    
        :type random_seed: int
        :param random_seed: random seed for the random number generator
        
        :type storedBestScores: int
        :param storedBestScores: number of best scores to store

        :type output_file: string
        :param output_file: name of file where results are written. Default is "output_path.csv"

        """

        # distance matrix parameters
        self.SetDistanceMatrix(distance_matrix, distance_matrix_file)                                
        
        # cost function parameters
        self.cost = cost;
        self.connectivity_degree = connectivity_degree
        
        # batch size information
        self.SetBatch(batchSize, maximum_cycles, accuracy_patience)
        
        # Convergence parameters
        self.accuracy = accuracy   
        self.score_momentum = score_momentum
        
        # Path parameters
        self.start_position = None             
        self.rng = numpy.random.RandomState(random_seed)
        self.fixed_position = fixed_position
        self.fixed_connect = fixed_connect                
        
        # running model
        self.bestPath = numpy.zeros((1,self.n_nodes))     
        self.bestScore = 0.0
        self.startProbs = numpy.ones((self.n_nodes)) / self.n_nodes              
        self.startFreq = numpy.zeros((self.n_nodes))               
        self.startProbsCum = numpy.cumsum(self.startProbs) 
        self.transitionFreq = numpy.zeros((self.n_nodes, self.n_nodes))
        self.transitionProbs = self.distance_matrix + 0.00000000001
        self.transitionProbs = self.transitionProbs / numpy.sum(self.transitionProbs,axis=1)[:,None]
        self.transitionProbsCum = numpy.cumsum(self.transitionProbs,axis=1)   
        self.avgscores = []

        # enforced data
        self.EnforcePositions()
        self.EnforceConnections()

        #output
        self.output_file = output_file
        self.reportingCycle = numpy.trunc(self.accuracy_patience / 2)

        # timing
        self.scoring_time = 0
        self.walking_time = 0
        self.recalculating_time = 0

        # parameter for non-total convergence
        self.small_number = 0.000001
        
        # storing several best scores
        self.storedBestScores = storedBestScores
        self.bestPathScoreList = [0 for x in xrange(self.storedBestScores)]
        self.bestPathList = [[] for x in xrange(self.storedBestScores)]
        
        #testing vars
        self.firstPrint = True
        
        
    def SetDistanceMatrix(self, distance_matrix, distance_matrix_file):
        self.distance_matrix = distance_matrix
                
        self.n_nodes = self.distance_matrix.shape[0]
                
    def SetBatch(self, batchSize, maximum_cycles, accuracy_patience):
        self.batchSize = batchSize
        self.maximum_cycles = maximum_cycles
        self.accuracy_patience = accuracy_patience
        if batchSize == None:
            if (self.n_nodes != None):
                self.batchSize = 5 * self.n_nodes
        if maximum_cycles == None:
            if (self.n_nodes != None):
                self.maximum_cycles = 20000 * self.n_nodes
        if accuracy_patience == None:
            if (self.n_nodes != None):
                self.accuracy_patience = 30 * self.n_nodes             
        
        
    def ScorePaths(self, path):        
        scores = numpy.zeros(path.shape[0])
        distance = numpy.arange(self.connectivity_degree) + 1
        weights = numpy.zeros((path.shape[0],self.connectivity_degree))
        for st in xrange(self.n_nodes):                       
            for delta in xrange(1, self.connectivity_degree + 1):                 
                left_node = st - delta
                right_node = st + delta                  
                for r in xrange(path.shape[0]):
                    if (left_node) >= 0:
                        weights[r,delta-1] += self.distance_matrix[path[r,st],path[r,left_node]] 
                    if (right_node) < self.n_nodes:                    
                        weights[r,delta-1] += self.distance_matrix[path[r,st],path[r,right_node]]
        scores = self.cost(weights.flatten(), distance).T  
        return scores
    
    
    def LaunchRaidingParty(self, raidingPartySize = None):
        # when batchsize is None, then it is assumed that it is initializing the network    
        start = time.time()
        initialization = False        
        if raidingPartySize == None:
            initialization = True
            raidingPartySize = self.n_nodes

        probs = self.rng.rand(raidingPartySize, self.n_nodes)                
        
        forayPaths = numpy.zeros((raidingPartySize,self.n_nodes), dtype=numpy.int) - 1             
        ant_transCum = numpy.zeros(self.n_nodes)
        ant_trans = numpy.zeros(self.n_nodes)
        for n_ant in xrange(raidingPartySize):       
            for n_node in xrange(self.n_nodes):                 
                if n_node == 0:                                    
                    # first node - when initializing select first node, if not initializing use constraints or random choice                    
                    if initialization:
                        if (self.start_position == None):
                            new_node = n_ant                    
                        else:
                            new_node = self.start_position
                    else:                        
                        new_node =  next((i for i, d in enumerate(self.startProbsCum) if d >= probs[n_ant,n_node]), None)
                else:                 
                    new_node = next((i for i, d in enumerate(ant_transCum[:]) if d >= probs[n_ant,n_node]),None)                                           
                forayPaths[n_ant,n_node]=int(new_node)                               

                for repath in xrange(self.n_nodes):                    
                    ant_trans[repath] = self.transitionProbs[new_node,repath]
                for repath in xrange(n_node + 1):
                    ant_trans[forayPaths[n_ant,repath]] = 0  
                
                ant_sum = ant_trans.sum()
                ant_transCum = numpy.cumsum(ant_trans) / ant_sum
                
                old_node = -1
                if n_node > 0:
                    old_node = node
                node = new_node                                
                
        walking = time.time()        
        scores = self.ScorePaths(forayPaths)  
        scoring = time.time()     
        self.RecalculateProbabilities(scores, forayPaths)                
        recalculating = time.time()
        
        self.walking_time += (walking - start)
        self.scoring_time += (scoring - walking)        
        self.recalculating_time += (recalculating - scoring)
    
    def RecalculateProbabilities(self, scores, forayPaths):          
        
        # score paths and transitions
        self.startFreq.fill(0)
        self.transitionFreq.fill(0)
        
        oldBest = self.bestScore
        improved = False
        first_improved = False
        maxScore = numpy.max(scores)
        minScore = numpy.min(scores)
        if (maxScore > oldBest):
            self.bestScore = maxScore
            improved = True
            first_improved = True
        
        for cycle in xrange(scores.shape[0]):
            # new version of the code, to get a list of best scores
            self.CalculateBestScores(scores,forayPaths)
            
            # old version of the code for the best score
            if (scores[cycle] == self.bestScore and improved and first_improved):                
                self.bestPath = numpy.empty((1,self.n_nodes))
                self.bestPath[0,:] = forayPaths[cycle]                 
                first_improved = False
            elif (scores[cycle] == self.bestScore and not first_improved):
                found = False
                for row in self.bestPath:
                    if numpy.all(row == forayPaths[cycle]):
                        found = True        
                        break                  
                if (not found):                    
                    self.bestPath = numpy.vstack((self.bestPath, forayPaths[cycle]))                    
            self.startFreq[forayPaths[cycle,0]] +=  numpy.power((scores[cycle] - minScore)/minScore,2)
            if (scores[cycle] == self.bestScore):
                self.startFreq[forayPaths[cycle,0]] +=  numpy.power((scores[cycle] - minScore)/minScore,2)
            for cycle2 in xrange(self.n_nodes - 1):
                p1 = forayPaths[cycle,cycle2]
                p2 = forayPaths[cycle,cycle2+1]
                self.transitionFreq[p1,p2] += numpy.power((scores[cycle] - minScore)/minScore,2)
                if (scores[cycle] == self.bestScore):
                    self.transitionFreq[p1,p2] +=  numpy.power((scores[cycle] - minScore)/minScore,2)
        
        total_start = numpy.sum(self.startFreq) + self.small_number 
        total_trans = numpy.sum(self.transitionFreq, axis=1) + self.small_number
        total_trans[total_trans == 0] = 1
        #print "before recalc:"
        #print "startProbs", self.startProbs
        #print "transitionProbs", self.transitionProbs
        self.startProbs = self.startProbs * self.score_momentum + self.startFreq / total_start * (1.0 - self.score_momentum)
        self.transitionProbs = self.transitionProbs * self.score_momentum + self.transitionFreq / total_trans * (1.0 - self.score_momentum)

        #print "after recalc:"
        #print "startProbs", self.startProbs
        #print "transitionProbs", self.transitionProbs
        
        #renormalize
        self.transitionProbs = (self.transitionProbs + 0.0) / numpy.sum(self.transitionProbs,axis=1)[:,None]        
        self.startProbs = (self.startProbs + 0.0) / numpy.sum(self.startProbs)
        self.startProbsCum = numpy.cumsum(self.startProbs)            
        self.transitionProbsCum = numpy.cumsum(self.transitionProbs,axis=1)  
        self.avgscores.append(numpy.average(scores,axis=None))
        
    
    def LaunchColony(self, raidingPartySize = None, cycles = None): 
        start = time.time()
        stop_rule = False
        if (cycles == None):
            cycles = self.maximum_cycles
            stop_rule = True            
        if raidingPartySize == None:
            raidingPartySize = self.n_nodes * 5        
            
        self.LaunchRaidingParty()        
        prev_best = self.bestScore
        attempt = 0
        reportingCycle = 100
        
        for runs in xrange(cycles):   
            total_runs = runs
            if (runs%self.reportingCycle == 0):                
                #print runs, type(runs)
                print "start cycle ", runs, "while best score is", self.bestScore, "and running average score is", numpy.average(self.avgscores)
                self.avgscores = []
            self.LaunchRaidingParty(raidingPartySize = raidingPartySize)            
            if (stop_rule):                
                improvement = self.bestScore/prev_best - 1
                prev_best = self.bestScore
                if (improvement < self.accuracy):
                    if (attempt > self.accuracy_patience):                        
                        break
                        print "solution found"
                    else:
                        attempt+=1
                else:
                    attempt = 1
        end = time.time()
        
        #print results
        string1 = "Finished using " + str(raidingPartySize * total_runs) + " ants and obtaining " + str(self.bestScore)
        string2 = "Times are: walking: " + str(self.walking_time) + " scoring: " + str(self.scoring_time) + " recalculating: " + str(self.recalculating_time)
        string3 = "Best Paths are: "
        stringPaths = list()
        for pnumber in xrange(len(self.bestPathScoreList)):
           stringPaths.append("rank: " + str(pnumber) +", score: " +str(self.bestPathScoreList[pnumber])+ ", path:" + str(self.bestPathList[pnumber]))
        print string1
        print string2
        self.PrintToFile(string1, "w")
        self.PrintToFile(string2, "a+")
        self.PrintToFile(string3, "a+")
        for pnumber in xrange(len(self.bestPathScoreList)):
            self.PrintToFile(stringPaths[pnumber], "a+")

    def EnforceConnections(self):
        for n_rule in xrange(self.fixed_connect.shape[0]):
            print "nothing done yet"

    def EnforcePositions(self):
        for n_rule in xrange(self.fixed_position.shape[0]):
            print "nothing done yet other than start position"
            if (self.fixed_position[n_rule,0] == 0):
                self.start_position = self.fixed_position[n_rule,1]
                self.startProbs.fill(0)
                self.startProbs[self.start_position] = 1
                self.startProbsCum = numpy.cumsum(self.startProbs)

    def PrintToFile(self, string, openFlag):
        f = open(self.output_file, openFlag)
        f.write(string + "\n")
        f.close()
        
    
    def CalculateBestScores(self, scores, forayPaths):        
        minStored = min(self.bestPathScoreList) if len(self.bestPathScoreList) > 0 else 0
        maxNew = max(scores)       
        sortedScores = numpy.argsort(scores)[::-1]  
        for index in sortedScores: 
            if (scores[index] <= minStored) or (list(forayPaths[index]) in self.bestPathList):
                break
            for position in xrange(self.storedBestScores):
                if self.firstPrint:
                    print scores[index], self.bestPathScoreList, position
                    self.firstPrint = False
                if scores[index] > self.bestPathScoreList[position]:
                    self.bestPathScoreList.insert(position, scores[index])
                    self.bestPathScoreList = self.bestPathScoreList[0:len(self.bestPathScoreList)-1]
                    self.bestPathList.insert(position, list(forayPaths[index]))
                    self.bestPathList = self.bestPathList[0:len(self.bestPathList)-1]
                    minStored = min(self.bestPathScoreList)                      
                    break                
