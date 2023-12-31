import tensorflow as tf
import numpy as np
from random import *
import copy
import matplotlib.pyplot as plt
from AI import IA_clotilde

# =============== some testing functions =====================

def AI_coop(H_s=[],H_o=[]):
    return 'c'

def AI_defect(H_s=[],H_o=[]):
    return 'd'

def AI_constant(H_s=[], H_o=[]):
    a_0 = ['c', 'd']
    if H_s == []:
        a_s = a_0[randint(0,1)]
    else:
        a_s = H_s[-1]
    return a_s

def AI_50(H_s=[], H_o=[]):
    if H_s == [] and H_o == []:
        a_s = 'c'
    else:
        nb_coop = 0
        for i in range(len(H_o)):
            if H_o[i] == 'c':
                nb_coop += 1
        if nb_coop / len(H_o) >= 0.5:
            a_s = 'c'
        else:
            a_s = 'd'
    return a_s

def AI_copykitten(H_s=[], H_o=[]):
    if H_s == [] and H_o == []:
        a_s = 'c'
    elif len(H_s) == 1 and len(H_o) == 1:
        a_s = 'c'
    else:
        if H_o[-1] == 'c':
            a_s = 'c'
        elif H_o[-1] == 'd':
            if H_o[-2] == 'd':
                a_s = 'd'
            else:
                a_s = 'c'
    return a_s

def AI_copycat(H_s=[], H_o=[]):
    if H_s == [] and H_o == []:
        a_s = 'c'
    else:
        if H_o[-1] == 'c':
            a_s = 'c'
        elif H_o[-1] == 'd':
            a_s = 'd'
    return a_s
# ============= the game ==============

def fight(nb_rounds:int, f_player1, f_player2):
    H1 = []
    H2 = []
    score1 = 0
    score2 = 0

    for i in range(nb_rounds):
        a1 = f_player1(H1, H2)
        a2 = f_player2(H2, H1)
        H1.append(a1)
        H2.append(a2)


    for i in range(len(H1)):
        score1 += payoffs(H1[i], H2[i])[0]
        score2 += payoffs(H1[i], H2[i])[1]
    
    return score1, score2

def payoffs(a1, a2):
    match = a1 + a2
    if match == 'cc':
        payoff = [2,2]
    elif match == 'dd':
        payoff = [0,0]
    elif match == 'cd':
        payoff = [-1,3]
    else:
        payoff = [3,-1]
    return payoff


#to launch the tournament
def contest(nb_rounds):
    #data description (names of the players, scores, etc)
    players = {
        AI_coop : 0,
        AI_constant : 0,
        AI_50 : 0,
        IA_clotilde : 0
    }

    #the list of the playeurs
    players_l = list(players.keys())

    for i in range(len(players_l)):
        for j in range(i + 1, len(players_l)):
            res = fight(nb_rounds, players_l[i], players_l[j])

            players[list(players.keys())[i]] += res[0]
            players[list(players.keys())[j]] += res[1]


    return players

players_boxplot = {
    AI_coop : [],
    AI_50 : [],
    AI_constant: [],
    AI_copycat : [],
    AI_copykitten : []
}

#the list of the playeurs
AIs = list(players_boxplot.keys())

for i in range(len(AIs)):
    for j in range(5):
        res = fight(100, AIs[i], IA_clotilde)

        diff_scores = res[1] - res[0]

        players_boxplot[AIs[i]].append(diff_scores)

print(players_boxplot)

data = list(players_boxplot.values())
labels = [i.__name__ for i in list(players_boxplot.keys())]

plt.boxplot(data, labels=labels)
plt.xlabel("Our AI versus the other given AI")
plt.ylabel("Performance through 5 games of 100 rounds")
plt.savefig('AI_performances.png')
plt.show()