import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random


def spiski_smezh(matrix):
    spiski=[]
    for i,row in enumerate(matrix):
        spisok = []
        for j,col in enumerate(row):
            if col == 1:
                spisok.append(j+1)
        spiski.append(spisok)
    return(spiski)



def nazad(q_,q ,k,nez,x, graph):
    k -= 1
    nez.remove(nez[k+1])
    q_[k] = q_[k]+[(x[k+1][0])]
    q_.pop(k+1)
    q[k].remove(x[k+1][0])
    q.pop(k+1)

    x.remove(x[k+1])

    if k == 0 and len(q[k]) == 1:

        return True, q_, q, k, nez, x, graph
    else:
        return False, q_, q, k, nez, x, graph

def check(q_, q, k, nez, x, graph):
    counter = 0
    if len(q_[k]) > 1:
        for u in q_[k]:
            for l in graph[u-1]:
                for j in q[k]:
                    if u != 0 and j !=0:
                        if l == j:
                            counter += 1

        if counter == 0:
            return True, q_, q, k, nez, x, graph


    if len(q[k]) == 1:
        if len(q_[k]) == 1:
            nezmn = []
            for el in nez[k]:
                if el != 0:
                    nezmn.append(el-1)
            print('Максимальное независимое множество:', nezmn)

        return True, q_, q, k, nez, x, graph

    else:
        return False,q_, q, k, nez, x, graph


def vperyod(q_,q ,k,nez, x, graph):

    x.append([q[k][1]])
    nez.append(nez[k]+[q[k][1]])

    q1_ = copy.deepcopy(q_)

    if len(q_[k]) > 1:
        for el in q_[k]:
            for i in graph[x[k+1][0]-1]:
                if el == i:
                    q1_[k].remove(el)

    q_.append(q1_[k])


    q1 = copy.deepcopy(q)
    if len(q[k]) > 1:
        for el in q[k]:

            for i in graph[x[k+1][0]-1]:

                if el == i and el!=0:

                    q1[k].remove(el)


    del q1[k][1]
    q.append(q1[k])
    k += 1
    return q_, q, k, nez, x, graph




def mnm(graph):
    graph = spiski_smezh(graph)
    k = 0
    nez = [[0]]
    q_ = [[0]]
    q = []
    [q.append([i for i in range(len(graph)+1)])]
    x = [[0]]


    while True:

        kuda, q_, q, k, nez, x, graph = check(q_, q, k, nez, x, graph)
        if kuda:
            is_stop, q_, q, k, nez, x, graph = nazad(q_, q, k, nez, x, graph)
            if is_stop:
                return
        else:
            q_, q, k, nez, x, graph = vperyod(q_, q, k, nez, x, graph)




def generatsiya(p, n):
    m = int(p*n*(n-1)/2)
    M = []
    for i in range(n-1):
        for j in range(i+1,n):
            M.append([i, j])

    M =sorted(random.SystemRandom().sample(M, m))

    graph = [[0] * n for _ in range(n)]
    for i in range(m):
        graph[M[i][0]][M[i][1]] = 1
        graph[M[i][1]][M[i][0]] = 1
    return graph




A = generatsiya(0.5, 100)

mnm(A)


G=nx.DiGraph(np.matrix(A))
nx.draw(G, with_labels=True, node_size=300,arrows=False)
plt.show()