import os
import networkx as nx
import numpy as np

# =============================================================================
# Extract the graph information from the lg file
# =============================================================================

def openLg(file_name):
    # Open file
    f = open(file_name, "r")
    # Graphs: List for containing all graphs in the file
    Graphs = []
    G = None
    lineN = 0
    for line in f:
        lineN = lineN + 1
        # t new graph
        # v: vertex id label
        # e edge source end label
        x = line.split(" ")
        if x[0] == 't':
            # Creates graph
            G = nx.Graph()
            Graphs.append(G)
        elif x[0] == 'v':
            # x[1] # ID
            # x[2] # label
            x[2] = x[2].replace('\n', '')
            #print("x", x[2])
            G.add_node(x[1], label=x[2])
        elif x[0] == 'e':
            # x[1]  # from
            # x[2]  # to
            # x[3]  # label
            #print(x[3])
            if x[3].isnumeric():
                G.add_edge(x[1], x[2], label=int(x[3].rstrip("\n")))
            elif x[3].isalpha():
            #print(x)
                G.add_edge(x[1], x[2], label=x[3].rstrip("\n"))
            if len(x) == 4:
                #print(x)
                if x[3].isdigit():
                    G.add_edge(x[1], x[2], label=int(x[3].rstrip("\n")))
                else:
                    G.add_edge(x[1], x[2], label=x[3].rstrip("\n"))
            elif len(x) == 3:
                #print(x)
                G.add_edge(x[1], x[2].rstrip("\n"), label=0)
        else:
            # Error
            print("Error: The line " + str(lineN) + " is not part of a graph")
            print(line)
    return Graphs

# =============================================================================
# Loads the patterns information
# =============================================================================

def loadPatters (DS_f, P_f):
    D = openLg(DS_f)
    logP = open(P_f, "r")
    Patterns = []
    Supports = []
    for line in logP:
        line = line.replace('\n','').strip()
        if 'Pattern' in line and not "Mined" in line and not "Unique" in line:
            S = None
            campos = line.split('|')
            graph_info = {}
            for c in campos:
                c = c.strip()
                if "Graph:" in c: # get graph id
                    graph_info['id'] = int(c.replace('Graph:',''))
                if "Node Index:" in c:
                    c = c.replace('Node Index: ','')
                    nodes = []
                    for n in c.split(' '):
                        nodes.append(str(int(n)))
                    graph_info['nodes'] = nodes
                if "Support: " in c:
                    S = int (c.replace("Support: ", ''))
                if "Edge Index: " in c:
                    edges = []
                    c = c.replace("Edge Index: ", '')
                    edges_str = c.replace("'", '').replace(")",'').split('(')
                    for edge_str in edges_str:
                        if len(edge_str.strip()) > 0 :
                            e_arr = edge_str.strip().split(',')
                            edge = (str(int(e_arr[0])),str(int(e_arr[1])))
                            edges.append(edge)
                    graph_info['edges'] = edges
            #print (graph_info)   
            G = D[graph_info['id']]
            P = None
            
            if 'edges' in graph_info and len(graph_info['edges']) > 0:
                P = G.edge_subgraph(graph_info['edges'])
            else:
                P = G.subgraph(graph_info['nodes'])
            Patterns.append(P)
            Supports.append(S)
    return Patterns, Supports


# =============================================================================
# Functions to compute the dissimilarity
# =============================================================================


def dissimilarity_nodes (g1, g2, n1, n2):
    if g1.nodes[n1]['label'] == g2.nodes[n2]['label']:
        return 0
    return 1

def dissimilarity_edges (g1, g2, e1, e2):
    if g1.edges[e1]['label'] == g2.edges[e2]['label']:
        return 0
    return 1

def edge_nneighbors(G, sg):
    nodos = sg.nodes()
    vecinos = []
    for n in nodos:
        for ne in G.neighbors(n):
            if not (sg.has_edge(n,ne)):
                vecinos.append((n,ne))
    return vecinos

def grewPattern(G, sg, e):
    new = sg.copy()
    n = None
    edge = None
    if not (sg.has_node(e[0])):
	# ERROR
    if not (sg.has_node(e[1])):
        new.add_node(e[1], label=G.nodes[e[1]]['label'])
        n = e[1]
    new.add_edge(e[0], e[1], label=G.edges[e]['label'])
    edge = e
    return new, n, edge


def copy_occurrence(o):
    occ_new = {}
    occ_new['g'] = o['g'].copy()
    occ_new['d'] = o['d']
    occ_new['m'] = o['m'].copy()
    return occ_new

def is_mapped(map, n_p):
    for m in map:
        if map[m] == n_p:
            return True
    return False


def calculate_mapping (map, dis , added_n_o, added_e_o,  added_n_p, added_e_p, G, P):
    map_new = map.copy()
    new_dis = dis + 2 #add two edges
    if added_n_p != None:
        new_dis = new_dis + 1
    if added_n_o != None:
        new_dis = new_dis + 1

    if added_n_p != None and added_n_o != None:
        if  added_e_o[0] in map and map[added_e_o[0]] == added_e_p[0]:
            #the edge and new node can be mapped
            new_dis = (new_dis - 4 + dissimilarity_edges(G,P,added_e_o, added_e_p) +
                       dissimilarity_nodes(G, P, added_n_o, added_n_p))
            map_new[added_n_o] = added_n_p
        elif (not added_e_o[0] in map) and (not is_mapped(map, added_e_p[0]) ):
            #four nodes can be mapped
            new_dis = (new_dis - 6 + dissimilarity_edges(G, P, added_e_o, added_e_p) +
                       dissimilarity_nodes(G, P, added_e_o[0], added_e_p[0]) +
                       dissimilarity_nodes(G, P, added_e_o[1], added_e_p[1]) )
            map_new[added_e_o[0]] = added_e_p[0]
            map_new[added_e_o[1]] = added_e_p[1]
        elif added_e_o[0] in map and map[added_e_o[0]] != added_e_p[0]:
            # only the node can be mapped
            new_dis = (new_dis - 2 + dissimilarity_nodes(G, P, added_n_o, added_n_p))
            map_new[added_n_o] = added_n_p
    if added_n_p == None and added_n_o == None:
        if ((added_e_o[0] in map and map[added_e_o[0]] == added_e_p[0])
                and (added_e_o[1] in map and map[added_e_o[1]] == added_e_p[1])) :
            #The edge can be mapped
            new_dis = new_dis - 2  + dissimilarity_edges(G, P, added_e_o, added_e_p)
        if ((added_e_o[0] in map and map[added_e_o[0]] == added_e_p[1])
                and (added_e_o[1] in map and map[added_e_o[1]] == added_e_p[0])) :
            # The edge can be mapped
            new_dis = new_dis - 2  + dissimilarity_edges(G, P, added_e_o, added_e_p)
    return map_new, new_dis


def expandOcurrence(o,  added_n_p, added_e_p, max_dissimilarity, G, P):
    O = []
    # Ocurrence without chnges
    occ_new = copy_occurrence(o)
    if added_n_p != None:
        occ_new['d'] = occ_new['d'] + 1 #Node added in the pattern
    occ_new['d'] = occ_new['d'] + 1     #Edge added in the pattern
    if occ_new['d'] <= max_dissimilarity :
        O.append(occ_new)

    for e in edge_nneighbors(G, o['g']):
        occ_new = {}
        occ_new['g'], added_n_o, added_e_o = grewPattern(G, o['g'], e)
        occ_new['m'], occ_new['d'] = calculate_mapping (o['m'], o['d'] , added_n_o, added_e_o,  added_n_p, added_e_p, G, P)

        if occ_new['d']<= max_dissimilarity:
            O.append(occ_new)

        #Generate the recursive process
        occ_new = {}
        occ_new['g'], added_n_o, added_e_o = grewPattern(G, o['g'], e)
        occ_new['m'] = o['m'].copy()
        occ_new['d'] = o['d'] + 1
        if added_n_o != None:
            occ_new['d'] = occ_new['d'] + 1

        O_exp = []
        if (occ_new['d'] <= max_dissimilarity):
            O_exp = expandOcurrence(occ_new, added_n_p, added_e_p, max_dissimilarity, G, P)

        for occ_new in O_exp :
            O.append(occ_new)

    return O


def expandPossibility (p, G, P, max_dissimilarity):
    complete_pat_list = []
    isCompletePos = True
    for e in edge_nneighbors(P, p['g']):
        isCompletePos = False
        newp, added_n, added_e = grewPattern(P,p['g'], e)
        O_new  = []
        for o in p['O']:
            for o_new in expandOcurrence(o, added_n, added_e, max_dissimilarity, G, P):
                O_new.append(o_new)
        if len(O_new) > 0:
            new_pos = {'g': newp, 'O': O_new}
            for complete_pat in expandPossibility(new_pos, G, P, max_dissimilarity):
                complete_pat_list.append( complete_pat)

    if isCompletePos:
        complete_pat_list.append(p)

    return  complete_pat_list


def get_occ_one_node(G, P, freq_n):
    O = []
    for n in G.nodes():
        if G.nodes[n]['label'] == P.nodes[freq_n]['label']:
            occ = {}
            occ['g'] = nx.subgraph(G, [n])
            occ['d'] = 0.0
            occ['m'] = {}
            occ['m'][n] = freq_n
            O.append( occ)
    return O



def evaluatePatternOnGraph (P, G, max_dissimilarity, frequent_node = None):
    min_dissimilarity_found = np.inf
    Posibilidades = []
    frequent_nodes = []
    if frequent_node != None:
        frequent_nodes.append(frequent_node)
    else:
        for freq_n in P.nodes():
            frequent_nodes.append(freq_n)

    complete_pat_on_graph = []
    for freq_n in frequent_nodes:
        p = nx.subgraph(P, [freq_n])
        O = get_occ_one_node(G, P, freq_n)
        possible_pat = {'g': p, 'O': O}
        expand_pat = expandPossibility(possible_pat, G, P, max_dissimilarity)
        for complete in expand_pat:
            complete_pat_on_graph.append(complete)

    for c in complete_pat_on_graph:
        for o in c['O']:
            if (o['d'] < min_dissimilarity_found ):
                min_dissimilarity_found = o['d']
    return min_dissimilarity_found
