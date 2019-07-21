import csv
from collections import defaultdict
import pydotplus
from cart import CART


def dotgraph(decisionTree, dcHeadings):
    dcNodes = defaultdict(list)

    def toString(iSplit, decisionTree, bBranch, szParent="null", indent=''):
        if decisionTree.results is not None:  # leaf node
            lsY = []
            for szX, n in decisionTree.results.items():
                lsY.append('%s:%d' % (szX, n))
            dcY = {"name": "%s" % ', '.join(lsY), "parent": szParent}
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append([
                'leaf', dcY['name'], szParent, bBranch, dcSummary['impurity'],
                dcSummary['samples']
            ])
            return dcY
        else:
            szCol = 'Column %s' % decisionTree.col
            if szCol in dcHeadings:
                szCol = dcHeadings[szCol]
            if isinstance(decisionTree.value, int) or isinstance(
                    decisionTree.value, float):
                decision = '%s >= %s' % (szCol, decisionTree.value)
            else:
                decision = '%s == %s' % (szCol, decisionTree.value)
            toString(iSplit + 1, decisionTree.trueBranch, True, decision,
                     indent + '\t\t')
            toString(iSplit + 1, decisionTree.falseBranch, False, decision,
                     indent + '\t\t')
            dcSummary = decisionTree.summary
            dcNodes[iSplit].append([
                iSplit + 1, decision, szParent, bBranch, dcSummary['impurity'],
                dcSummary['samples']
            ])
            return

    toString(0, decisionTree, None)
    lsDot = [
        'digraph Tree {',
        'node [shape=box, style="filled, rounded", color="black", fontname=helvetica] ;',
        'edge [fontname=helvetica] ;'
    ]
    i_node = 0
    dcParent = {}
    dcNodes = sorted(dcNodes.items(), key=lambda obj: obj[0])
    for nSplit, lsY in dcNodes:
        # print('-----')
        # print(nSplit, lsY)
        for lsX in lsY:
            iSplit, decision, szParent, bBranch, szImpurity, szSamples = lsX

            if type(iSplit) == int:
                szSplit = '%d-%s' % (iSplit, decision)
                dcParent[szSplit] = i_node
                lsDot.append(
                    '%d [label=<%s<br/>impurity %s<br/>samples %s>, fillcolor="#e5813900"] ;'
                    % (i_node, decision.replace('>=', '&ge;').replace(
                        '?', ''), szImpurity, szSamples))
            else:
                lsDot.append(
                    '%d [label=<impurity %s<br/>samples %s<br/>class %s>, fillcolor="#e5813900"] ;'
                    % (i_node, szImpurity, szSamples, decision))

            if szParent != 'null':
                if bBranch:
                    szAngle = '45'
                    szHeadLabel = 'True'
                else:
                    szAngle = '-45'
                    szHeadLabel = 'False'
                szSplit = '%d-%s' % (nSplit, szParent)

                p_node = dcParent[szSplit]
                if nSplit == 1:
                    lsDot.append(
                        '%d -> %d [labeldistance=2.5, labelangle=%s, headlabel="%s"] ;'
                        % (p_node, i_node, szAngle, szHeadLabel))
                else:
                    lsDot.append('%d -> %d ;' % (p_node, i_node))
            i_node += 1
    lsDot.append('}')
    dot_data = '\n'.join(lsDot)
    return dot_data


def loadCSV(file):
    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt'))
    dcHeader = {}
    bHeader = True
    if bHeader:
        lsHeader = next(reader)
        for i, szY in enumerate(lsHeader):
            szCol = 'Column %d' % i
            dcHeader[szCol] = str(szY)
    return dcHeader, [[convertTypes(item) for item in row] for row in reader]


if __name__ == "__main__":

    dcHeadings, trainingData = loadCSV('fishiris.csv')
    # print(trainingData[0][0])
    miniGain = 0.05
    clf = CART(miniGain)
    clf.buildDecisionTree(trainingData)

    dot_data = dotgraph(clf.decisionTree, dcHeadings)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png("tree{}.png".format(miniGain))
