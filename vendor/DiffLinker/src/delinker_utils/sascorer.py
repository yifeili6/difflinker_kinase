from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.six.moves import cPickle
from rdkit.six import iteritems
import math
from collections import defaultdict
import os.path as op
_fscores = None

def readFragmentScores(name='resources/fpscores'):
    import gzip
    global _fscores
    if name == 'fpscores':
        name = op.join(op.dirname(__file__), name)
    _fscores = cPickle.load(gzip.open('%s.pkl.gz' % name))
    outDict = {}
    for i in _fscores:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return (nBridgehead, nSpiro)

def calculateScore(m):
    if _fscores is None:
        readFragmentScores()
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for (bitId, v) in iteritems(fps):
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    (nBridgeheads, nSpiro) = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1
    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)
    score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5
    sascore = score1 + score2 + score3
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0
    return sascore

def processMols(mols):
    print('smiles\tName\tsa_score')
    for (i, m) in enumerate(mols):
        if m is None:
            continue
        s = calculateScore(m)
        smiles = Chem.MolToSmiles(m)
        print(smiles + '\t' + m.GetProp('_Name') + '\t%3f' % s)
if __name__ == '__main__':
    import sys, time
    t1 = time.time()
    readFragmentScores('fpscores')
    t2 = time.time()
    suppl = Chem.SmilesMolSupplier(sys.argv[1])
    t3 = time.time()
    processMols(suppl)
    t4 = time.time()
    print('Reading took %.2f seconds. Calculating took %.2f seconds' % (t2 - t1, t4 - t3), file=sys.stderr)
