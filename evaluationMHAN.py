from hitsScore import hits_score
from MHAN import MHAN
from dataloader import DataLoader
from ndcgScore import ndcg_score
import dgl
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def constructDGLGraph(dl, similarity=1, isRandomFeature=False):
    rawGraph = dl.generateRawTrainGraph()
    trailGraph = dl.generateTrailGraph(similarity)
    hetero_graph = dgl.heterograph({
        ('NCT', 'hasProblem', 'Problem'): (rawGraph['NCTP'], rawGraph['Problem']), 
        ('NCT', 'hasIntervention', 'Intervention'): (rawGraph['NCTI'], rawGraph['Intervention']),
        ('NCT', 'hasOutcome', 'Outcome'): (rawGraph['NCTO'], rawGraph['Outcome']),
        ('Problem', 'hasNCT1', 'NCT'): (rawGraph['Problem'], rawGraph['NCTP']),  
        ('Intervention', 'hasNCT2', 'NCT'): (rawGraph['Intervention'], rawGraph['NCTI']),
        ('Outcome', 'hasNCT3', 'NCT'): (rawGraph['Outcome'], rawGraph['NCTO']),
        ('NCT', 'similar', 'NCT'): (trailGraph['sourceNCT'], trailGraph['targetNCT'])
    })
    if isRandomFeature:
        num = dl.get_num()
        hetero_graph.nodes['Problem'].data['feature'] = torch.randn(num['num_P'], 768)
        hetero_graph.nodes['NCT'].data['feature'] = torch.randn(num['num_NCT'], 768)
        hetero_graph.nodes['Intervention'].data['feature'] = torch.randn(num['num_I'], 768)
        hetero_graph.nodes['Outcome'].data['feature'] = torch.randn(num['num_O'], 768)
    else:
        feature = dl.loadAttributeEmb('./AttributeEmbedding.pt')
        hetero_graph.nodes['Problem'].data['feature'] = feature['ProblemFeature']
        hetero_graph.nodes['NCT'].data['feature'] = feature['NCTFeature']
        hetero_graph.nodes['Intervention'].data['feature'] = feature['InterventionFeature']
        hetero_graph.nodes['Outcome'].data['feature'] = feature['OutcomeFeature']
    print(hetero_graph)
    return hetero_graph


def construct_NCT_DGLGrpah(dl, similarity, isRandomFeature):
    trailGraph = dl.generateTrailGraph(similarity)
    graph = dgl.graph((trailGraph['sourceNCT'], trailGraph['targetNCT']))
    graph = dgl.add_self_loop(graph)
    if isRandomFeature:
        num = dl.get_num()
        graph.ndata['feature'] = torch.randn(num['num_NCT'], 768)
    else:
        feature = dl.loadAttributeEmb('./AttributeEmbedding.pt')
        graph.ndata['feature'] = feature['NCTFeature']
    print(graph)
    return graph


def compute_loss(pos_score, neg_score):
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def calculateEdgeScore(src, dst):
    matrix = np.zeros((len(src), len(dst)), dtype=float)
    for i in range(len(src)):
        for j in range(len(dst)):
            matrix[i][j] = sigmoid(src[i].dot(dst[j]))
    return matrix


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)
    g = dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})
    g = g.to(device)
    return g


def adjacencyMatrix(src, dst, c, l):
    if len(src) != len(dst):
        print('wrong')
        exit(1)
    matrix = np.zeros((c, l), dtype=int)
    for i in range(len(src)):
        matrix[src[i]][dst[i]] = 1
    return matrix


def toZeroTrainEdge(dl, matrix):
    train = dl.generateRawTrainGraph()
    for i in range(len(train['NCTP'])):
        matrix[train['Problem'], train['NCTP']] = 0.0
    return matrix


def evaluator(model, dl, best_roc_auc, best_ndcg, best_hr):
    with torch.no_grad():
        model.eval()
        node_embeddings = model.sage(hetero_graph)
        matrix = calculateEdgeScore(node_embeddings['Problem'].detach().cpu().numpy(), node_embeddings['NCT'].detach().cpu().numpy())
        PreMatrix = toZeroTrainEdge(dl, matrix)
        testSet = dl.generateTestData()
        num = dl.get_num()
        LabelMetrix = adjacencyMatrix(testSet['testProblem'], testSet['testNCTP'], num['num_P'], num['num_NCT'])
        # AUC
        fpr, tpr, thersholds = roc_curve(LabelMetrix.flatten(), PreMatrix.flatten())
        roc_auc = auc(fpr, tpr)
        print("AUC", roc_auc)
        #NDCG
        ndcg_1 = ndcg_score(PreMatrix, LabelMetrix, k=1)
        ndcg_3 = ndcg_score(PreMatrix, LabelMetrix, k=3)
        ndcg_5 = ndcg_score(PreMatrix, LabelMetrix, k=5)
        ndcg_7 = ndcg_score(PreMatrix, LabelMetrix, k=7)
        ndcg_9 = ndcg_score(PreMatrix, LabelMetrix, k=9)
        print("NDCG@1", ndcg_1, "NDCG@3", ndcg_3, "NDCG@5", ndcg_5, "NDCG@7", ndcg_7, "NDCG@9", ndcg_9)
        #HR
        hr_1 = hits_score(PreMatrix, LabelMetrix, k=1)
        hr_3 = hits_score(PreMatrix, LabelMetrix, k=3)
        hr_5 = hits_score(PreMatrix, LabelMetrix, k=5)
        hr_7 = hits_score(PreMatrix, LabelMetrix, k=7)
        hr_9 = hits_score(PreMatrix, LabelMetrix, k=9)
        print("HR@1", hr_1, "HR@3", hr_3, "HR@5", hr_5, "HR@7", hr_7, "HR@9", hr_9)

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
        if ndcg_1 > best_ndcg[0]:
            best_ndcg[0] = ndcg_1
        if ndcg_3 > best_ndcg[1]:
            best_ndcg[1] = ndcg_3
        if ndcg_5 > best_ndcg[2]:
            best_ndcg[2] = ndcg_5
        if ndcg_7 > best_ndcg[3]:
            best_ndcg[3] = ndcg_7
        if ndcg_9 > best_ndcg[4]:
            best_ndcg[4] = ndcg_9
        if hr_1 > best_hr[0]:
            best_hr[0] = hr_1
        if hr_3 > best_hr[1]:
            best_hr[1] = hr_3
        if hr_5 > best_hr[2]:
            best_hr[2] = hr_5
        if hr_7 > best_hr[3]:
            best_hr[3] = hr_7
        if hr_9 > best_hr[4]:
            best_hr[4] = hr_9
        return best_roc_auc, best_ndcg, best_hr


def trainModel(model, hetero_graph, NCT_graph, epoch, lr, dl):
    k = 5
    
    NCT_feats = hetero_graph.nodes['NCT'].data['feature']
    Problem_feats = hetero_graph.nodes['Problem'].data['feature']
    Intervention_feats = hetero_graph.nodes['Intervention'].data['feature']
    Outcome_feats = hetero_graph.nodes['Outcome'].data['feature']
    node_features = {'NCT': NCT_feats, 'Problem': Problem_feats, 'Intervention': Intervention_feats,
                     'Outcome': Outcome_feats}

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_roc_auc = 0
    best_ndcg = [0, 0, 0, 0, 0]
    best_hr = [0, 0, 0, 0, 0]
    for i in range(epoch):
        model.train()
        opt.zero_grad()
        negative_graph = construct_negative_graph(hetero_graph, k, ('NCT', 'hasProblem', 'Problem'))
        pos_score, neg_score = model(hetero_graph, NCT_graph, negative_graph, ('NCT', 'hasProblem', 'Problem'))
        loss = compute_loss(pos_score, neg_score)
        loss.backward()
        opt.step()
        print(loss.item())

        best_roc_auc, best_ndcg, best_hr = evaluator(model, dl, best_roc_auc, best_ndcg, best_hr)
    print("best NDCG", best_ndcg)
    print("best_HR", best_hr)


if __name__ == '__main__':
    setup_seed(2022)
    similarity = 0.8
    in_feats = 768
    hid_feats = 256
    out_feats = 128
    epoch = 500
    n_layers = 2
    n_heads = 4
    n_fuse_heads = 16
    lr = 1e-3
    isRandomFeature = False

    dl = DataLoader()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    hetero_graph = constructDGLGraph(dl, isRandomFeature=isRandomFeature)
    hetero_graph = hetero_graph.to(device)
    NCT_graph = construct_NCT_DGLGrpah(dl, similarity=similarity, isRandomFeature=isRandomFeature)
    NCT_graph = NCT_graph.to(device)

    model = MHAN(hetero_graph, in_feats, hid_feats, out_feats, n_layers=n_layers, n_heads=n_heads, n_fuse_heads=n_fuse_heads)
    model = model.to(device)
    trainModel(model, hetero_graph, NCT_graph, epoch, lr, dl)


