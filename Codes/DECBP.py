
# coding: utf-8

# In[1]:


# %load DEC.py
import os
import time
import numpy as np
from copy import copy
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import nibabel

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


def save_nifti(file_path, data, mask, affine=None, zooms=[3, 3, 3, 0]):
    if affine is None:
        affine = [[-3, 0, 0, 90],
                  [0, 3, 0, -126],
                  [0, 0, 3, -72],
                  [0, 0, 0, 1]]

    # Set voxels values
    coords = np.where(mask)
    if data.ndim > 1:
        out = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], data.shape[1]))
        temp = np.zeros(mask.shape)
        for i in range(data.shape[1]):
            temp[coords] = data[:, i]
            out[:, :, :, i] = temp
    else:
        out = np.zeros(mask.shape)
        out[coords] = data

    nifti = nibabel.Nifti1Image(out, affine)
    nifti.header.set_data_dtype(np.float64)
    nifti.header.set_zooms(zooms)
    nifti.header.set_qform(affine, 'aligned')
    nifti.to_filename(file_path)

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_true = np.tile(y_true, int(y_pred.shape[0] / y_true.shape[0]))
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).transpose()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def dist_2_label(q_t):
    _, label = torch.max(q_t, dim=1)
    return label.data.cpu().numpy()


def labelchange(label1, label2):
    changes = label1-label2
    return sum(changes != 0)/changes.shape[0]


def interstop(interchanges, inters, threshold):
    index = np.arange(inters)*(-1)-1
    return all(np.array(interchanges)[index] < threshold)


def test_acc(model, data, n_cluster=2):
    model.eval()
    with torch.no_grad():
        feat, q = model(data.to(device))
        p = DEC.target_distribution(q)
        loss = DEC.lossf(q, p)
        pred_label = dist_2_label(q)
        pred_labelall = dist_2_label(q.reshape([-1, n_voxels, n_cluster]).mean(dim=0))

    return loss, pred_label, pred_labelall, feat, q


        
class AE(nn.Module):
    def __init__(self, input_num, output_num):
        super(AE, self).__init__()        
        self.encoder = nn.Sequential(
            nn.Linear(input_num, output_num),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.Linear(output_num, input_num),
            nn.ReLU()
            )
        for m in self.modules():
           if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
               torch.nn.init.xavier_uniform_(m.weight)
               torch.nn.init.constant_(m.bias, 0) 
    def forward(self, x):
        x = self.encoder(x)
        x_ae = x
        x = self.decoder(x)
        x_de = x
        return x_ae, x_de


        
def pretrain(premodel, train, train_loader, name='AE', epochs=10000, inter=1):          
    mseloss = nn.MSELoss().to(device)
    optimizer = optim.Adam(premodel.parameters(), lr=1e-3)
    train = train.to(device)
    ep = []
    train_idx = 0
    train_minloss = 1
    train_loss = []

    for epoch in range(1, epochs+1):
        ep.append(epoch)
        premodel.train()
        for i, data in enumerate(train_loader):
            x = data[0].to(device)
            optimizer.zero_grad()
            x_ae, x_de = premodel(x)
            loss = mseloss(x_de, x)
            loss.backward()
            optimizer.step()
        premodel.eval()
        with torch.no_grad():
            x_ae, x_de = premodel(train)
            train_loss.append(mseloss(x_de, train).item())
            if train_minloss > train_loss[-1]:
                torch.save(premodel.state_dict(), outputpath_p + '/ckpt/' + name + 'FCP_train'+ 'pretrain_ae.pkl')
                train_minloss = train_loss[-1]
                train_idx = copy(epoch)
                embed = x_ae
        if epoch % inter == 0:  # print every inter epochs
            print('====> Epoch: {} Average tran loss: {:.8f}'.
                  format(epoch, train_loss[-1]))


    loss = np.concatenate((ep, train_loss), axis=0).reshape(2, -1)
    np.save(outputpath_p+'/ckpt/PreLoss' + name + 'FCP_train', loss)
    np.save(outputpath_p+'/ckpt/Encoder' + name + 'FCP_train', embed.cpu().numpy())
    print('Min train loss in epoch {}/{} , loss: {}'.format(train_idx, epochs, train_minloss))
    return embed.cpu()

def extract_para(model):
    return [model['encoder.0.weight'],model['encoder.0.bias']],\
        [model['decoder.0.weight'],model['decoder.0.bias']]
        
class DEC_AE(nn.Module):
    def __init__(self, num_classes, num_features, para=None):
        super(DEC_AE, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(n_input, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_features)
        self.relu = nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.fc_d1 = nn.Linear(256, n_input)
        self.fc_d2 = nn.Linear(128, 256)
        self.fc_d3 = nn.Linear(64, 128)
        self.fc_d4 = nn.Linear(num_features, 64)
        self.alpha = 1.0
        self.clusterCenter = nn.Parameter(torch.zeros(num_classes, num_features))
        self.pretrainMode = True
        if para is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)
        else:
            for i in range(4):
                s='self.fc'+str(i+1)+'.weight.data = en_para['+str(i)+'][0].detach().cpu()'
                exec(s)
                s='self.fc'+str(i+1)+'.bias.data = en_para['+str(i)+'][1].detach().cpu()'
                exec(s)  
                s='self.fc_d'+str(i+1)+'.weight.data = de_para['+str(i)+'][0].detach().cpu()'
                exec(s)
                s='self.fc_d'+str(i+1)+'.bias.data = de_para['+str(i)+'][1].detach().cpu()'
                exec(s)                
            
    def setPretrain(self, mode):
        """To set training mode to pretrain or not, 
        so that it can control to run only the Encoder or Encoder+Decoder"""
        self.pretrainMode = mode

    def updateClusterCenter(self, cc):
        """
        To update the cluster center. This is a method for pre-train phase.
        When a center is being provided by kmeans, we need to update it so
        that it is available for further training
        :param cc: the cluster centers to update, size of num_classes x num_features
        """
        self.clusterCenter.data = torch.from_numpy(cc)

    def getTDistribution(self, x, clusterCenter):
        """
        student t-distribution, as same as used in t-SNE algorithm.
         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
         
         :param x: input data, in this context it is encoder output
         :param clusterCenter: the cluster center from kmeans
         """
        xe = torch.unsqueeze(x, 1).to(device) - clusterCenter.to(device)
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe, xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t()  # due to divison, we need to transpose q
        return q

    def forward(self, x):
        # x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x_ae = x
        # if not in pretrain mode, we only need encoder
        if self.pretrainMode == False:
            return x, self.getTDistribution(x, self.clusterCenter)
        # encoder is done, followed by decoder #
        x = self.fc_d4(x)
        x = self.relu(x)
        x = self.fc_d3(x)
        x = self.relu(x)
        x = self.fc_d2(x)
        x = self.relu(x)
        x = self.fc_d1(x)
        x = self.relu(x)
        x_de = x
        return x_ae, x_de


class DEC:
    """The class for controlling the training process of DEC"""

    def __init__(self, n_clusters=2, alpha=1.0, n_features=32):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.alpha = alpha

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        # return torch.tensor((weight.t() / weight.sum(1)).t().data, requires_grad=True)
        # p = (weight.t() / weight.sum(1)).t().data
        # p.requires_grad = True
        return (weight.t() / weight.sum(1)).t().clone().detach().requires_grad_(True)

    def logAccuracy(self, pred, label):
        print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
              % (acc(label, pred), nmi(label, pred, average_method='arithmetic')))

    @staticmethod
    def kld(q, p):
        res = torch.sum(p * torch.log(p / q), dim=-1)
        return res

    @staticmethod
    def lossf(q, p):

        res = 0.5*(torch.nn.functional.kl_div(torch.log(q), (p+q)/2, reduction='batchmean') +
                   torch.nn.functional.kl_div(torch.log(p), (p+q)/2, reduction='batchmean'))
        return res

    def pretrain(self, train, train_loader, test_data, epochs=100, inter=1, para=None):
        dec_ae = DEC_AE(self.n_clusters, self.n_features, para=para).to(device)  # auto encoder
        mseloss = nn.MSELoss().to(device)
        optimizer = optim.Adam(dec_ae.parameters(), lr=1e-3)
        train = train.to(device)
        test_data = test_data.to(device)
        ep = []
        train_idx = 0
        train_minloss = 1
        train_loss = []
        test_idx = 0
        test_minloss = 1
        test_loss = []
        for epoch in range(1, epochs+1):
            ep.append(epoch)
            dec_ae.train()
            for i, data in enumerate(train_loader):
                x = data[0].to(device)
                optimizer.zero_grad()
                x_ae, x_de = dec_ae(x)
                loss = mseloss(x_de, x)
                loss.backward()
                optimizer.step()
            dec_ae.eval()
            with torch.no_grad():
                x_ae, x_de = dec_ae(train)
                train_loss.append(mseloss(x_de, train).item())
                if train_minloss > train_loss[-1]:
                    torch.save(dec_ae, outputpath_p + '/ckpt/'+'FCP_train'+'pretrain_ae.pkl')
                    train_minloss = train_loss[-1]
                    train_idx = copy(epoch)
                x_ae, x_de = dec_ae(test_data)
                test_loss.append(mseloss(x_de, test_data).item())
                if test_minloss > test_loss[-1]:
                    torch.save(dec_ae, outputpath_p + '/ckpt/'+'FCP_test'+'pretest_ae.pkl')
                    test_minloss = test_loss[-1]
                    test_idx = copy(epoch)

            if epoch % inter == 0:  # print every inter epochs
                print('====> Epoch: {} Average tran loss: {:.8f} test loss: {:.8f}'.
                      format(epoch, train_loss[-1], test_loss[-1]))

        loss = np.concatenate((ep, train_loss, test_loss), axis=0).reshape(3, -1)
        np.save(outputpath_p+'/PreLoss' + 'FCP_train' + '+' + 'FCP_test', loss)
        print('Min train loss in epoch {}/{} , loss: {}'.format(train_idx, epochs, train_minloss))
        print('Min test loss in epoch {}/{} , loss: {}'.format(test_idx, epochs, test_minloss))

    def clustering(self, mbk, x, model):
        model.eval()
        y_pred_ae, _ = model(x)
        y_pred_ae = y_pred_ae.data.cpu().numpy()
        y_pred = mbk.fit_predict(y_pred_ae)  # seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_  # keep the cluster centers
        model.updateClusterCenter(self.cluster_centers)
        return y_pred

    def train(self, train, batch_size=1190*5, epochs=10, inter=1, inters=3, stopthreshold=0.001):
        """This method will start training for DEC cluster"""
        model = torch.load(outputpath_p + '/ckpt/'+'FCP_train'+'pretrain_ae.pkl').to(device)
        model.setPretrain(False)
        optimizer = optim.SGD([{'params': model.parameters()}, ], lr=0.01, momentum=0.9)
        print('Initializing cluster center with pre-trained weights')
        mbk = KMeans(n_clusters=self.n_clusters, n_init=100)
        # kmeans init
        with torch.no_grad():
            model.eval()
            feature_pred, _ = model(train.to(device))
        y_pred = mbk.fit_predict(feature_pred.data.cpu().numpy().reshape(-1, n_voxels, self.n_features).mean(axis=0))
        y_pred = mbk.predict(feature_pred.data.cpu().numpy())
        cluster_centers = mbk.cluster_centers_
        model.updateClusterCenter(cluster_centers)
        with torch.no_grad():
            model.eval()
            feature_pred, q_ini = model(train.to(device))


        y_predgroup = y_pred.reshape([-1, n_voxels])
        y_predgroup = stats.mode(y_predgroup)[0][0]
        feat0 = feature_pred

        ep, train_loss, interchanges= [], [], []
        train_label = np.ones((3, n_voxels), dtype=np.int64)*(-1)
        train_label[0] = y_predgroup
        train_minloss = 1
        labelold = y_pred
        
        eta_list = train.reshape(-1, n_voxels, n_input)
        randindex = np.arange(eta_list.shape[0])
        np.random.shuffle(randindex)
        eta_list = eta_list[randindex].reshape(-1, n_input)

        train_data = Data.TensorDataset(eta_list)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=int(batch_size/n_voxels), shuffle=True)

        for epoch in range(1, epochs+1):
            for i, data in enumerate(train_loader):
                x = data[0].reshape(-1, n_input).to(device)
                optimizer.zero_grad()
                model.train()
                # now we start training with acquired cluster center
                feature_pred, q = model(x)
                # get target distribution
                p = self.target_distribution(q)
                loss = self.lossf(q, p)
                loss.backward()
                optimizer.step()

            # Evaluate model
            ep.append(epoch)
            model.eval()
            # Train data
            loss, pred_label, pred_labelall, feature_pred, Q = test_acc(model, train, n_cluster=self.n_clusters)
            interchanges.append(labelchange(labelold, pred_label))
            labelold = pred_label
            train_loss.append(loss.item());
            if train_minloss >= loss:
                train_minloss = loss
                train_label[1] = pred_labelall
                Losslabel = pred_label
                feat1 = feature_pred
                proq_minloss = Q.data.cpu().numpy()
                torch.save(model, outputpath + '/ckpt/'+'FCP_train'+'DECtrain_minloss.pkl')


            # Visual
            if epoch % inter == 0:
                print('========> Epoch: {} Loss:{:.8f}\t  Interchanges: {:.4f}%'
                      .format(epoch, loss, interchanges[-1]*100))


            # labels changed less than "stopthreshold" for consecutive "inters" iterations
            if epoch >= inters:
                if interstop(interchanges, inters, stopthreshold):
                    torch.save(model, outputpath + '/ckpt/' + 'FCP_train' + 'DECtrain_nochanges.pkl')
                    train_label[2] = pred_labelall
                    feat2 = feature_pred
                    break
                elif epoch == epochs:
                    torch.save(model, outputpath + '/ckpt/' + 'FCP_train' + 'DECtrain_nochanges.pkl')
                    train_label[2] = pred_labelall
                    feat2 = feature_pred
                    print('The epochs are more than maximum iterations')




        Labels = np.concatenate((y_pred, Losslabel, pred_label), axis=0).reshape(3, -1)
        Labels[Labels == 0] = self.n_clusters
        train_label[train_label == 0] = self.n_clusters

        np.save(outputpath + '/' + 'FCP_train'+'Q', np.array([q_ini.data.cpu().numpy(), proq_minloss,
                                                       Q.data.cpu().numpy()]))
        np.save(outputpath + '/Label' + 'FCP_train', Labels)
        np.save(outputpath + '/GroupLabel' + 'FCP_train', train_label)
        np.save(outputpath + '/Labelchanges' + 'FCP_train', np.array(interchanges))
        np.save(outputpath + '/dae_middletrain.npy',
                np.array([feat0.data.cpu().numpy(), feat1.data.cpu().numpy(),
                          feat2.data.cpu().numpy()]))


        idx = [k for k in range(len(train_loss)) if train_loss[k] == min(train_loss)][-1]
        print('Min trainloss in epoch {}/{}, loss: {}'.format(idx+1, epochs, min(train_loss)))

        print('Min changes in epoch {}/{}, loss: {}, final changes: {:.4f}%'.
              format(epoch, epochs, train_loss[-1], labelchange(y_pred, pred_label)*100))


def main(key1, key2, batch_size=1190):
    eta2_list = np.load(path + key1).astype(np.float32)
    eta2 = torch.from_numpy(eta2_list).view(-1, n_input)
    train_data = Data.TensorDataset(eta2)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    eta2_list = np.load(path + key2).astype(np.float32)
    test_data = torch.from_numpy(eta2_list).view(-1, n_input)
    return eta2, train_loader, test_data



if __name__ == '__main__':
    Cudauser = torch.cuda.is_available()
    device = torch.device("cuda:0")
    path = './'

    traindata = 'eta_putamen_caudateFCPtrainhighpassleft.npy'   #train dataset numpy format subject*voxle*feature
    testdata = 'eta_putamen_caudateFCPtesthighpassleft.npy'    #test dataset, could be same as train dataset due to the unsupervised learning

    n_voxels = 584  #the number of voxels
    n_input = 584  #the dimension of features for first layer
    n_features = 32  #the dimension of latent features, this network n_input>256>158>64>n_features
    alpha = 1.0 #the parameters for target distribution, default 1.0
    batchsz = n_voxels*5  #batch size, suggest the multiple n_voxels
    outputpath_p = path + 'DECBP'
    if not os.path.exists(outputpath_p+'/ckpt'):
        os.makedirs(outputpath_p+'/ckpt')
          
    #greedy pretrain
    eta, train_loader, test_data = main(traindata, testdata, batch_size=batchsz)

    premodel = AE(n_input, 256).to(device)
    encoders = pretrain(premodel, eta, train_loader,name='Lay1',epochs=5000)

    train_data = Data.TensorDataset(encoders)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True)
    premodel = AE(256, 128).to(device)
    encoders = pretrain(premodel, encoders, train_loader,name='Lay2',epochs=5000)

    train_data = Data.TensorDataset(encoders)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True)
    premodel = AE(128, 64).to(device)
    encoders = pretrain(premodel, encoders, train_loader,name='Lay3',epochs=3000)

    train_data = Data.TensorDataset(encoders)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batchsz, shuffle=True)
    premodel = AE(64, n_features).to(device)
    encoders = pretrain(premodel, encoders, train_loader,name='Lay4',epochs=3000)


    #pretrain
    #initialization parameters
    en_para , de_para = [], []
    for i in range(1, 5):
        premodel = torch.load(outputpath_p + '/ckpt/Lay' + str(i) + 'FCP_train'+ 'pretrain_ae.pkl')
        p1, p2 = extract_para(premodel)
        en_para.append(p1); de_para.append(p2)

    eta, train_loader, test_data = main(traindata, testdata, batch_size=batchsz)
    dec = DEC(alpha=alpha, n_features=n_features)
    Prestart = time.perf_counter()
    dec.pretrain(eta, train_loader, test_data, epochs=20000, para=True)
    Preend = time.perf_counter(); Pretime = Preend-Prestart


    #train
    eta, _, test_data = main(traindata, testdata, batch_size=batchsz)
    Trainstart = time.perf_counter()
    brain = nibabel.load(path + "FCROI_1_H001.nii")
    brain = np.array(brain.get_fdata())
    roi_mask = np.zeros(brain.shape, dtype=bool)
    roi_mask[brain == 71] = True  # caudate_L
    # roi_mask[brain == 72] = True  # caudate_R
    roi_mask[brain == 73] = True  # putamen_L
    # roi_mask[brain == 74] = True  # putamen_R

    for n_clusters in range(6, 10):
        outputpath = path + 'DECBP/Relu' + str(n_clusters) + 'cluster'
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
            os.makedirs(outputpath+'/ckpt')
        dec = DEC(n_clusters, alpha=alpha, n_features=n_features)
        if n_clusters < 10:
            t = 0.0001
        else:
            t = (n_clusters-1) // 5 * 5 * 0.0001
        dec.train(eta, batch_size=batchsz, epochs=5, inters=2, stopthreshold=t)

        feat = np.load(outputpath + '/GroupLabel' + 'FCP_train' + '.npy')
        save_nifti(outputpath + "/GroupLabelFCP_train.nii", feat.transpose(), roi_mask)
    Trainend = time.perf_counter(); Traintime = Trainend-Trainstart

    print('Train run time: {:.4f}'.format(Traintime/3600))

    # obtain the parcellations results in testing dataset
    path = "./"
    key = 'eta_putamen_caudateFCPtesthighpassleft.npy'    #test dataset
    brain = nibabel.load(path + "FCROI_1_H001.nii")
    brain = np.array(brain.get_fdata())
    roi_mask = np.zeros(brain.shape, dtype=bool)
    roi_mask[brain == 71] = True  # caudate_L
    roi_mask[brain == 73] = True  # caudate_R

    data = np.load(path + key).astype(np.float32)
    device = torch.device("cuda:0")
    for i in range(6, 10):
        modelpath = path + 'DECBP/Relu'+str(i)+'cluster'+'/ckpt/FCP_trainDECtrain_nochanges.pkl'
        model = torch.load(modelpath).to(device)
        model.setPretrain(False)
        model.eval()
        x = torch.from_numpy(data.reshape(-1, data.shape[-1]))
        with torch.no_grad():
            _, q = model(x.to(device))
            q = q.reshape([-1, data.shape[-1], i])
            pro, pred = torch.max(q.mean(dim=0), dim=1)
            pred = pred.data.cpu().numpy()
            pro = pro.data.cpu().numpy()
        pred[pred == 0] = pred.max()+1
        q = q.mean(dim=0).data.cpu().numpy()
        outputpath = path + 'DECBP/Relu'+str(i)+'cluster/ProResultscluster'+ str(i)
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)

        save_nifti(outputpath + '/parcels.nii', pred, roi_mask, zooms=[3, 3, 3])
        save_nifti(outputpath + '/parcelspro.nii', q, roi_mask)
        # for j in range(q.shape[-1]):
        #     save_nifti(outputpath + '/glabelpro' + str(j+1) + '.nii', q[:, j], roi_mask, zooms=[3, 3, 3])

