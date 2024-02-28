# -*- coding: utf-8 -*-
# @Time    : 2023/8/19 21:18
# @Author  :
# @File    : train.py
# @Software: PyCharm
import pandas as pd
from my_model import my_model
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn import metrics

positive_sample = pd.read_csv("./data/mask_positivate.csv")
nevigate_sample = pd.read_csv("./data/mask_nevigate.csv")
recall_sample = pd.read_csv("./data/recall_sample.csv")
feature_vector_data = pd.read_csv("./data/feature_vector.csv",index_col=0)

#### Divide all positive samples into five groups for the 5-fold cross verification
positive_num = positive_sample.shape[0]
np.random.seed(111)
positive_sort = np.random.choice(positive_num, positive_num, replace=False) # shuffle
len_single_loop = positive_num // 2

nevigate_num = nevigate_sample.shape[0] # The number of negative samples
# recall_num = recall_sample.shape[0]     # 召回样本的数量

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_node = feature_vector_data.shape[1] * 2
hid1_node = 1500
hid2_node = 800
hid3_node = 300
out_node = 1



batchSize = 2000    # Too much data during training. Train in batches
loop_train = 1
pre_nevigate = []


df_no_choose_save = None
no_choose_nevigate_sort = None
save_roc_auc_score = []
save_recall = []

# # 生成召回率样本的特征值
# recall_disease_sample = recall_sample["disease"].values.reshape(-1)
# recall_miRNA_sample = recall_sample['miRNA'].values.reshape(-1)
# recall_disease = feature_vector_data.loc[recall_disease_sample,:]
# recall_miRNA = feature_vector_data.loc[recall_miRNA_sample,:]
#
# recall_disease.reset_index(drop=True, inplace=True)
# recall_miRNA.reset_index(drop=True, inplace=True)
# df_recall= pd.concat([recall_disease,recall_miRNA],axis=1)
# x_recall = df_recall.values
#
#
# save_recall = []
for model_loop in tqdm(range(5),desc='model_loop'):
    for train_group in range(5):
        net = my_model(input_node, hid1_node, hid2_node, hid3_node, out_node)
        net.to(device)
        criterion = nn.BCELoss()
        criterion.to(device)
        optimzer = optim.Adam(net.parameters(), lr=0.0001)

        # Select the samples for training and testing from the positive and negative samples
        test_positive_sample_sort = positive_sort[len_single_loop * train_group : len_single_loop * (train_group + 1)]
        train_positive_sample_sort = [p for p in positive_sort if p not in test_positive_sample_sort]
        test_positive_sample = positive_sample.iloc[test_positive_sample_sort,:]
        train_positive_sample = positive_sample.iloc[train_positive_sample_sort,:]
        #
        if model_loop == 0 :  # Used for two-step calculation
            nevigate_sample_sort = np.random.choice(nevigate_num,10*positive_num-recall_num,replace=False)
        elif model_loop != 0 and train_group == 0:  # Set train_group == 0 5-fold cross validation process, negative samples are not filtered
            no_choose_nevigate_sample_sort = np.random.choice(df_no_choose_save.shape[0], 10*positive_num -
                                                    recall_num, replace=False)
            df_no_choose_name = df_no_choose_save.iloc[no_choose_nevigate_sample_sort,:]    # 从未训练的负样本中选取
            nevigate_sample_sort = []
            for i, row in df_no_choose_name.iterrows():
                index = nevigate_sample[
                    (nevigate_sample['disease'] == row['disease']) & (nevigate_sample['miRNA'] == row['miRNA'])].index
                nevigate_sample_sort.extend(index)

        # nevigate_sample for negative samples, disease and miRNA pairs
        test_nevigate_sample_sort = nevigate_sample_sort[len_single_loop*train_group:len_single_loop*(train_group+1)]
        if train_group == 0:
            train_nevigate_sample_sort = nevigate_sample_sort[len_single_loop:]
        else:
            train_nevigate_sample_sort = np.append(nevigate_sample_sort[:len_single_loop * train_group],
                                                   nevigate_sample_sort[len_single_loop * (train_group + 1):])
        train_nevigate_sample = None
        test_nevigate_sample = None
        if model_loop == 0:
            test_nevigate_sample = nevigate_sample.iloc[test_nevigate_sample_sort,:]
            train_nevigate_sample = nevigate_sample.iloc[train_nevigate_sample_sort,:]
        else:
            df_choose_new = df_no_choose_save.iloc[:positive_num-recall_num,:2]
            shuffled_df_choose_new = df_choose_new.sample(frac=1)
            test_nevigate_sample = shuffled_df_choose_new.iloc[:len_single_loop,:]
            train_nevigate_sample = shuffled_df_choose_new.iloc[len_single_loop:,:]
        train_nevigate_sample = pd.concat([train_nevigate_sample, recall_sample])
        test_label = [1 for _ in range(test_positive_sample.shape[0])] + [0 for _ in range(test_nevigate_sample.shape[0])]
        train_label = [1 for _ in range(train_positive_sample.shape[0])] + [0 for _ in range(train_nevigate_sample.shape[0])]
        test_sample = pd.concat([test_positive_sample,test_nevigate_sample])
        save_test_sample = test_sample.copy()
        train_sample = pd.concat([train_positive_sample,train_nevigate_sample])
        test_sample = test_sample.values.tolist()
        train_sample = train_sample.values.tolist()
        combined = list(zip(train_sample, train_label))
        np.random.shuffle(combined)
        # Unzip and get the scrambled list(解压缩并得到打乱后的列表)
        train_sample, train_label = zip(*combined)
        train_disease_sample = [connect_group[0] for connect_group in train_sample]
        train_miRNA_sample = [connect_group[1] for connect_group in train_sample]
        df_train_disease = feature_vector_data.loc[train_disease_sample,:]
        df_train_miRNA = feature_vector_data.loc[train_miRNA_sample,:]

        df_train_disease.reset_index(drop=True, inplace=True)
        df_train_miRNA.reset_index(drop=True, inplace=True)
        df_train = pd.concat([df_train_disease,df_train_miRNA],axis=1)

        test_disease_sample = [connect_group[0] for connect_group in test_sample]
        test_miRNA_sample = [connect_group[1] for connect_group in test_sample]
        df_test_disease = feature_vector_data.loc[test_disease_sample, :]
        df_test_miRNA = feature_vector_data.loc[test_miRNA_sample, :]
        df_test_disease.reset_index(drop=True, inplace=True)    # Reset tag
        df_test_miRNA.reset_index(drop=True, inplace=True)
        df_test = pd.concat([df_test_disease, df_test_miRNA], axis=1)
        for epoch in tqdm(range(loop_train),desc='loop_train'):
            batch_train = int((train_positive_sample.shape[0] + train_nevigate_sample.shape[0]) / batchSize)
            save_loss = []
            if batch_train != 0:
                for batch in range(batch_train):
                    p1 = batch * batchSize
                    p2 = (batch + 1) * batchSize
                    x = df_train.iloc[p1:p2,:]
                    X = torch.from_numpy(x.values).float().to(device)

                    y = train_label[p1:p2]
                    y = np.array(y)

                    Y = torch.from_numpy(y.reshape(len(y),1)).float().to(device)
                    y_pred = net(X)
                    loss = criterion(y_pred,Y)
                    print(f"epoch:{epoch},batch:{batch},loss{loss.item()}")
                    save_loss.append(loss.item())
                    optimzer.zero_grad()
                    loss.backward()
                    optimzer.step()
                print("epoch:",epoch,"mean_loss:",np.mean(save_loss))
            else:
                x = df_train
                X = torch.from_numpy(x.values).float().to(device)

                y = train_label
                y = np.array(y)

                Y = torch.from_numpy(y.reshape(len(y), 1)).float().to(device)
                y_pred = net(X)
                loss = criterion(y_pred, Y)
                optimzer.zero_grad()
                loss.backward()
                optimzer.step()
                print("only_1_train:", 0, "mean_loss:", loss.item())

            x_test = df_test
            X_test = torch.from_numpy(x_test.values).float().to(device)
            y_test = test_label

            y = np.array(y)
            Y = torch.from_numpy(y.reshape(len(y), 1)).float().to(device)

            y_pre_test = net(X_test)
            y_score = y_pre_test.reshape(-1)
            y_pre_0_1 = np.where(y_score > 0.5, 1, 0)   # Set the threshold to 1 if it is greater than 0.5 and 0 if it is not
            acc =  metrics.accuracy_score(y_test, y_pre_0_1)

            print(f"-----两步法：{model_loop}---五折：{train_group}-训练轮数：{epoch}--准确率：{acc}--------------")
            save_roc_auc_score.append([model_loop,train_group,epoch, acc])

        X_recall = torch.from_numpy(x_recall).float().to(device)
        y_pre_recall = net(X_recall)
        y_recall = y_pre_recall.reshape(-1).tolist()
        y_recall_positivate = [i for i in y_recall if i>0.5]
        recall_cal = len(y_recall_positivate) / len(y_recall)
        save_recall.append([model_loop,train_group,recall_cal])


        y_score = y_score.tolist()
        test_pre_label = pd.DataFrame({"pre":y_score,"label":test_label})
        save_test_sample.reset_index(drop=True, inplace=True)  # Reset tag
        test_pre_label.reset_index(drop=True, inplace=True)
        save_test_pre_label = pd.concat([save_test_sample,test_pre_label],axis=1)
        save_test_pre_label.to_csv(f"./data/result_pre_test/{model_loop}_{train_group}.csv")
        # Delete the previous data
        del df_train
        del x_test
        del df_train_miRNA
        del df_train_disease
        del X_test,df_test_disease,df_test_miRNA

        # Calculates the predicted value of the negative sample that is not selected
        if train_group == 0:  # Set train_group == 0 5-fold cross validation process, negative samples are not filtered
            no_choose_nevigate_sort = [i for i in range(nevigate_sample.shape[0]) if i not in nevigate_sample_sort]
            no_choose_nevigate_sample = nevigate_sample.iloc[no_choose_nevigate_sort,:]
        loop_num = 4000
        no_choose_loop = int(no_choose_nevigate_sample.shape[0] // loop_num)
        is_Divisible = no_choose_nevigate_sample.shape[0] % loop_num    # The part that is not divisible
        pre_nevigate = []
        for all_nevigate in tqdm(range(no_choose_loop),desc='no_choose_loop'):  # Due to the large number of negative samples to predict, the predicted values need to be calculated in batches
            no_choose_disease = no_choose_nevigate_sample.iloc[all_nevigate*loop_num:(all_nevigate+1)*loop_num,0]
            no_choose_miRNA = no_choose_nevigate_sample.iloc[all_nevigate*loop_num:(all_nevigate+1)*loop_num,1]
            df_no_choose_disease = feature_vector_data.loc[no_choose_disease, :]
            df_no_choose_miRNA = feature_vector_data.loc[no_choose_miRNA, :]
            df_no_choose_disease.reset_index(drop=True, inplace=True)  # Reset tag
            df_no_choose_miRNA.reset_index(drop=True, inplace=True)
            df_no_choose = pd.concat([df_no_choose_disease, df_no_choose_miRNA], axis=1)

            x_no_choose = torch.from_numpy(df_no_choose.values).float().to(device)

            pre_no_choose = net(x_no_choose)
            pre_nevigate.extend(pre_no_choose.reshape(-1).tolist())
        if is_Divisible != 0:   # Whether all unselected negative samples can be evenly divided by loop_num
            no_choose_disease = no_choose_nevigate_sample.iloc[no_choose_loop * loop_num:, 0]
            no_choose_miRNA = no_choose_nevigate_sample.iloc[no_choose_loop * loop_num:, 1]
            df_no_choose_disease = feature_vector_data.loc[no_choose_disease, :]
            df_no_choose_miRNA = feature_vector_data.loc[no_choose_miRNA, :]
            df_no_choose_disease.reset_index(drop=True, inplace=True)  # Reset tag
            df_no_choose_miRNA.reset_index(drop=True, inplace=True)
            df_no_choose = pd.concat([df_no_choose_disease, df_no_choose_miRNA], axis=1)
            x_no_choose = torch.from_numpy(df_no_choose.values).float().to(device)

            pre_no_choose = net(x_no_choose)
            pre_nevigate.extend(pre_no_choose.reshape(-1).tolist())
        df_no_choose_pre = pd.DataFrame({"pre":pre_nevigate})
        index_range = [i for i in range(df_no_choose_pre.shape[0])]
        no_choose_nevigate_sample.set_index(pd.Index(index_range), inplace=True)
        df_no_choose_pre.set_index(pd.Index(index_range), inplace=True)

        df_no_choose_save = pd.concat([no_choose_nevigate_sample, df_no_choose_pre], axis=1)
        # df_no_choose_save.reset_index(drop=True, inplace=True)
        # ascending：升序
        df_no_choose_save.sort_values(by="pre", inplace=True, ascending=True)
        df_no_choose_save.reset_index(drop=True, inplace=True)
        df_no_choose_save.to_feather('./data/result_pre_nevigate/'+str(model_loop)+"_"+str(train_group)+"_pre.feather")

        # torch.save(net, './data/result_model/'+str(model_loop)+"_"+str(train_group)+"net_slf.pth")
save_roc_auc_score_df = pd.DataFrame(save_roc_auc_score,columns=[["model_loop","train_group","epoch","auc"]])
save_roc_auc_score_df.to_csv("./data/result_pre_test/acc.csv",index=None)

save_recall_df = pd.DataFrame(save_recall,columns=["model_loop","train_group","recall"])
save_recall_df.to_csv(f"./data/result_pre_recall/recall.csv")