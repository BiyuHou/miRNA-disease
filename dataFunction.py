import pandas as pd
import os
import numpy as np
from tqdm import tqdm

### 这个类用来生成正负样本、划分数据集、树的特征向量
### 输入为关联矩阵的文件路径
class data_operate(object):
	def __init__(self, dir_path):
		super(data_operate, self).__init__()
		self.dir_path = dir_path

		if self.dir_path.endswith(".feather"):
			print(f"你正在读取{self.dir_path}文件")
			self.df = pd.read_feather(self.dir_path)
		if self.dir_path.endswith(".csv"):
			print(f"你正在读取{self.dir_path}文件")
			self.df = pd.read_csv(self.dir_path,index_col = 0,header = 0)

	def mask_function(self):
		'''
			将数据中1通过掩膜转化成0，用于后续计算召回率
			将转换的数值的横纵坐标（疾病和miRNA）进行输出
			num_mask为需要进行转换的个数
			将经过处理的关联矩阵和被转换成无关联的疾病和miRNA名字保存成表
		'''
		df = self.df
		rows_values_1, cols_values_1 = (df == 1).values.nonzero()
		rows_values_0, cols_values_0 = (df == 0).values.nonzero()
		connect_group = [(df.index[row], df.columns[col]) for row, col in zip(rows_values_1, cols_values_1)]
		num_connect = len(connect_group)
		print(f"存在{num_connect}个正样本")
		row_sum = df.sum(axis=1)	# 行，疾病名字
		col_sum = df.sum(axis=0)	# 列，RNA名字

		# 剔除部分存在较少关联的疾病和RNA
		can_be_choose_row = row_sum[row_sum>10].index.values
		can_be_choose_col = col_sum[col_sum>10].index.values
		can_be_choose_connect = []
		for tuple_connect in connect_group:
			if tuple_connect[0] in can_be_choose_row and \
					tuple_connect[1] in can_be_choose_col:
				can_be_choose_connect.append([tuple_connect[0],tuple_connect[1]])


		mask_df = df.copy()
		choose_row = []
		choose_col = []
		can_choose_num = len(can_be_choose_connect)
		print(f"可计算召回率的数据：{can_choose_num}")

		if can_choose_num < int(num_connect * 0.1): 	# 如果可计算召回率的值小于正样本的10%,则直接采用总数的10%
			num_mask = can_choose_num
		else:
			num_mask = int(num_connect * 0.1)					# 选取可用于计算召回率的数据的10%计算


		# 随机生成避免集中
		sort_choose = np.random.choice(len(can_be_choose_connect),len(can_be_choose_connect),replace=False)
		for i in sort_choose:
			if len(choose_row) < num_mask:
				name_disease_row= can_be_choose_connect[i][0]
				name_miRNA_col = can_be_choose_connect[i][1]
				if choose_row.count(name_disease_row) < 5 and choose_col.count(name_miRNA_col) < 5:
					choose_row.append(name_disease_row)
					choose_col.append(name_miRNA_col)
					mask_df.loc[name_disease_row,name_miRNA_col] = 0
			# 当达到预设的召回率样本数量，直接退出
			else:
				break
		print(f"选取了{len(choose_row)}个样本计算召回率")
		mask_df.to_csv("./data/mask_matrix_data.csv")
		recall_sample_dataFrame = {'disease': choose_row, 'miRNA': choose_col}
		recall_sample_dataFrame = pd.DataFrame(recall_sample_dataFrame)
		recall_sample_dataFrame.to_csv("./data/recall_sample.csv",index=None)


		mask_rows_values_1, mask_cols_values_1 = (mask_df == 1).values.nonzero()
		mask_connect_group = [[mask_df.index[row], mask_df.columns[col]] for row, col in zip(mask_rows_values_1, mask_cols_values_1)]
		mask_connect_group_DataFrame = pd.DataFrame(mask_connect_group)
		mask_connect_group_DataFrame.columns = ["disease","miRNA"]
		mask_connect_group_DataFrame.to_csv("./data/mask_positivate.csv",index=None)

		mask_rows_values_0, mask_cols_values_0 = (mask_df == 0).values.nonzero()
		mask_unconnect_group = [[mask_df.index[row], mask_df.columns[col]] for row, col in zip(mask_rows_values_0, mask_cols_values_0)]
		mask_unconnect_group_DataFrame = pd.DataFrame(mask_unconnect_group)
		mask_unconnect_group_DataFrame.columns = ["disease","miRNA"]
		mask_unconnect_group_DataFrame.to_csv("./data/mask_nevigate.csv",index=None)

class create_tree_data(object):
	"""docstring for create_tree_data"""
	def __init__(self, dir_path):
		super(create_tree_data, self).__init__()
		self.df = pd.read_csv(dir_path,index_col=0)
		self.miRNA_list = self.df.columns.values.tolist()
		self.disease_list = self.df.index.values.tolist()
		self.searched_node = []
		self.saveResult = []
		self.father_node = None

	def search_next_node(self,next_node,num):
		miRNA_list = self.miRNA_list
		disease_list = self.disease_list
		df = self.df
		father_node = self.father_node
		saveResult = self.saveResult
		layer_node = []
		searched_node = self.searched_node
		num += 1
		for node_name in next_node:
			if node_name in miRNA_list and node_name not in searched_node:
				searched_node.append(node_name)
				new_next_node = df[df[node_name] == 1].index.tolist()
				new_next_node = [s for s in new_next_node if s not in searched_node]
				layer_node = layer_node + ["split"] + new_next_node
			elif node_name in disease_list and node_name not in searched_node:
				searched_node.append(node_name)
				new_next_node = df.columns[df.loc[node_name] == 1].tolist()
				new_next_node = [s for s in new_next_node if s not in searched_node]
				layer_node = layer_node + ["split"] + new_next_node
		layer_node_set = set(layer_node)
		layer_node = list(layer_node_set)
		saveResult.append(layer_node)

		if len(layer_node)>0:
			self.search_next_node(layer_node,num,)
		else:
			dataFrame = pd.DataFrame(saveResult)
			if father_node in miRNA_list:
				dataFrame.to_csv(os.path.join(r"./data/miRNA_tree", father_node+".csv"),index=None,header=None)
			elif father_node in disease_list:
				dataFrame.to_csv(os.path.join(r"./data/disease_tree", father_node+".csv"),index=None,header=None)

	def create_miRNA_tree(self):
		df = self.df
		miRNA_list = self.miRNA_list
		for father_node in tqdm(miRNA_list):
			self.father_node = father_node
			num = 0
			searched_node = []  # 存放搜索过的结点
			searched_node.append(father_node)
			saveResult = []
			self.searched_node = searched_node
			self.saveResult = saveResult
			next_node = df[df[father_node] == 1].index.tolist()
			saveResult.append([father_node])
			saveResult.append(next_node)
			self.search_next_node(next_node, num)

	def create_disease_tree(self):
		df = self.df
		disease_list = self.disease_list
		for father_node in tqdm(disease_list):
			self.father_node = father_node
			num = 0
			searched_node = []  # 存放搜索过的结点
			searched_node.append(father_node)
			saveResult = []

			self.searched_node = searched_node
			self.saveResult = saveResult

			next_node = df.columns[df.loc[father_node] == 1].tolist()
			saveResult.append([father_node])
			saveResult.append(next_node)
			self.search_next_node(next_node, num)

	## 下面的这个方法要实现，将是数据转换成向量表达
	def create_feature_vector(self):
		dir_miRNA_path = "./data/miRNA_tree"
		dir_disease_path = "./data/disease_tree"
		disease_path_list = os.listdir(dir_disease_path)
		miRNA_path_list = os.listdir(dir_miRNA_path)

		disease_name = [d[:-4] for d in disease_path_list]
		miRNA_name = [m[:-4] for m in miRNA_path_list]
		feature_vector_sort = disease_name + miRNA_name	# 特征包含所有疾病和miRNA
		np.random.seed(101)
		np.random.shuffle(feature_vector_sort)	# 将特征随机打乱
		feature_vector_dict = []
		for sort in range(len(feature_vector_sort)):
			feature_vector_dict.append([feature_vector_sort[sort], sort])
		feature_vector_dict = dict(feature_vector_dict)


		length_feature = len(feature_vector_sort)
		save_feature_vector = []
		for father_node in tqdm(feature_vector_sort):
			init_feature = np.zeros(length_feature,dtype=int).tolist()
			df_tree = None
			if father_node in disease_name:
				df_tree = pd.read_csv(os.path.join(r".\data\disease_tree", father_node+".csv"))
			elif father_node in miRNA_name:
				df_tree = pd.read_csv(os.path.join(r".\data\miRNA_tree", father_node+".csv"))
			row = df_tree.shape[0]
			for r in range(row):
				node = df_tree.iloc[r, :].values.tolist()
				node = [i for i in node if i == i and i != "split"]
				try:
					for sub_node in node:
						ord_num = feature_vector_dict[sub_node]
						init_feature[ord_num] = r
				except:
					print("这个结点文件不存在",sub_node)
			save_feature_vector.append(init_feature)

		save_feature_vector_df = pd.DataFrame(save_feature_vector,columns = feature_vector_sort,index = feature_vector_sort)
		save_feature_vector_df.to_csv("./data/feature_vector.csv")





data_operate = data_operate('./data/matrix_data.csv')
data_operate.mask_function()
create_tree_data = create_tree_data('./data/mask_matrix_data.csv')
# create_tree_data.create_disease_tree()
# create_tree_data.create_miRNA_tree()
# create_tree_data.create_feature_vector()