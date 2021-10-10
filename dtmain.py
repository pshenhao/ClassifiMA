# 参考1：https://blog.csdn.net/qq_27697703/article/details/81358245
# 参考2：https://zhuanlan.zhihu.com/p/56242525
# 参考3：https://www.it610.com/article/1291565762963644416.htm

############  导包  ############
import operator
import pickle
from math import log
import numpy as np
import xlrd

###########  参数  #############
folder_path = r"D:\MyDownload\Jupyter\MLearningl1\dt-ClassifiMarAnimal"
train_file_name = 'data_train.xlsx'
test_file_name = 'data_test.xlsx'
tree_save_name = 'myTree.pkl'

########### 变量  #############
train_file_path = folder_path + "//" + train_file_name
test_file_path = folder_path + "//" + test_file_name
tree_save_path = folder_path + "//" + tree_save_name

######### 读取数据 ###########
def readXlsx(file_path):
    read_book = xlrd.open_workbook(file_path)
    sheet = read_book.sheet_by_index(0)
    n_rows = sheet.nrows  # 行
    n_cols = sheet.ncols  # 列
    data_arr = np.ones([n_rows - 1, n_cols])  # 第一行为属性名字，非数字

    for i in range(n_rows - 1):
        for j in range(n_cols):
            data_arr[i, j] = sheet.cell_value(i + 1, j)

    return data_arr


########  获取信息  ##########
def getArrData(data_arr):
    nrows, ncols = data_arr.shape
    name_vec = np.zeros((nrows, 1))
    label_vec = np.zeros((nrows, 1))
    data_array = np.zeros((nrows, ncols - 1))  # 为了计算简单，保存了最后一列数据

    for i in range(nrows):
        name_vec[i, 0] = data_arr[i, 0]
        label_vec[i, 0] = data_arr[i, -1]
        # 第一行为名字，最后一行为标签
        for j in range(1, ncols):
            data_array[i, j - 1] = data_arr[i, j]

    return name_vec, data_array, label_vec


######## 计算熵  #############
def getEntroy(data_arr):
    rows, cols = data_arr.shape
    label_cnt = {}  # 记录标签出现次数

    for i in range(rows):
        if data_arr[i, -1] not in label_cnt:
            label_cnt[data_arr[i, -1]] = 0  # 不存在，则为0
        label_cnt[data_arr[i, -1]] += 1  # 已存在，则记录出现次数

    shannonEnt = 0.0  # 香农熵
    for key in label_cnt:
        key_p = float(label_cnt[key]) / rows  # 标签出现概率
        shannonEnt -= (key_p * log(key_p, 2))  # 计算该标签香农熵，并进行累减最终得到香农熵

    return shannonEnt


######## 划分数据集  ##########
## 该函数根据给定的特征值提取数据
def getSplitData(data_mat, feature_inx, feature_val):
    rows, cols = data_mat.shape
    data_mat_split = np.zeros((1, cols))
    data_mat_flag = 0

    for i in range(rows):
        if data_mat[i, feature_inx] == feature_val:  # 如果是我们要分割出来的
            data_mat_arr = data_mat[i, :]  # 提取该行
            data_mat_arr = data_mat_arr.reshape(1, cols)

            if data_mat_flag == 0:
                data_mat_flag = data_mat_flag + 1
                data_mat_split = data_mat_arr  # 如果还未添加数据，则直接赋值
            else:
                data_mat_split = np.vstack((data_mat_split, data_mat_arr))  # 否则在矩阵后面追加数据

    data_mat_split = np.delete(data_mat_split, feature_inx, axis=1)  # 去除改列的特征
    return data_mat_split  # 返回分割出来的数据矩阵


########  选择划分特征  ##########
## 选择熵增益最大的特征进行划分数据集
def chooseFeature(data_mat):
    nrows, ncols = data_mat.shape
    feature_num = ncols - 1
    shann_ent_old = getEntroy(data_mat)  # 计算原来的香农熵
    feature_best_add = 0.0  # 最大熵增益
    feature_best_inx = -1  # 最好的特征的索引

    for i in range(feature_num):  # 遍历所有的特征
        # 第i个特征的所有的特征的值
        feature_value_list = [feature_vect[i] for feature_vect in data_mat]
        feature_value_set = set(feature_value_list)  # 转set去重
        shann_ent_feati = 0.0  # 第i个特征划分后的香农熵

        for val in feature_value_set:
            data_split = getSplitData(data_mat, i, val)  # 根据此特征的值划分数据
            # 计算该特征划分后的得到的香农熵
            val_p = float(len(data_split) / float(len(data_mat)))
            shann_ent_feati = shann_ent_feati + val_p * getEntroy(data_split)
        shann_ent_add = shann_ent_old - shann_ent_feati  # 熵增益

        if (shann_ent_add > feature_best_add):
            feature_best_add = shann_ent_add
            feature_best_inx = i

    return feature_best_inx


######### 多数表决法 #########
def majorityCnt(class_list):
    class_cont = {}  # 记录标签出现次数

    for c in class_list:
        if c not in class_cont.keys():
            class_cont[c] = 1
        else:
            class_cont[c] = class_cont + 1

    # 排序，降序排列
    sorted_class_cont = sorted(class_cont.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_cont[0][0]


######### 构造决策树 #########
## 由于数据需要被多次划分，并且划分之后的数据被传递到下一节点，
## 因此需要递归构造决策树
def creatTree(data_mat, labels):
    class_list = [examp[-1] for examp in data_mat]

    # 如果所有的类标签一样，说明分类完全，返回该类标签
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    # 如果所有特征已经被使用完，应返回，若分类不完全，则使用多数表决法决定叶子节点的分类
    if len(data_mat[0]) == 1:
        return majorityCnt(class_list)

    # 若上面两个返回条件均未达到，则继续递归
    best_feature_inx = chooseFeature(data_mat)  # 选择的特征索引
    best_feature_label = labels[best_feature_inx]  # 选择的特征名字

    decision_tree = {best_feature_label: {}}  # 此特征作为父节点
    del labels[best_feature_inx]  # 删除该特征

    ## 选择好特征之后，接下来进行分类操作
    feature_val_list = [examp[best_feature_inx] for examp in data_mat]  # 提取
    feature_val_set = set(feature_val_list)  # 去重
    for val in feature_val_set:
        split_label = labels[:]
        data_split = getSplitData(data_mat, best_feature_inx, val)  # 获取分割后的数据
        decision_tree[best_feature_label][val] = creatTree(data_split, split_label)  # 对后续的标签进行递归

    return decision_tree


##########  决策树分类  ########
def dtClassfiy(dt_tree, feature_labels, test_vect):
    f_node, = dt_tree.keys()
    t_node = dt_tree[f_node]
    feature_index = feature_labels.index(f_node)
    for key in t_node.keys():
        if test_vect[feature_index] == key:
            if type(t_node[key]).__name__ == 'dict':
                class_label = dtClassfiy(t_node[key], feature_labels, test_vect)
            else:
                class_label = t_node[key]
    return class_label


######### 保存决策树  #########
def saveTree(input_tree, file_path):
    pkl_file = open(file_path, 'wb')
    pickle.dump(input_tree, pkl_file)
    pkl_file.close()


######## 导入决策树 ##########
def loadTree(file_path):
    pkl_file = open(file_path, 'rb')
    pkl_tree = pickle.load(pkl_file)
    pkl_file.close()
    return pkl_tree


######### main #############
# 读取
train_arr = readXlsx(train_file_path)
test_arr = readXlsx(test_file_path)

# 转int
train_arr = train_arr.astype(int)
test_arr = test_arr.astype(int)

# 处理
train_name, train_mat, train_label = getArrData(train_arr)
test_name, test_mat, test_label = getArrData(test_arr)

label_name = ['no_surfacing', 'flippers']

# 构造
myTree = creatTree(train_mat, label_name)
# 保存
# saveTree(myTree, tree_save_path)
# 导入
# mytree_load = loadTree(tree_save_path)
# 测试
feature_labels = ['no_surfacing', 'flippers']
print(dtClassfiy(myTree, feature_labels, [1, 0]))





data_split0 = getSplitData(train_mat, 0, 0)
data_split1 = getSplitData(train_mat, 0, 1)
print(test_mat)
print(myTree)
