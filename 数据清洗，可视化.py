#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）


# 获取酒庄

fdata = pd.concat([
    pd.read_csv("factories-东北(23).csv"),
    pd.read_csv("factories-甘肃(8).csv"),
    pd.read_csv("factories-河北(42).csv"),
    pd.read_csv("factories-内蒙古(3).csv"),
    pd.read_csv("factories-宁夏(37).csv"),
    pd.read_csv("factories-山东(55).csv"),
    pd.read_csv("factories-新疆(29).csv"),
    pd.read_csv("factories-河南(2).csv"),
    pd.read_csv("factories-云南(3).csv")
])
fdata.reset_index(drop=True, inplace=True)
fdata.head()

# 获取红酒数据

wdata = pd.concat([
    pd.read_csv("wines-云南(3).csv"),
    pd.read_csv("wines-东北(23).csv"),
    pd.read_csv("wines-甘肃(8).csv"),
    pd.read_csv("wines-河北(42).csv"),
    pd.read_csv("wines-河南(2).csv"),
    pd.read_csv("wines-内蒙古(3).csv"),
    pd.read_csv("wines-宁夏(37).csv"),
    pd.read_csv("wines-山东(55).csv"),
    pd.read_csv("wines-新疆(29).csv")
])
wdata.reset_index(drop=True, inplace=True)
wdata.head()


# ## 获取葡萄种类


gdata = pd.read_csv("grapes.csv")
gdata.head()


# ## 获取各红酒的酿酒葡萄的频率



table = fdata.merge(wdata, on="酒庄")



gcount = {}
for i in table["酿酒葡萄"]:
    for grape in eval(i):
        if grape in gdata.values:
            if grape in gcount:
                gcount[grape] += 1
            else:
                gcount[grape] = 1
gcount = dict(sorted(gcount.items(), key=lambda x : x[1])[::-1])
gcount


# ## 各产庄红酒酿酒葡萄统计



plt.figure(figsize=(10, 10))
plt.bar(range(len(gcount)), list(gcount.values()), width=0.5)
plt.xticks(range(len(gcount)), list(gcount.keys()), rotation=80)
for x, y in zip(range(len(gcount)), gcount.values()):
    plt.text(x, y, y, ha="center")
plt.title("各产庄红酒的酿酒葡萄条形统计", fontsize=15)
plt.ylabel("用该红酒酿酒的数量")
plt.xlabel("葡萄种类")
plt.show()


gcount_g10 = {}
s = sum(gcount.values())
for k, v in gcount.items():
    if v <= 10:
        gcount_g10["其它"] = gcount_g10.get("其它", 0) + v / s
    else:
        gcount_g10[k] = v / s

plt.figure(figsize=(10,10))
plt.pie(list(gcount_g10.values()), labels=list(gcount_g10.keys()), autopct='%1.2f%%')
plt.title("各产庄红酒的酿酒饼形统计图", fontsize=15)
plt.savefig('各产庄红酒的酿酒饼形统计图.png')
plt.show()



# ## 分析红酒数据

# ### 红酒风味



wdata[wdata["风味特征"].isnull()]



def fruity_flavor(s: str)->bool:
    keyword = ("柠檬", "葡萄柚", "黑莓", "覆盆子", "山莓", "悬钩子", "蓝莓", "桑葚", "黑醋栗", "草莓", "樱桃", "车厘子", "杏子", "桃子", "苹果", "梨", "菠萝", "凤梨", "香蕉", "芒果", "甜瓜", "草莓酱", "葡萄干", "李子干", "西梅干", "无花果", "椰子", "水果")
    return any(i in str(s) for i in keyword)


fruity_wine = wdata[list(map(fruity_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
fruity_wine["酒款类型"].value_counts()


# In[11]:


def vegetative_flavor(s: str)->bool:
    keyword = ("葡萄梗", "割青草味", "甜椒", "大黄", "灌木丛", "蕨类植物", "桉树", "尤加利树", "薄荷", "嫩菜豆", "芦笋", "绿橄榄", "黑橄榄", "洋蓟", "球蓟", "麦秆", "稻草", "烟草", "茶叶")
    return any(i in str(s) for i in keyword)


vegetative_wine = wdata[list(map(vegetative_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
vegetative_wine["酒款类型"].value_counts()



def spicy_flavor(s: str)->bool:
    keyword = ("肉桂", "丁香", "黑胡椒", "大茴香", "肉豆蔻", "欧亚甘草")
    return any(i in str(s) for i in keyword)


spicy_wine = wdata[list(map(spicy_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
spicy_wine["酒款类型"].value_counts()




def nutty_flavor(s: str)->bool:
    keyword = ("榛子", "杏仁", "核桃", "腰果")
    return any(i in str(s) for i in keyword)


nutty_wine = wdata[list(map(nutty_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
nutty_wine["酒款类型"].value_counts()





def caramel_flavor(s: str)->bool:
    keyword = ("蜂蜜", "丁二酮", "巧克力", "浆果", "黄油", "奶油硬糖", "酱油")
    return any(i in str(s) for i in keyword)


caramel_wine = wdata[list(map(caramel_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
caramel_wine["酒款类型"].value_counts()




def woody_flavor(s: str)->bool:
    keyword = ("烟熏味", "烤焦的面包片", "可可", "咖啡", "药用味", "酚醛树脂", "烟肉", "烟熏肉", "橡木", "香草", "香子兰", "雪松")
    return any(i in str(s) for i in keyword)


woody_wine = wdata[list(map(woody_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
woody_wine["酒款类型"].value_counts()




def earthy_flavor(s: str)->bool:
    keyword = ("霉", "松露", "蘑菇", "腐殖土", "尘土", "灰尘")
    return any(i in str(s) for i in keyword)

earthy_wine = wdata[list(map(earthy_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
earthy_wine["酒款类型"].value_counts()

def pungent_flavor(s: str)->bool:
    keyword = ("薄荷醇", "乙醇", "杂醇油")
    return any(i in str(s) for i in keyword)

pungent_wine = wdata[list(map(pungent_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
pungent_wine["酒款类型"].value_counts()




def oxidized_flavor(s: str)->bool:
    keyword = ("雪莉", "雪利", "乙醛", "醋醛")
    return any(i in str(s) for i in keyword)


oxidized_wine = wdata[list(map(oxidized_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
oxidized_wine["酒款类型"].value_counts()


def chemical_flavor(s: str)->bool:
    keyword = ("杂醇油", "山梨酸酯", "肥皂", "乙酸", "醋酸", "乙醇", "乙酸乙酯", "潮湿纸板", "过滤垫", "湿羊毛", "湿狗", "二氧化硫", "燃烧火柴", "烹甘蓝", "臭鼬", "大蒜", "硫醇", "天然气", "硫化氢", "橡胶", "柴油", "煤油", "塑胶", "沥青", "柏油")
    return any(i in str(s) for i in keyword)


chemical_wine = wdata[list(map(chemical_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
chemical_wine["酒款类型"].value_counts()



def microb_flavor(s: str)->bool:
    keyword = ("老鼠", "马", "优酪乳", "乳酸", "酸乳酪", "丁酸", "酸泡菜", "发酵渣", "发面酵母", "酒花")
    return any(i in str(s) for i in keyword)


microb_wine = wdata[list(map(microb_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
microb_wine["酒款类型"].value_counts()


def floral_flavor(s: str)->bool:
    keyword = ("紫罗兰", "玫瑰花", "香橙花", "蔷薇", "天兰葵", "金合欢", "薰衣草", "茉莉", "金银花")
    return any(i in str(s) for i in keyword)


floral_wine = wdata[list(map(floral_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
floral_wine["酒款类型"].value_counts()



def other_flavor(s: str)->bool:
    keyword = ("鱼腥", "野味", "麝香", "皮革", "火石", "琥珀")
    return any(i in str(s) for i in keyword)


other_wine = wdata[list(map(other_flavor, wdata["风味特征"]))][["葡萄酒名称", "酒款类型", "风味特征"]]
other_wine["酒款类型"].value_counts()

# 葡萄酒酒款类型统计



wdata["酒款类型"].value_counts()


# 酒庄分析
#葡萄采摘方式


fdata["葡萄采摘方式"].value_counts()


# 酒庄条件

enviroment = fdata[fdata["土壤类型"] != 'None']

grape_tree = fdata[fdata["平均葡萄树树龄"] != "None"]

