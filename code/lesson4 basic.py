##### Numpy #####

import numpy as np

array = [1,2,3,4,5,6,7,8,9,10]
array + 1

array = np.array([1,2,3,4,5,6,7,8,9,10])
print(type(array))

array2 = array + 1
array2

array2 + array
array2 * array

array.shape # 展现数据维度，一个逗号“,”表示一个维度，数字表示元素数目

sample_list = [1,2,3,4,5,6,7,8,9,10]
sample_list.shape # python的list没有shape属性，所以会报错，导入numpy后，list就变成了numpy的array
sample_list = np.array([1,2,3,4,5,6,7,8,9,10])


np.array([[1,2,3],[4,5,6]]) # 二维数组，注意双重中括号

sample_list = [1,2,3,4,5]
sample_array = np.array(sample_list)
sample_array.shape


sample_list = [1,2,3,4,5,'6']
sample_array = np.array(sample_list)
print(sample_array)

sample_list = [1,2,3,4,5,6.0]
sample_array = np.array(sample_list)
print(sample_array) # ndarray中的元素必须是同一类型，否则会自动转换为同一类型，int转换为float，float转换为string

type(sample_array) # 查看数据格式
sample_array.dtype # 查看数据类型
sample_array.size # 查看数据元素个数
sample_array.ndim # 查看数据维度

sample_array[1:3] # 切片，注意是左闭右开区间,从第二个元素开始到第三个元素
sample_array[-2:] # 从倒数第二个元素开始到最后

sample_array = np.array([[1,2,3],[4,5,6],[7,8,9]]) # 二维数组
sample_array[1,1] = 10 # 修改第二行第二列的元素

sample_array[1] # 取第二行
sample_array[:,1] # 取第二列，冒号表示所有行或者所有列，逗号前面表示行，逗号后面表示列

sample_array = np.arange(0,100,10) # 生成0到100的等差数列，步长为10
mask = np.array([1,0,1,0,1,0,1,0,1,0],dtype=bool) # 生成一个布尔型的掩码，1表示True，0表示False
sample_array[mask] # 通过掩码取出数组中的元素，掩码中为True的元素会被取出，为False的元素会被忽略

random_array = np.random.rand(10) # 生成10个随机数,均匀分布,[0,1)

mask = random_array > 0.5 # 生成一个掩码，大于0.5的为True，小于0.5的为False
mask
random_array[mask]

sample_array = np.array([10,20,30,40,50]) # 生成一个数组
np.where(sample_array > 30) # 返回大于30的元素的索引，索引是以0开头
sample_array[np.where(sample_array > 30)] # 返回大于30的元素

y = np.array([1,1,1,4])
x = np.array([1,1,1,2])
x == y

np.logical_and(x,y) # 逻辑与，两个都为True才为True
np.logical_or(x,y) # 逻辑或，两个都为False才为False

sample_array = np.array([1,2,3,4,5],dtype=np.float32)
sample_array.dtype

sample_array = np.array(['1','10','3.5','str'],dtype=np.object)
sample_array = np.array([1,2,3,4,5]) 
sample_array2 = np.asarray(sample_array,dtype=np.float32) # 将sample_array转换为float32类型

sample_array = np.array([[1,2,3],[4,10,6],[7,8,9]])
sample_array2 = sample_array
sample_array2[1,1] = 100

sample_array2 = sample_array.copy() # 深拷贝，sample_array2和sample_array是两个独立的数组
sample_array2[1,1] = 10000

sample_array = np.array([[1,2,3],[4,5,6]])
np.sum(sample_array) # 求和

np.sum(sample_array,axis=0) # 按列求和
np.sum(sample_array,axis=1) # 按行求和

sample_array.prod() # 求积
sample_array.prod(axis = 0)  # 按列求积
sample_array.prod(axis = 1)  # 按行求积
sample_array.min() # 求元素中的最小值

sample_array.min(axis = 0) # 求每一列的最小值
sample_array.min(axis = 1) # 求每一行的最小值

sample_array.mean() # 求均值
sample_array.mean(axis = 0) # 求每一列的均值
sample_array.mean(axis = 1) # 求每一行的均值

sample_array.std() # 求标准差
sample_array.std(axis = 0) # 求每一列的标准差
sample_array.std(axis = 1) # 求每一行的标准差

sample_array.var() # 求方差
sample_array.var(axis = 0) # 求每一列的方差
sample_array.var(axis = 1) # 求每一行的方差

sample_array.clip(2,4) # 将数组中的元素限制在2到4之间

sample_array = np.array([1.2,3.56,6.41])
sample_array.round() # 四舍五入
sample_array.round(decimals=1) # 小数点后保留一位

sample_array.argmin() # 求最小值的索引
sample_array.argmin(axis = 0) # 求每一列最小值的索引
sample_array.argmin(axis = 1) # 求每一行最小值的索引

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
np.multiply(x,y) # 对应元素相乘
np.dot(x,y) # 矩阵乘法

x.shape = 5,1 # 将x转换为5行1列的矩阵
np.dot(x,y) # 矩阵乘法

y.shape = 1,5 # 将y转换为1行5列的矩阵
np.dot(x,y) 
np.dot(y,x) 





##### Matplotlib #####
import matplotlib.pyplot as plt # 导入matplotlib.pyplot模块
# %matplotlib inline # 在jupyter notebook中显示图像，魔法函数，不是python语法，只在jupyter notebook中使用

plt.plot([1,2,3,4,5],[1,4,9,16,25]) # 画图
plt.xlabel('xlabel',fontsize = 16) # x轴标签设置为xlabel，字体大小为16
plt.ylabel('ylabel',fontsize = 16) 
plt.show()

plt.plot([1,2,3,4,5],[1,4,9,16,25],'-.') # 画图，线条为虚点线
plt.xlabel('xlabel',fontsize = 16) # x轴标签设置为xlabel，字体大小为16
plt.ylabel('ylabel',fontsize = 16)
plt.show()

plt.plot([1,2,3,4,5],[1,4,9,16,25],'-.',color = 'r') # 画图，线条为红色虚点线
plt.show()
sample_array = np.arange(0,10,0.5) # 生成0到10，步长为0.5的数组
plt.plot(sample_array,sample_array,'r--') # 画图，线条为红色虚线
plt.show()
plt.plot(sample_array,sample_array**2,'bs') # 画图，线条为蓝色方块
plt.show()
plt.plot(sample_array,sample_array**3,'go') # 画图，线条为绿色圆点
plt.show()

x = np.linspace(-10,10) # 生成-10到10的等差数列
y = np.sin(x) # 计算x的正弦值
plt.plot(x,y,linewidth = 3.0)
plt.show()
plt.plot(x,y,color='b',linestyle=':',marker='o',markerfacecolor='r',markersize=10) # 画图，线条为蓝色，线型为虚线，点为红色圆点，点的大小为10
plt.show()

line = plt.plot(x,y)
plt.plot(line,color='r',linewidth=3.0,alpha=0.5) # 画图，线条为红色，线宽为3.0，透明度为0.5
plt.show()

plt.subplot(221) # 2行2列的第一个子图
plt.show()
plt.subplot(212) # 2行1列的第二个子图
plt.show()

plt.subplot(121) # 1行2列的第一个子图
plt.plot(x,y,color = 'r')
plt.show()
plt.subplot(122) # 1行2列的第二个子图
plt.plot(x,y,color = 'b')
plt.show()

plt.subplot(321) # 3行2列的第一个子图
plt.plot(x,y,color = 'r')
plt.show()
plt.subplot(324) # 3行2列的第四个子图
plt.plot(x,y,color = 'b')
plt.show()


plt.plot(x,y,color = 'b',linestyle=':',marker='o',markerfacecolor='r',markersize=10) # 画图，线条为蓝色，线型为虚线，点为红色圆点，点的大小为10
plt.xlabel(' x:---' )
plt.ylabel(' y:---' )
plt.title('sample plot:---')
plt.text(0,0,'(0,0)') # 在坐标(0,0)处添加文本
plt.grid(True) # 添加网格
plt.annotate('sample annotation',xy=(-5,0),xytext=(-2,0.3),arrowprops=dict(facecolor='red',shrink=0.05,headwidth=20,headlength=20))
# xy为箭头尖端的坐标，xytext为注释文本的坐标，arrowprops为箭头的属性
plt.show()

x = range(10)
y = range(10)
fig = plt.gca() # 获取当前的坐标轴
plt.plot(x,y)
fig.axes.get_xaxis().set_visible(False) # 隐藏x轴
fig.axes.get_yaxis().set_visible(False) # 隐藏y轴
plt.show()

import math
x = np.random.normal(loc = 0.0,scale = 1.0,size = 300) 
width = 0.5
bins = np.arange(math.floor(x.min())-width,math.ceil(x.max())+width,width) # 生成等差数列
ax = plt.subplot(111) # 1行1列的第一个子图

ax.spines['top'].set_visible(False) # 隐藏上边框
ax.spines['right'].set_visible(False) # 隐藏右边框
plt.tick_params(bottom='off',top='off',right='off',left='off') # 隐藏刻度
plt.grid() # 添加网格
plt.hist(x,alpha=0.5,bins=bins) # 画直方图，透明度为0.5，bins为等差数列
plt.show()

x = range(10)
y = range(10)
labels = ['sample plot' for i in range(10)]
fig,ax = plt.subplots()
plt.plot(x,y)
plt.title('sample plot')
ax.set_xticklabels(labels,rotation=45,horizontalalignment='right') # 设置x轴刻度标签，旋转45度，水平对齐方式为右对齐
plt.show()

x = np.arange(10)

for i in range(1,4):
    plt.plot(x,i*x**2,label = ' Grooup %d' %i)  # for循环画图，label为图例
plt.legend(loc='best') # 显示图例，loc为图例的位置
plt.show()

print(help(plt.legend)) # 查看legend的帮助文档

fig = plt.figure()
ax = plt.subplot(111)
x = np.arange(10)
for i in range(1,4):
    plt.plot(x,i*x**2,label = ' Grooup %d' %i)  # for循环画图，label为图例
ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=3)
plt.show()

plt.style.available()
help(plt.style.available)

x = np.linspace(-10,10) # 生成等差数列，从-10到10，默认个数为50
y = np.sin(x)
plt.plot(x,y)
plt.show()

plt.style.use('dark_background') # 设置背景为黑色
plt.plot(x,y)
plt.show()

plt.style.use('bmh') # 设置背景为灰色
plt.plot(x,y)
plt.show()

plt.style.use('ggplot') # 设置背景为灰色
plt.plot(x,y)
plt.show()

np.random.seed(0)
x = np.arange(5) # 生成等差数列，从0到4，默认个数为5
y = np.random.randint(-5,5,5) # 生成随机整数，从-5到5，个数为5
fig,axes = plt.subplots(ncols = 2)
v_bars = axes[0].bar(x,y,color = 'red') # 画垂直柱状图
h_bars = axes[1].barh(x,y,color = 'red') # 画水平柱状图
axes[0].axhline(0,color='grey',linewidth=2) # 画水平线，颜色为灰色，线宽为2
axes[1].axvline(0,color='grey',linewidth=2) # 画垂直线，颜色为灰色，线宽为2
plt.show()

mean_values = [1,2,3]
variance = [0.2,0.4,0.5]
bar_label = ['bar1','bar2','bar3']
x_pos = list(range(len(bar_label))) 
plt.bar(x_pos,mean_values,yerr=variance,alpha=0.3) # 画柱状图，yerr为误差线，alpha为透明度
max_y = max(zip(mean_values,variance)) # 获取最大值，zip函数将两个列表合并为元组
plt.ylim([0,(max_y[0]+max_y[1])*1.2]) 
plt.ylabel('variable y')
plt.xticks(x_pos,bar_label) # 设置x轴刻度标签
plt.show()


data = range(200,225,5)
bar_labels = ['a','b','c','d','e']
fig = plt.figure(figsize=(10,8)) # 设置画布大小
y_pos = np.arange(len(data)) # 生成等差数列，从0到4，默认个数为5
plt.yticks(y_pos,bar_labels,fontsize=16) # 设置y轴刻度标签，字体大小为16
bars = plt.barh(y_pos,data,alpha=0.5,color='green') # 画水平柱状图，透明度为0.5，颜色为绿色
plt.vlines(min(data),-1,len(data)+0.5,linestyles='dashed') # 画垂直线，颜色为灰色，线宽为2
for b,d in zip(bars,data):
    plt.text(b.get_width()+b.get_width()*0.05,b.get_y()+b.get_height()/2,'{0:.2%}'.format(d/min(data)))
plt.show()


pattern = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
mean_value = range(1,len(pattern)+1)
x_pos = list(range(len(mean_value)))
bars = plt.bar(x_pos,mean_value,color='white')
for bar,pattern in zip(bars,pattern):
    bar.set_hatch(pattern) # 设置柱状图的填充样式
plt.show()



data = range(200,225,5) # 生成等差数列，从200到225，步长为5
bar_labels = ['a','b','c','d','e']
fig = plt.figure(figsize=(10,8)) # 设置画布大小
y_pos = np.arange(len(data)) # 生成等差数列，从0到4，默认个数为5
plt.yticks(y_pos,bar_labels,fontsize=16) # 设置y轴刻度标签，字体大小为16
bars = plt.barh(y_pos,data,alpha=0.5,color='green') # 画水平柱状图，透明度为0.5，颜色为绿色
plt.vlines(min(data),-1,len(data)+0.5,linestyles='dashed') # 画垂直线，颜色为灰色，线宽为2
for b,d in zip(bars,data):
    plt.text(b.get_width()+b.get_width()*0.05,b.get_y()+b.get_height()/2,'{0:.2%}'.format(d/min(data)))
plt.show()


print(help(plt.boxplot))



fig,axes = plt.subplots(ncols=1,nrows=2,figsize=(12,5)) # 设置画布大小
sample_data = [np.random.normal(0,std,100) for std in range(6,10)] # 生成正态分布数据
axes[0].violinplot(sample_data,showmeans=False,showmedians=True) # 画小提琴图，不显示均值，显示中位数
axes[0].set_title('violin plot') # 设置标题
axes[1].boxplot(sample_data)
axes[1].set_title('box plot') # 设置标题
plt.show()

for ax in axes: # 将axes中的元素赋值给ax
    ax.yaxis.grid(True) # 设置y轴网格线
    ax.set_xticks([y+1 for y in range(len(sample_data))]) # 设置x轴刻度
    ax.set.xticklabels(['x1','x2','x3','x4']) # 设置x轴刻度标签
plt.show()


data = np.random.normal(0,20,1000) # 生成正态分布数据
bins = np.arange(-100,100,5) # 生成等差数列，从-100到100，步长为5
plt.hist(data,bins=bins) # 画直方图
plt.xlim([min(data)-5,max(data)+5]) # 设置x轴范围
plt.show()


import random
data1 = [random.gauss(15,10) for i in range(500)] # random.gauss()生成正态分布数据，for循环生成500个数据
data2 = [random.gauss(5,5) for i in range(500)]
bins = np.arange(-50,50,2.5)
plt.hist(data1,bins=bins,label='class 1',alpha=0.3) # 画直方图，透明度为0.3
plt.hist(data2,bins=bins,label='class 2',alpha=0.3)
plt.legend(loc='best')
plt.show()

N = 1000
x = np.random.randn(N)
y = np.random.randn(N)
plt.scatter(x,y,alpha=0.3)
plt.grid('True')
plt.show()



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax =  fig.add_subplot(111,projection='3d')
plt.show()

np.random.seed(1)
def randrange(n,vmin,vmax):
    return (vmax-vmin)*np.random.rand(n)+vmin
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
n = 100
for c,m,zlow,zhigh in [('r','o',-50,-25),('b','x',-30,-5)]:
    xs = randrange(n,23,32)
    ys = randrange(n,0,100)
    zs = randrange(n,int(zlow),int(zhigh))
    ax.scatter(xs,ys,zs,c=c,marker=m)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for c,z in zip (['r','g','b','y'],[30,20,10,0]):
    xs = np.arange(20)
    ys = np.random.rand(20)
    cs = [c]*len(xs)
    ax.bar(xs,ys,zs=z,zdir='y',color=cs,alpha=0.5)
plt.show()






##### Pandas #####

import pandas as pd

df = pd.read_csv('data.csv') 
df.head() # 查看前5行数据
df.tail() # 查看后5行数据

df.info() # 查看数据信息，df是Pandas中的DataFrame对象
df.index() # 查看索引
df.columns() # 查看列名
df.dtypes # 查看数据类型
df.values

age = df['Age'] # 选择Age列
age[:5] # 查看前5个值
age.values[:5] # 查看前5个值，

df = df.set_index('Name') # 将Name列设置为索引
df.head()

age = df['Age'] # 选择Age列
age['Allen, Mr. William Henry'] # 选择索引为Allen, Mr. William Henry的值

df[['Age','Fare']][:5] # 选择Age和Fare两列，查看前5行数据

df.iloc[0] # 选择第一行数据
df.iloc[0:5] # 选择前5行数据
df.iloc[[0,3,6,24],0:2] # 选择第1、4、7、25行，第1、2列的数据

df = df.set_index('Name') # 将Name列设置为索引
df.loc['Allen, Mr. William Henry'] # 选择索引为Allen, Mr. William Henry的行数据
df.loc['Allen, Mr. William Henry','Age'] # 选择索引为Allen, Mr. William Henry的行数据，Age列的值
df.loc['Allen, Mr. William Henry':'Heikkinen, Miss. Laina',:] # 选择索引为Allen, Mr. William Henry到Heikkinen, Miss. Laina的行数据
df.loc['Allen, Mr. William Henry','Age'] = 1000 # 将索引为Allen, Mr. William Henry的行数据，Age列的值修改为1000

df['Fare'] > 40 # 选择Fare列大于40的数据
df[df['Fare'] > 40][:5] # 选择Fare列大于40的数据，查看前5行数据
df[df['Sex'] == 'male'][:5] # 选择Sex列为male的数据，查看前5行数据
df.loc[df['Sex'] == 'male','Age'].mean() 
(df['Age'] > 70).sum() # 选择Age列大于70的数据，统计个数



# 创建dataframe
data = {'country':['Brazil','Russia','India','China','South Africa'],'popilation':[200.4,143.5,1252,1357,52.98]}
df.data = pd.DataFrame(data)
df.data

pd.get_option('display.max_rows') 
pd.set_option('display.max_rows',10) # 设置最大显示行数为10
pd.Series(index = range(0,100)) # 创建Series对象，索引为0-99

pd.get_option('display.max_columns')
pd.set_option('display.max_columns',30) # 设置最大显示列数为30
pd.DataFrame(columns=range(0,30))
 
# Series对象是dataframe中的一列，即dataframe是由Service对象组成的
data = [10,11,12]
index = ['a','b','c']
s = pd.Series(data=data,index=index)
s



s.loc['b'] # 选择索引为b的值
s.iloc[1] # 选择索引为1的值
s1 = s.copy() # 复制s
s1['a'] = 100 # 修改s1中索引为a的值
s1.replace(to_replace = 100,value = 101, inplace = True) # 将s1中的100替换为101, inplace=True表示替换原数据
s1.index
s1.index = ['a','b','c'] # 修改索引
s1.rename(index = {'a':'A'},inplace=True) # 将索引a修改为A

data = [100,110]
index = ['h','k']
s2 = pd.Series(data=data,index=index)
s3 = s1.append(s2) # 将s1和s2合并
s3['j'] = 500

del s1['A'] # 删除索引为A的值
s1.drop(['b','d'],inplace=True) # 删除索引为b和d的值


age = age + 10 
df = pd.DataFrame([[1,2,3],[4,5,6]],index = ['a','b'],columns=['A','B','C'])
df.sum()
df.sum(axis=1) # 按行求和
df.describe() # 查看数据的描述性统计信息

df = pd.read_csv('data.csv')
df.cov() # 查看协方差
df.corr() # 查看相关系数
df['Sex'].value_counts() # 查看Sex列的值的个数
df['Sex'].value_counts(ascending=True) # 查看Sex列的值的个数，按升序排列
df['Age'].value_counts(ascending=True,bins=5) # 查看Age列的值的个数，按升序排列，分为5组


ages = [15,18,20,21,22,34,41,52,63,79]
bins = [10,40,80]
bins_res = pd.cut(ages,bins) # 将ages按bins分组
bins_res.labels # 查看分组结果
pd.value_counts(bins_res) # 查看分组结果的个数
pd.cut(ages,[10,30,50,80])
group_names = ['Youth','Middle','Old']
pd.value_counts(pd.cut(ages,[10,20,50,80],labels=group_names)) 


# merge函数

left = pd.DataFrame({'key':['K0','K1','K2','K3'],'A':['A0','A1','A2','A3'],'B':['B0','B1','B2','B3']})
right = pd.DataFrame({'key':['K0','K1','K2','K3'],'C':['C0','C1','C2','C3'],'D':['D0','D1','D2','D3']})
res = pd.merge(left,right,on='key') # 将left和right按key列合并


left = pd.DataFrame({'key1':['K0','K1','K2','K3'],'key2':['K0','K1','K2','K3'],'A':['A0','A1','A2','A3'],'B':['B0','B1','B2','B3']})
righe = pd.DataFrame({'key1':['K0','K1','K2','K3'],'key2':['K0','K1','K2','K4'],'C':['C0','C1','C2','C3'],'D':['D0','D1','D2','D3']})
res = pd.merge(left,right,on=['key1','key2']) # 将left和right按key1和key2列合并
res = pd.merge(left,right,on=['key1','key2'],how='outer') # 将left和right按key1和key2列合并，how='outer'表示合并后的数据集包含所有的key1和key2的值
res = pd.merge(left,right,on=['key1','key2'],how='outer',indicator = Ture) # indicator=True表示显示合并方式
res = merge(left,right,how = 'left') # how='left'表示合并后的数据集包含left中所有的key1和key2的值，以left为主

data = pd.DataFrame({'k1':['one']*3+['two']*4,'k2':[3,2,1,3,3,4,4]})
data.drop_duplicates() # 删除重复的行
data.drop_duplicates(subset='k1') # 删除重复的行，以k1列为标准

df = pd.DataFrame({'data1':np.random.randn(5),'data2':np.random.randn(5)})
df2 = df.assign(ration = df['data1']/df['data2']) # 在df中添加一列ration，值为data1/data2
df = pd.DataFrame([range(3),[0,np.nan,0],[0,0,np.nan],range(3)] 

df.isnull() # 查看df中的值是否为空
df.isnull().any() # 查看df中的值是否为空，any()表示只要有一个为空就返回True
df.isnull().any(axis=1) # 查看df中的值是否为空，any()表示只要有一个为空就返回True，axis=1表示按行查看
df.fillna(5) # 将df中的空值用5填充


# apply()函数
data = pd.DataFrame({'food':['A1','A2','B1','B2','B3','C1','C2'],'data':[1,2,3,4,5,6,7]})
def food_map(series):
    if series['food'] == 'A1':
        return 'A'
    elif series['food'] == 'A2':
        return 'A'
    elif series['food'] == 'B1':
        return 'B'
    elif series['food'] == 'B2':
        return 'B'
    elif series['food'] == 'B3':
        return 'B'
    elif series['food'] == 'C1':
        return 'C'
    elif series['food'] == 'C2':
        return 'C'
data['food_map'] = data.apply(food_map,axis = 'columns') # 将food列的值映射到food_map列中

def nan_count(columns):
    columns_null = pd.isnull(columns)
    null = columns[columns_null]
    return len(null)
columns_null_count = titanic.apply(nan_count) # 统计每一列的空值个数


def is_minor(row):
    if row['age'] < 18:
        return Ture
    else:
        return False
minors = titanic.apply(is_minor,axis = '1') # 统计每一行的age是否小于18



%matplotlib inline
df = pd.DataFrame(np.random.randn(10,4).cumsum(0),columns=['A','B','C','D'],index=np.arange(0,100,10))
df.plot()

import matplotlib.pyplot as plt
fig.axes = plt.subplots(2,1)
data = pd.Series(np.random.rand(16),index=list('abcdefghijklmnop'))
data.plot(ax = axes[0],kind='bar')
data.plot(ax = axes[1],kind='barh')

df = pd.DataFrame(np.random.rand(6,4),index=['one','two','three','four','five','six'],columns=pd.Index(['A','B','C','D'],name='Genus'))
df.plot(kind='bar')
macro = pd.read_csv('macrodata.csv') # 读取数据
data.plot.scatter('quarter','realgdp')




























