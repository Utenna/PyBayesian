<div style="text-align: center;">	
	
# 高级心理统计[Advanced Statistics in Psychological Science]
## 《贝叶斯统计及其在Python中的实现》 [Bayesian inference in Python]
## Instructor： 胡传鹏（博士）[Dr. Hu Chuan-Peng]
### 南京师范大学心理学院[School of Psychology, Nanjing Normal University]
	
</div>

<div style="text-align: center;">
	
研究人类心理与行为的规律，容易吗？
	
</div>

## Outlines
* 1. 为什么要学习本课程 [Why Bayesia inference]
* 2. 本课程的内容将是什么 [What is the syllabus]
* 3. 如何学好这门课[How can I learn this course well]

##  1. 为什么要学习本课程 [Why Bayesia inference]

### 1.1 为什么心理学需要更好的方法【Why does psychological science need better methods?]

#### 原因1: 复杂的研究问题

![Image Name](https://cdn.kesci.com/upload/image/rhdcyu860w.gif?imageView2/0/w/960/h/960)


Source: https://www.science.org/toc/science/309/5731

Q1: What is the Uiverse Made of [physics ]

Q2: What is the Biological Basis of Consciouness [psychological science]

##### 问题
同样重要和复杂的问题，是否意味着类似复杂和高级的方法？

##### 物理学中的方法 [Methods in Physics]:

Example 1: Webb telescope (韦伯望远镜)  [**equipment**]


![Image Name](https://cdn.kesci.com/upload/image/rhdd0r46k3.png?imageView2/0/w/720/h/640)


Example 2: Big-team science (CERN, the European Organization for Nuclear Research) [**equipment & practices**]

Example 3: **Mathematics**

##### 其他研究人类智能的领域所采用的方法 [Methods in other fields that also study "intelligence"]

**AI**


![Image Name](https://cdn.kesci.com/upload/image/rhdd1sr5y2.png?imageView2/0/w/640/h/640)


##### 心理科学的研究方法 [What do psychological scientists have?]
你们能够想到的研究方法包括哪些？


![Image Name](https://cdn.kesci.com/upload/image/rhdd2dgwc8.png?imageView2/0/w/640/h/640)


**实证研究：**
* 质性研究
* 观察法
* 问卷
* 行为实验
* 眼动、生理数据记录
* EEG/ERP/MEG
* fMRI/PET/fNIRs
* TMS/tDCS
* ...

**统计方法：**
* t-test
* ANOVA
* Correlation
* Structural equation model (SEM)
* ?

##### 相关方法课程：
* 心理测量
* 心理统计（包括SPSS等）
* 实验心理学（包括Eprime等）
* ？

* 更好的仪器
* **更好的统计/数据分析**
* 更好的实践 (e.g., big-team science)

#### 原因2: 更复杂的数据

* 数据字化的时代，大数据
* 神经成像/生理数据
* 多模态的数据融合

### 1.2 确实有更好的统计方法

贝叶斯统计 (Bayesian inference)


![Image Name](https://cdn.kesci.com/upload/image/rhdf3bb12c.png?imageView2/0/w/640/h/640)



* 灵活/强大/能用
* 易用
* 可拓展性强
* 方便交流
* ...

##### 灵活/强大/通用

不需要解析解

贝叶斯分析在多个学科中得到广泛应用，尤其是AI

##### （相对）易用

概率编程语言(Probabilistic Programming Languages)的发展和普及


PPLs: *computational languages for statistical modeling*

* PyMC
* Stan
* NumPyro
* Pyro
* BUGS
* ...

大部分情况下，开发者使用它可以轻松地定义概率模型，然后程序会自动地求解模型。


![Image Name](https://cdn.kesci.com/upload/image/rhdf4r9fbh.png?imageView2/0/w/640/h/640)


Source: https://towardsdatascience.com/intro-to-probabilistic-programming-b47c4e926ec5


##### 可拓展

贝叶斯概念已经应用到以深度学习为中心的新技术的发展，包括深度学习框架(TensorFlow, Pytorch)，创建表示能力更强、数据驱动的模型

##### 方便交流
大部分PPLs都有类似的数据结构，但是不同的学科使用的语言不同。

心理学/社会科学/神经科学：
* **PyMC3**
* Stan
* BUGS

<div style="text-align: center;">	
	
# part 2
	
</div> 

### 例1：社会关系地位与幸福感的关系

实例的数据来自[ Many Labs 2 项目](osf.io/uazdm/)中的一个研究。

该研究探究了社会关系地位对于幸福感的影响 “Sociometric status and well-being”， (Anderson, Kraus, Galinsky, & Keltner, 2012)。

该数据集包括6905个被试的数据。


```python
# import modules
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import xarray as xr

%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(1234)

import matplotlib
matplotlib.rcParams['figure.figsize'] = [4, 3]
```

    WARNING (theano.link.c.cmodule): install mkl with `conda install mkl-service`: No module named 'mkl'
    


```python
# 导入数据
SMS_data = pd.read_csv('./bayesian-analysis-nnupsy/Notebooks/data_chp1_SMS_Well_being.csv')[['uID','variable','factor','Country']]

# 把数据分为高低两种社会关系的地位的子数据以便画图与后续分析
plot_data = [
    sorted(SMS_data.query('factor=="Low"').variable[0:3000]),
    sorted(SMS_data.query('factor=="High"').variable[0:3000])]
```

#### 通过画图对于两种社会关系地位对幸福感的影响

图中横坐标代表高低两种社会关系地位，纵坐标代表了主观幸福感评分。


```python
# import matplotlib
# a = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# for i in a:
#    print(i)

# 字体样式
font = {'family' : 'Source Han Sans CN'}
# 具体使用
plt.rc('font',**font)
```


```python
# 画图对比两种社会地位对幸福感的影响
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('社会关系地位')

fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), sharey=True)

parts = ax1.violinplot(
        plot_data, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1, medians, quartile3 = np.percentile(plot_data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(plot_data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
ax1.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax1.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax1.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# set style for the axes
labels = ['低','高']
plt.xticks(np.arange(2)+1, labels)
plt.xlabel('社会关系地位')
plt.ylabel('幸福感')

plt.subplots_adjust(bottom=0.15, wspace=0.05)
plt.show()
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:45: UserWarning: This figure was using constrained_layout, but that is incompatible with subplots_adjust and/or tight_layout; disabling constrained_layout.
    


<img src="https://cdn.kesci.com/upload/rt/59C5972D76C34D0CB9137A7B46DD0154/rhdfv9kcli.png">


#### 通过t检验，分析两种社会关系地位下幸福感的差异

结果发现，两种社会关系水平下被试的主观幸福感边缘显著，*t*(6903) = -1.76, p = .08。


```python
from scipy import stats
SMS_low = SMS_data.query('factor=="Low"').variable.values
SMS_high = SMS_data.query('factor=="High"').variable.values
print(
    f"低社会关系：{np.around(np.mean(SMS_low),3)} ± {np.around(np.std(SMS_low),2)}；",
    f"高社会关系：{np.around(np.mean(SMS_high),3)} ± {np.around(np.std(SMS_high),2)}")
    
stats.ttest_ind(
    a= SMS_low,
    b= SMS_high, 
    equal_var=True)
```

    低社会关系：0.014 ± 0.66； 高社会关系：-0.014 ± 0.67
    


    Ttest_indResult(statistic=1.7593310889762195, pvalue=0.07856558333862036)


#### 通过贝叶斯推断替代*t*检验

零假设显著性检验（Null hypothesis significance test, NHST）的框架之下，*t*检验只提供了一个二分的结果：拒绝或者无法拒绝$H_0$。 但 *p* = 0.078这样的结果无法支持$H_0$

贝叶斯推断是否可以带来不一样的结果？

一个简单的线性模型：

1. 通过建立线性模型去替代原本的*t*检验模型。

2. 通过PyMC对后验进行采样

3. 通过Arviz对结果进行展示，辅助统计推断


```python
# 通过pymc建立基于贝叶斯的线性模型
x = pd.factorize(SMS_data.factor)[0] # high为0，low为1

with pm.Model() as linear_regression:
    sigma = pm.HalfCauchy("sigma", beta=2)
    β0 = pm.Normal("β0", 0, sigma=5)
    β1 = pm.Normal("β1", 0, sigma=5)
    x = pm.Data("x", x)
    # μ = pm.Deterministic("μ", β0 + β1 * x)
    pm.Normal("y", mu=β0 + β1 * x, sigma=sigma, observed=SMS_data.variable)
```

可以通过pymc自带的可视化工具将模型关系可视化。

x 为自变量，其中1为低社会关系，0为高社会关系。

参数 $\beta0$ 是线性模型的截距，而 $\beta1$ 是斜率。

截距代表了高社会关系地位被试的幸福感；而截距加上斜率表示低社会关系地位被试的幸福感。

参数$sigma$是残差，因变量$y$即主观幸福感。

模型图展示了各参数通过怎样的关系影响到因变量。


```python
pm.model_to_graphviz(linear_regression)
```


<img src="https://cdn.kesci.com/upload/rt/BECEC5A3264F49F59EF4D89F47F57785/rhdfvbd656.svg">



```python
# 模型拟合过程 (mcmc采样过程)
with linear_regression:
    idata = pm.sample(2000, tune=1000, target_accept=0.9, return_inferencedata=True)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (4 chains in 4 jobs)
    NUTS: [β1, β0, sigma]
    



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>



    Sampling 4 chains for 1_000 tune and 2_000 draw iterations (4_000 + 8_000 draws total) took 6 seconds.
    

#### 参数的后验分布
这里的模型分析结果展示了各参数的分布(后验)情况


```python
az.plot_trace(idata);
```


<img src="https://cdn.kesci.com/upload/rt/E54DEC474F6049F89408157C78B15D7D/rhdfvpdp25.png">


下图反应了参数β1的可信度，即两个社会关系下幸福感差异的可信度。

结果显示，两个社会关系下幸福感差异的可信度为96%。


```python
(idata.posterior.β1 > 0).mean().values
```


    array(0.960125)



```python
az.plot_posterior(idata, var_names=['β1'], kind='hist',ref_val=0)
```


    <AxesSubplot:title={'center':'β1'}>



<img src="https://cdn.kesci.com/upload/rt/F500A43067C4488F94A672EB6A77BE57/rhdfvpxyyg.png">



```python

az.plot_posterior(idata, var_names=['β1'], kind='hist', rope = [-0.1, 0.1], hdi_prob=.95)
```


    <AxesSubplot:title={'center':'β1'}>



<img src="https://cdn.kesci.com/upload/rt/3E302511A7FB4C0E8BFBD80013CBCA5E/rhdfvpeh73.png">


#### 模型诊断

通过模型思维进行数据分析需要注意模型检验，即检验模型是否能有效的反应数据的特征。

下表格为模型参数的基本信息：

mean和sd 为各参数的均值和标准差；
hdi 3%-97% 为参数分布的可信区间；
msce mean和sd 为mcmc采样标准误统计量的均值和标准差；
ess bulk和tail 反应了mcmc采样有效样本数量相关性能；
r hat 为参数收敛性的指标。


```python
az.summary(idata)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>β0</th>
      <td>-0.014</td>
      <td>0.011</td>
      <td>-0.034</td>
      <td>0.008</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4233.0</td>
      <td>4505.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>β1</th>
      <td>0.028</td>
      <td>0.016</td>
      <td>-0.002</td>
      <td>0.058</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4381.0</td>
      <td>4873.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.661</td>
      <td>0.006</td>
      <td>0.650</td>
      <td>0.671</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4904.0</td>
      <td>4846.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


### 后验预测检验 ppc (posterior predictive check)


```python
with linear_regression:
    pm.set_data({"x": np.array([0,1])})
    ppc_y = pm.sample_posterior_predictive(idata, var_names=["y"],keep_size=True)["y"] # keep size不是data的size 而是mcmc的size
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='8000' class='' max='8000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [8000/8000 00:21&lt;00:00]
</div>




```python
labels = ['低', '高']
obs_low = SMS_data.query('factor=="Low"').variable
obs_high = SMS_data.query('factor=="High"').variable
ppc_y2 = [j for i in ppc_y for j in i]
ppc_low = [i[1] for i in ppc_y2]
ppc_high = [i[0] for i in ppc_y2]
# reg_post = idata.posterior.stack(chain_draw=("chain", "draw"))
# ppc_x = np.repeat([0,1],len(reg_post.sigma)/2)
# ppc_y = reg_post['β0'] + reg_post['β1']*ppc_x
# ppc_low = ppc_y[ppc_x==1].values
# ppc_high = ppc_y[ppc_x==0].values

fig, ax = plt.subplots()
part1 = ax.violinplot(
    [list(obs_low),list(obs_high)], 
    [1,4], points=100, widths=0.3, 
    showmeans=True, showextrema=True, showmedians=True)
part2 = ax.violinplot(
    [list(ppc_low),list(ppc_high)], 
    [2,5], points=100, widths=0.3, 
    showmeans=True, showextrema=True, showmedians=True)
part1['bodies'][0].set_label('观测数据')
part2['bodies'][0].set_label('预测数据')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('幸福感')
ax.set_title('Posterior predictive check')
plt.xticks([1.5,4.5], labels)
ax.legend()

fig.tight_layout()
plt.show()
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:30: UserWarning: This figure was using constrained_layout, but that is incompatible with subplots_adjust and/or tight_layout; disabling constrained_layout.
    


<img src="https://cdn.kesci.com/upload/rt/D27A38FFC2E14DE7AE59DED2B16E366B/rhdfwdjayt.png">


### 例2: 贝叶斯推断在认知模型中的应用

在实验数据的收集时，研究者往往会采集个体反应的正确率与反应时。

而传统分析方法并不能同时对两种数据进行分析，从而推断潜在的认知机制。比如，个体是否愿意牺牲更多的反应时间去获得一个更准确的判断。

认知模型能有效的弥补这一问题，比如 drift-diffusion model, DDM。


![DDM1](https://cdn.kesci.com/upload/image/rhb2957an5.png?imageView2/0/w/960/h/960)


<div style="text-align: center;">	
	
# part 3
	
</div> 

# 高级心理统计[Advanced Statistics in Psychological Science]
## 《贝叶斯统计及其在Python中的实现》 [Bayesian inference in Python]
## Instructor： 胡传鹏（博士）[Dr. Hu Chuan-Peng]
### 南京师范大学心理学院[School of Psychology, Nanjing Normal University]

# 2. 课程内容
### 2.0 课时
8.29 ～ 12.30，18周

### 2.1 教学目标：

（1）理解贝叶斯推断的基本原理；

（2）了解PyMC3的语法和结构；

（3）可以使用PyMC3解决相对简单的统计推断问题（如层级线性模型）


### 2.2 考核方式：

#### 考勤
10%




#### 小作业
45%
	
（1）简单的代码作业 notebook
	
（2）概念理解 notebook
	
（3）workflow notebook

#### 大作业：
真实的数据分析 45%
* 合作完成
* 包括代码与文字报告
* 进行汇报 
* 标准：分工合理、数据分析流程完整、汇报展示清晰美观


### 2.3 课程风格：
	（1）内容有挑战、考核不复杂
	（2）1/3一节课展示或互动抄代码
	（3）专门设有答疑时间，助教给大家答疑解惑





### 2.4 课程大纲
#### 0 Intro （第一课）
* 1 课程介绍
#### I Basics：
* 2 贝叶斯与频率主义的对比、概率（离散、连续）/条件概率；
* 3 贝叶斯法则/联合分布；
* 4 ～ 6. Likelihood, Prior （PPC）, Denominator, Posterior （PPC）[student’s Guide, Part II]

#### II 现代贝叶斯统计的内在工作机制（sampler）
* 7 MCMC

#### III Bayesian Workflow
* 8 LM + PyMC3
* 9 诊断
* 10 比较
* 11 推断

#### IV Applications
* 12 GLM & more LM
* 13 层级模型 LMM （RT/调查数据）
* 14 GLMM (信号检验论)
* 15 扩展示例

#### V 讲座与作业展示
* 学术报告：劳俊鹏博士 (Google)
* 大作业

#### 2.5 参考书


![Image Name](https://ts1.cn.mm.bing.net/th?id=AMMS_56bf8f54d4ebfaa69e6430302ae0ea6d&w=100&h=150&c=7&rs=1&qlt=80&pcl=f9f9f9&o=6&cdv=1&dpr=2&pid=16.1)


![Image Name](https://tse1-mm.cn.bing.net/th/id/OIP-C.KSBduDlouNql_1z-nwkohgAAAA?pid=ImgDet&rs=1)



![Image Name](https://tse3-mm.cn.bing.net/th/id/OIP-C.sypTLOUunm6FyDTJrVKQTgHaL3?w=119&h=191&c=7&r=0&o=5&dpr=2&pid=1.7)

