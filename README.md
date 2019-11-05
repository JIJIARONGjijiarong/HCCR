#  手写体汉字识别（HCCR）

## 目标：脱机手写汉字识别系统

目标按层次实现

1. 实现单个简单100汉字手写体的识别
   1. 完成预处理模块
2. 实现单个常用3000汉字手写体的识别
3. 实现联想输入
4. 实现单行文本的识别
5. 扩展
   1. 语音识别 
   2. 文本编辑器的构建
   3. 联机手写汉字识别系统

## 基本结构

### I/O模块（opencv）

- gnt转png模块 周义青finish

  [关于gnt文件格式的说明]( http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html )：

  - In brief, each file has a header with the header size given as the first 4-byte integer number in the file. The last two integer numbers in the header give the number of samples in the file and the feature dimensionality. Following the header are the records of all samples, each sample including a 2-byte label (GB code) and the feature vector, each dimension in a unsigned char byte. 
  - [gnt文件格式详细说明]( http://www.nlpr.ia.ac.cn/databases/download/feature_data/FileFormat-mpf.pdf )

- 输入模块（input）籍家荣 finish

- 输出模块（output）籍家荣 finish

### 预处理模块（opencv）

- 图片放缩（周义青）finish
- 滤波降噪（smoothing）籍家荣 
- 彩色图像转灰度图像（邬洲）
- 灰度图像二值化使用 OTSU（binaryzation）邬洲
- 倾斜校正（tilt_coorrection）籍家荣 finish
- 细化（refining）张炳辰finish
- 毛刺消除（remove_burr）周义青defeat

#### 字符分割模块 暂未确定人选情况

### 模型（使用pytorch）暂未确定人选情况

考虑具体梯度下降方法

过程中观察学习情况

- RNN
- CNN
- LSTM
- SAE
- DBM 

集成学习

## 计划

### 第一阶段（第一周：10.28——11.3）

- 完成预处理模块
- 模型的学习
- 预处理模块的测试

### 第二阶段（第二、三周）（11.4——11.17）

- 预处理模块的BUG修复
- 模型的学习
- 模型的选择
- 模型的训练（cpu版本）（100字或一个gnt文件）

### 第三阶段（第四周）

- 模型的改进、修补
- cpu版本的模型改为gpu版本的模型

### 第四阶段（第五周）

- 模型的大规模训练
- 调参

### 第五阶段

- 联想输入
- 扩展计划

## 超链接

[《联机手写汉字识别系统技术要求与测试规程(GB/T 18790-2010)》]( https://wenku.baidu.com/view/3ea05fe603d276a20029bd64783e0912a2167cab.html )

[CASIA Online and Offline Chinese Handwriting Databases]( http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html )

[gnt文件格式详细说明]( http://www.nlpr.ia.ac.cn/databases/download/feature_data/FileFormat-mpf.pdf )