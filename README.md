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

### 国家标准

 [《联机手写汉字识别系统技术要求与测试规程(GB/T 18790-2010)》]( https://blog.csdn.net/m0_37657841/article/details/88672119 )

## 基本结构

### I/O模块

- 输入模块（input）

- 输出模块（output）

### 预处理模块

- 滤波（smoothing）

- 二值化（binaryzation）

- 倾斜校正（tilt_coorrection）
- 细化（refining）
- 毛刺消除（remove——burr）

### 字符分割模块

### 模型（使用pytorch）

- RNN
- CNN
- LSTM
- SAE
- DBM