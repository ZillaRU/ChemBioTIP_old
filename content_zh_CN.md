面向化学药、生物药的多类型药物互作用预测框架设计

随人口老龄化趋势日益明显，药物互作用管理已成为重大的公共卫生问题。（预测药物互作用对临床用药和药物重定位有重大参考价值。）
然而，现有药物互作用建模仅考虑小分子化学药，而未考虑作为后起之秀的大分子生物药；
多数现有药物互作用模型仅预测药物间是否能够互作用，而不关注具体互作用类型，严重限制了临床应用。

因此，本项目组研究了涉及生物药和化学药的多类型药物互作用预测问题。
（1）整合多种数据源，构建增强的多视图属性网络，利用分子结构为每个分子构建含属性的内层分子图，
并且利用药物-靶点，靶点-靶点间的互作用关系，增强药物互作用网络的连接性，构建连接性增强的外层互作用图。
（2）在此基础上，搭建多视图端到端图神经网络模型，内层图编码模块分别为小分子、大分子的内层网络编码并得到每个分子的内层表示，该内层表示作为外层图的节点属性输入到外层图。
随后，互作用类型敏感的外层图编码模块抽取各药物、靶点的外层图表示。
（3）以对比的方式对齐内层、外层表示，给出某两种药物发生某类型互作用的可能性评分。
在“仅化学药参与”、“所有药参与”两个场景下真实数据集上的实验证实了本项目组所提出的药物互作用预测框架的有效性。
研究成果拟投稿VLDB（数据库领域TOP国际会议，CCF A类国际会议）。

# 1. Introduction
# 2. Preliminaries
small molecule - SMILES - graph
macro molecule - AA seq - contact map - graph
drug
target
DDI
DTI
TTI
formulation（只定义输入输出 问题本身 不涉及方法）

# 3. Multi-view graph construction
definition： intra-view graph
definition： enhanced inter-view graph
definition：multi-view graph
construction

# 4. Graph-based Prediction Model
# 5. Evaluation
## 场景1：overall、each type
## 场景2：overall、each type，biotech drugs
Parameter Sensitivity：alpha beta gamma #n_layers
exploring the embedding in ChemBioIP （understanding）：DDI type，drugs（with category labels）
Ablation Study（graph-construction、concat or ————）
# 6，Related Works
# 7. Conclusion
