import numpy as np
import pandas as pd
from semopy import Model
from semopy import calc_stats
from graphviz import Digraph

# 读取数据
df = pd.read_excel(r"C:\Users\Lucius\Desktop\问卷数据 填充空值.xlsx")

# 定义路径模型
model_desc = """
# 测量模型
User =~ Age + Spending + Gender_Female + Identity_graduate + Identity_part + Identity_work
Emo =~ Loneliness + Depre 
Reason =~ ReasonLonely + ReasonEmo + ReasonSupport + ReasonSocial + ReasonEnter + ReasonNovel + ReasonOther 
Exper=~ ExperOverall + ExperNature + ExperSupport + ExperPrivacy + ExperPersonal
Impor =~ ImporNature + ImporSupport + ImporPrivacy + imporPersonal 
Scenario =~ ScenarioSleep + ScenarioWork + ScenarioDown + ScenarioNovel + ScenarioTravel + ScenarioOther 
NotReason =~ NotReasonProduct + NotReasonNece + NotReasonPrivacy + NotReasonOther
Accept =~ Try + ConcernSocial + ConcernMoral + ConcernPrivacy + ConcernExper + ConcernNot + ConcernOther 
WTP =~  Pay + Price
Freq =~ UsedFreq_sometimes + UsedFreq_heard + UsedFreq_never


# 结构模型
Freq ~ User + Emo
Impor ~ Reason + Exper + Freq
Accept ~ Exper + Impor + NotReason
WTP ~ Accept
Market ~ Freq + Exper + WTP + Accept + User
"""

# 创建SEM模型
model = Model(model_desc)

# 将数据加载到模型中
model.fit(df)

# 输出模型估计结果
params = model.inspect()
print("模型参数估计结果：")
print(params)
params.to_excel(r"C:\Users\Lucius\Desktop\11SEM模型参数.xlsx", index=False)

# 计算拟合度
fit = calc_stats(model)  
print("\n拟合度指标：")
print(fit.T)
fit.to_excel(r"C:\Users\Lucius\Desktop\11SEM拟合度.xlsx", index=False)

# 创建一个空的有向图（Digraph）
dot = Digraph(comment='SEM Path Diagram')

# 定义颜色
node_color = '#f8d8a4'  # 浅色
edge_color = '#ebb089'  # 中性色
highlight_color = '#588797'  # 深色

# 调整节点和边的间距
dot.attr(dpi='300')  # 设置图像分辨率
dot.attr('graph', ranksep='1.2', nodesep='0.8')  # 增加节点之间的距离和节点与边之间的间距

# 设置字体大小和字体样式
dot.attr('node', fontname='Helvetica', fontsize='12')  # 设置字体和字号
dot.attr('edge', fontname='Helvetica', fontsize='10')  # 设置边的字体和字号

# 绘制路径图
dot.node('user', 'user', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('emo', 'emo', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('Reason', 'Reason', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('Exper', 'Exper', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('Impor', 'Impor', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('Scenario', 'Scenario', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('NotReason', 'NotReason', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('Accept', 'Accept', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('WTP', 'WTP', style='filled', fillcolor=node_color, fontcolor='black')
dot.node('Freq', 'Freq', style='filled', fillcolor=node_color, fontcolor='black')

# 测量模型路径
dot.edge('user', 'user', label='Age, Spending, Gender_Female, Identity_graduate, Identity_part, Identity_work', color=edge_color)
dot.edge('emo', 'emo', label='Loneliness, Depre', color=edge_color)
dot.edge('Reason', 'Reason', label='ReasonLonely, ReasonEmo, ReasonSupport, ReasonSocial, ReasonEnter, ReasonNovel, ReasonOther', color=edge_color)
dot.edge('Exper', 'Exper', label='ExperOverall, ExperNature, ExperSupport, ExperPrivacy, ExperPersonal', color=edge_color)
dot.edge('Impor', 'Impor', label='ImporNature, ImporSupport, ImporPrivacy, imporPersonal', color=edge_color)
dot.edge('Scenario', 'Scenario', label='ScenarioSleep, ScenarioWork, ScenarioDown, ScenarioNovel, ScenarioTravel, ScenarioOther', color=edge_color)
dot.edge('NotReason', 'NotReason', label='NotReasonProduct, NotReasonNece, NotReasonPrivacy, NotReasonOther', color=edge_color)
dot.edge('Accept', 'Accept', label='Try, ConcernSocial, ConcernMoral, ConcernPrivacy, ConcernExper, ConcernNot, ConcernOther', color=edge_color)
dot.edge('WTP', 'WTP', label='Pay, Price', color=edge_color)
dot.edge('Freq', 'Freq', label='UsedFreq_sometimes, UsedFreq_heard, UsedFreq_never', color=edge_color)

# 结构模型路径
dot.edge('Freq', 'user', label='', color=highlight_color)
dot.edge('Freq', 'emo', label='', color=highlight_color)
dot.edge('Impor', 'Reason', label='', color=highlight_color)
dot.edge('Impor', 'Exper', label='', color=highlight_color)
dot.edge('Impor', 'Freq', label='', color=highlight_color)
dot.edge('Accept', 'Exper', label='', color=highlight_color)
dot.edge('Accept', 'Impor', label='', color=highlight_color)
dot.edge('Accept', 'NotReason', label='', color=highlight_color)
dot.edge('WTP', 'Accept', label='', color=highlight_color)

# 保存并展示路径图
dot.render(r'C:\Users\Lucius\Desktop\SEM_Paths_with_color_and_spacing', format='png', view=True)
