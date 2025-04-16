# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:42:42 2025

@author: Lucius
"""

import geopandas as gpd
import matplotlib.pyplot as plt

# 加载手动下载的 shapefile 文件
world = gpd.read_file(r"C:\Users\Lucius\Downloads\110m_cultural\ne_110m_admin_0_countries.shp")

# 假设你要高亮显示的国家列表
highlight_countries = ['Vanuatu', 'Guam', 'Seychelles', 'Somalia', 'Mali', 'Bolivia', 'Yemen', 'Liechtenstein', 'Lesotho', 'Bangladesh', 'Aruba', 'Andorra', 'Cambodia', 'Papua New Guinea', 'Honduras', 'Belize', 'São Tomé and Príncipe', 'Lebanon', 'Bhutan', 'Oman', 'Eswatini', 'Sierra Leone', 'Marshall Islands', 'El Salvador', 'Kiribati', 'Angola', 'American Samoa', 'Comoros', 'Madagascar', 'Cayman Islands', 'Antigua and Barbuda', 'Democratic Republic of the Congo', 'Central African Republic', 'The Gambia', 'Guinea-Bissau', 'Nicaragua', 'Benin', 'Palau', 'Myanmar', 'Guinea', 'Saint Kitts and Nevis', 'Malta', 'Timor-Leste', 'Liberia', 'Brunei', 'British Virgin Islands', 'Mauritania', 'Palestine', 'Equatorial Guinea', 'South Sudan', 'Republic of the Congo', 'Maldives', 'Saint Vincent and the Grenadines', 'Solomon Islands', 'Federated States of Micronesia', 'Laos', 'Tuvalu', 'Chad', 'Samoa', 'Rwanda', 'Malawi', 'Bosnia and Herzegovina', 'Cook Islands', 'Nepal', 'Libya', 'Nauru']

# 选择需要高亮显示的国家，修改其颜色
world['color'] = world['ADMIN'].apply(lambda x: 'darkblue' if x in highlight_countries else 'lightgray')

# 创建绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制地图
world.plot(ax=ax, color=world['color'])

# 隐藏坐标轴
ax.set_axis_off()

# 显示图像
plt.title('As of Present, Countries That Have Never Won an Olympic Medal')
plt.show()