# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy import Item, Field


class Wine(Item):
    # 数据结构应面向可建立表格的 Excel
    name_cn = Field()  # 中文名
    name_en = Field()  # 英文名
    type = Field()  # 酒款类型
    grapes = Field()  # 酿酒葡萄
    factory = Field()  # 酒庄
    location = Field()  # 产区
    favor = Field()  # 风味特征
    age = Field()  # 酒款年份
    market_price = Field()  # 市场参考价
    price = Field()  # 红酒世界会员商城价
    score = Field()  # 分数，100分制
    click = Field()  # 点击次数

