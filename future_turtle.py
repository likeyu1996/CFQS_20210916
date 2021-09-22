#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    Created by Klaus Lee on 2021/9/16
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import datetime
import matplotlib.pyplot as plt
import os
import mplfinance as mpf

# 解决显示图形中中文乱码问题
plt.rcParams['font.sans-serif'] = ['Kaiti']
plt.rcParams['axes.unicode_minus'] = False

# numpy完整print输出
np.set_printoptions(threshold=np.inf)
# pandas完整print输出
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 各种路径
ROOT_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT_PATH, 'Data')
RESULT_PATH = os.path.join(ROOT_PATH, 'Result')


class FutureTransaction:
    # 期货交易类 交易账户
    def __init__(self, uid, date, name, variety, contract, position, price, volume, charge):
        if volume == 0 or position == 0:
            print('Cannot set zero volume or position')
            return
        self.uid = uid
        self.date = date
        self.name = name
        self.variety = variety
        self.contract = contract
        self.position = position
        self.price = price
        self.volume = volume
        self.charge = charge

        # 根据交易合约细则修改 TODO:自动填充
        self.leverage = 200
        self.per_lot_unit = 10
        self.point = 1

        # 交易状态分类：开(多/空)仓、增仓(海龟交易法中一退就退所有头寸)、平(多/空)仓
        # 记录进场时间、进场方向、进场价格、进场仓位
        # 仓位方向由position正负决定，1为多仓，-1为空仓
        # 增仓次数由times决定，每次增仓就执行times+1
        self.add_pos = {'date': [self.date],
                        'position': [self.position],
                        'price': [self.price],
                        'volume': [self.volume],
                        'times': 1,
                        'note': ['OPEN']
                        }

        self.date_clean = 0
        self.price_clean = 0
        self.total_price = np.sum(np.array(self.add_pos['price'])*np.array(self.add_pos['volume']))
        self.total_volume = np.sum(self.add_pos['volume'])
        self.average_price = self.total_price/self.total_volume
        self.total_profit = 0
        self.net_profit = 0
        self.transaction_status = 'open'
        self.commission = 0  # TODO:手续费
        self.transaction_max_potential_profit = 0
        self.transaction_max_potential_loss = 0
        self.transaction_current_profit = 0

    def update(self, high_price, low_price, settle_price):
        # 仓位数据更新，可用于计算最大回撤和最大浮盈
        if self.position > 0:
            # 当前仓位为多仓
            # 计算浮盈
            max_potential_profit = round((high_price - self.average_price)*self.total_volume*self.per_lot_unit, 2)
            # 计算回撤
            max_potential_loss = round((self.average_price - low_price)*self.total_volume*self.per_lot_unit, 2)
            # 判定浮盈/回撤是否是最大浮盈/最大回撤
            if max_potential_profit > self.transaction_max_potential_profit:
                self.transaction_max_potential_profit = max_potential_profit
            if max_potential_loss > self.transaction_max_potential_loss:
                self.transaction_max_potential_loss = max_potential_loss
        elif self.position < 0:
            # 当前仓位为空仓
            # 计算浮盈
            max_potential_profit = round((low_price - self.average_price)*self.total_volume*self.per_lot_unit, 2)
            # 计算回撤(绝对值)
            max_potential_loss = round((self.average_price - high_price)*self.total_volume*self.per_lot_unit, 2)
            # 判定浮盈/回撤是否是最大浮盈/最大回撤
            if max_potential_profit > self.transaction_max_potential_profit:
                self.transaction_max_potential_profit = max_potential_profit
            if max_potential_loss > self.transaction_max_potential_loss:
                self.transaction_max_potential_loss = max_potential_loss
        else:
            print('ERROR update')
        self.transaction_current_profit = round((settle_price - self.average_price)*self.total_volume*self.per_lot_unit, 2)
        # TODO:手续费
        self.commission = self.set_commission()

    def increase(self, date, position, price, volume, note):
        # 判定是否同向（不然不叫加仓）
        if position != self.position:
            print('ERROR increase')
            return -1
        # 更新加仓细节
        self.add_pos['date'].append(date)
        self.add_pos['position'].append(position)
        self.add_pos['price'].append(price)
        self.add_pos['volume'].append(volume)
        self.add_pos['times'] += 1
        self.add_pos['note'].append(note)
        # 更新部分指标
        self.total_price = np.sum(np.array(self.add_pos['price'])*np.array(self.add_pos['volume']))
        self.total_volume = np.sum(self.add_pos['volume'])
        self.average_price = self.total_price/self.total_volume
        pass

    def decrease(self, date, position, price, volume, note):
        # 海龟交易法直接清仓，不存在减仓的情况，因此不考虑
        pass

    def clean(self, date, price):
        # 判断是否有仓位
        if self.position == 0:
            print('ERROR, no position')
            return -1
        # 记录清仓时间和价格
        self.date_clean = date
        self.price_clean = price
        # 更新指标
        self.transaction_status = 'close'
        self.position = 0
        # 计算指标
        self.commission = self.set_commission()
        self.total_profit = round((self.price_clean - self.average_price)*self.total_volume*self.per_lot_unit, 2)
        self.net_profit = self.total_profit - self.commission
        pass

    def set_commission(self):
        commission = 0
        return commission


class FutureStrategy:
    def __init__(self, n_1, n_2, n_3, atr_parameter, capital_control_ratio, start_date, end_date,
                 data, account, min_volume, max_add_times):
        # 导入参数
        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.atr_parameter = atr_parameter
        self.capital_control_ratio = capital_control_ratio
        self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        self.account_initial = account
        self.account = account
        self.min_volume = min_volume
        self.max_add_times = max_add_times

        # 数据处理和指标计算(默认index为date)
        self.df = data[self.start_date:self.end_date]
        self.df = self.set_donchian()
        self.df = self.set_atr()
        self.set_others()
        # 指标计算完毕，不再需要用ta库，将index还原为自然数方便.loc索引
        self.df.reset_index(inplace=True)
        # 导出数据
        self.df.to_csv(os.path.join(RESULT_PATH, 'df_all.csv'), index=True)
        # 内置参数
        self.transaction_list = []
        self.transaction_dict = {}
        self.position = 0
        self.contract = ''

    def set_donchian(self):
        donchian_n1 = ta.donchian(high=self.df['high'], low=self.df['low'],
                                  lower_length=self.n_1, upper_length=self.n_1, offset=1)
        donchian_n2 = ta.donchian(high=self.df['high'], low=self.df['low'],
                                  lower_length=self.n_2, upper_length=self.n_2, offset=1)
        donchian_n3 = ta.donchian(high=self.df['high'], low=self.df['low'],
                                  lower_length=self.n_3, upper_length=self.n_3, offset=1)
        df_all = pd.concat([self.df, donchian_n1, donchian_n2, donchian_n3], axis=1)
        return df_all

    def set_atr(self):
        atr = round(ta.atr(self.df['high'], self.df['low'], self.df['close'], length=self.atr_parameter), 4)
        df_all = pd.concat([self.df, atr], axis=1)
        return df_all

    def set_others(self):
        self.df['buy_mark'] = np.nan
        self.df['buy_close_mark'] = np.nan
        self.df['sell_mark'] = np.nan
        self.df['sell_close_mark'] = np.nan

    def donchian_up(self, n_donchian, i):
        # 唐奇安通道上破
        return True if self.df.loc[i, 'high'] > self.df.loc[i, f'DCU_{n_donchian}_{n_donchian}'] and \
                       self.df.loc[i-1, 'high'] < self.df.loc[i-1, f'DCU_{n_donchian}_{n_donchian}'] else False

    def donchian_down(self, n_donchian, i):
        # 唐奇安通道下破
        return True if self.df.loc[i, 'low'] < self.df.loc[i, f'DCL_{n_donchian}_{n_donchian}'] and \
                       self.df.loc[i-1, 'low'] > self.df.loc[i-1, f'DCL_{n_donchian}_{n_donchian}'] else False

    def LastTradeProfitable(self):
        # 上一个交易为盈利交易
        return True if len(self.transaction_list) > 0 and self.transaction_list[-1].net_profit > 0 else False

    def AddLongPosition(self, i, N):
        # 多头加仓
        # 规则：在最大加仓次数内，当收盘价(or 最高价)超过上次进场位置+0.5N，则加仓
        # 时间为下一日开盘，价格为开盘价，加仓仓位为计算得到的volume
        # 对于期货交易，由于存在主力合约变动的问题，盲目取下一日很容易出现合约不连贯的问题
        # 因此修正规则：时间为当日，价格为当日结算价，加仓仓位不变
        if len(self.transaction_list) < 1:
            return False

        return True if (self.transaction_list[-1].transaction_status == 'open' and
                        self.transaction_list[-1].add_pos['times'] < self.max_add_times and
                        self.df.loc[i, 'high'] - self.transaction_list[-1].add_pos['price'][-1] >= 0.5 * N)\
            else False

    def AddShortPosition(self, i, N):
        # 空头加仓
        # 规则：在最大加仓次数内，当收盘价(or 最低价)低于上次进场位置-0.5N，则加仓
        # 时间为下一日开盘，价格为开盘价，加仓仓位为计算得到的volume
        # 对于期货交易，由于存在主力合约变动的问题，盲目取下一日很容易出现合约不连贯的问题
        # 因此修正规则：时间为当日，价格为当日结算价，加仓仓位不变
        if len(self.transaction_list) < 1:
            return False

        return True if (self.transaction_list[-1].transaction_status == 'open' and
                        self.transaction_list[-1].add_pos['times'] < self.max_add_times and
                        self.transaction_list[-1].add_pos['price'][-1] - self.df.loc[i, 'low'] >= 0.5 * N)\
            else False

    def update_pos(self, i):
        self.transaction_list[-1].update(high_price=self.df.loc[i, 'high'],
                                         low_price=self.df.loc[i, 'low'],
                                         settle_price=self.df.loc[i, 'settle'])

    def close_pos(self, i):
        # TODO：这里直接用了昨日结算价，很不精确，应当从大数据库中读取该合约当日结算价（非主力合约的数据无法从处理后的数据表中查询）
        self.transaction_list[-1].clean(date=self.df.loc[i, 'date'].strftime("%Y-%m-%d"),
                                        price=self.df.loc[i-1, 'settle'])
        self.position = 0
        self.account += self.transaction_list[-1].net_profit

    def back_test(self):
        # 回测函数
        for i in self.df.index:
            if self.account < 0:
                print('{0}，账户余额为{1}，停止交易'.format(self.df.loc[i, 'date'].strftime("%Y-%m-%d"), self.account))
                break
            N = self.df[f'ATRr_{self.atr_parameter}'][i]
            # volume为开仓/加仓手数 10为一手对应的报价单位（1手=10吨，报价点数为1元/吨）
            volume = max(self.min_volume, np.round(self.account*self.capital_control_ratio/N/10, 2))

            if self.position == 0:
                # 开仓
                if (self.donchian_up(self.n_1, i) and not self.LastTradeProfitable()) or self.donchian_up(self.n_2, i):
                    # 开多仓
                    self.position = 1
                    self.transaction_list.append(
                        FutureTransaction(uid=len(self.transaction_list)+1,
                                          date=self.df.loc[i, 'date'].strftime("%Y-%m-%d"),
                                          name=os.path.basename(__file__),
                                          variety='rb',  # TODO:由参数自动填充品种
                                          contract=self.df.loc[i, 'contract'],
                                          position=self.position,
                                          price=self.df.loc[i, 'settle'],
                                          volume=volume,
                                          charge=0)  # TODO:保证金占用系统
                    )
                    self.df.loc[i, 'buy_mark'] = self.df.loc[i, 'low']*0.99

                if (self.donchian_down(self.n_1, i) and not self.LastTradeProfitable()) or self.donchian_down(self.n_2, i):
                    self.position = -1
                    self.transaction_list.append(
                        FutureTransaction(uid=len(self.transaction_list)+1,
                                          date=self.df.loc[i, 'date'].strftime("%Y-%m-%d"),
                                          name=os.path.basename(__file__),
                                          variety='rb',  # TODO:由参数自动填充品种
                                          contract=self.df.loc[i, 'contract'],
                                          position=self.position,
                                          price=self.df.loc[i, 'settle'],
                                          volume=volume,
                                          charge=0)  # TODO:保证金占用系统
                    )
                    self.df.loc[i, 'sell_mark'] = self.df.loc[i, 'high']*1.01
            elif self.position == 1:
                # 更新仓位
                # 由于使用elif，开仓时设定的position=1不会导致这里重复执行
                self.update_pos(i)
                # 判断主力合约是否发生变动，若变动则移仓(清旧开新)
                if self.df.loc[i, 'contract'] != self.transaction_list[-1].contract:
                    # 清旧
                    self.close_pos(i)
                    # 开新(注意手数不变)
                    self.position = 1
                    self.transaction_list.append(
                        FutureTransaction(uid=len(self.transaction_list)+1,
                                          date=self.df.loc[i, 'date'].strftime("%Y-%m-%d"),
                                          name=os.path.basename(__file__),
                                          variety='rb',  # TODO:由参数自动填充品种
                                          contract=self.df.loc[i, 'contract'],
                                          position=self.position,
                                          price=self.df.loc[i, 'settle'],
                                          volume=self.transaction_list[-1].total_volume,
                                          charge=0)  # TODO:保证金占用系统
                    )
                # 判断是否要加多仓
                if self.AddLongPosition(i, N):
                    self.transaction_list[-1].increase(date=self.df.loc[i, 'date'].strftime("%Y-%m-%d"),
                                                       position=self.position,
                                                       price=self.df.loc[i, 'settle'],
                                                       volume=volume,
                                                       note='TURTLE ADD LONG')
                # 判断是否要清多仓
                if self.donchian_down(self.n_3, i):
                    self.close_pos(i)
                    self.df.loc[i, 'buy_close_mark'] = self.df.loc[i, 'high']*1.01

            elif self.position == -1:
                # 更新仓位
                # 由于使用elif，开仓时设定的position=-1不会导致这里重复执行
                self.update_pos(i)
                # 判断主力合约是否发生变动，若变动则移仓(清旧开新)
                if self.df.loc[i, 'contract'] != self.transaction_list[-1].contract:
                    # 清旧
                    self.close_pos(i)
                    # 开新(注意手数不变)
                    self.position = -1
                    self.transaction_list.append(
                        FutureTransaction(uid=len(self.transaction_list)+1,
                                          date=self.df.loc[i, 'date'].strftime("%Y-%m-%d"),
                                          name=os.path.basename(__file__),
                                          variety='rb',  # TODO:由参数自动填充品种
                                          contract=self.df.loc[i, 'contract'],
                                          position=self.position,
                                          price=self.df.loc[i, 'settle'],
                                          volume=self.transaction_list[-1].total_volume,
                                          charge=0)  # TODO:保证金占用系统
                    )
                # 判断是否要加空仓
                if self.AddShortPosition(i, N):
                    self.transaction_list[-1].increase(date=self.df.loc[i, 'date'].strftime("%Y-%m-%d"),
                                                       position=self.position,
                                                       price=self.df.loc[i, 'settle'],
                                                       volume=volume,
                                                       note='TURTLE ADD SHORT')
                # 判断是否要清空仓
                if self.donchian_up(self.n_3, i):
                    self.close_pos(i)
                    self.df.loc[i, 'sell_close_mark'] = self.df.loc[i, 'low']*0.99
                pass
            else:
                print('ERROR position')

            if i == len(self.df.index) - 1:
                if self.transaction_list[-1].transaction_status == 'open':
                    self.update_pos(i)
                    if self.position == 1:
                        self.df.loc[i, 'buy_close_mark'] = self.df.loc[i, 'high']*1.01
                    elif self.position == -1:
                        self.df.loc[i, 'sell_close_mark'] = self.df.loc[i, 'low']*0.99
                    else:
                        pass
                    self.close_pos(i)

    def turtle_plot(self):
        add_plot = [mpf.make_addplot(self.df[f'DCU_{self.n_1}_{self.n_1}'], markersize=10, color='g'),
                    mpf.make_addplot(self.df[f'DCL_{self.n_1}_{self.n_1}'], markersize=10, color='g'),
                    # 显示n1唐奇安通道
                    mpf.make_addplot(self.df[f'DCU_{self.n_2}_{self.n_2}'], markersize=10, color='b'),
                    mpf.make_addplot(self.df[f'DCL_{self.n_2}_{self.n_2}'], markersize=10, color='b'),
                    # 显示n2唐奇安通道
                    mpf.make_addplot(self.df[f'DCU_{self.n_3}_{self.n_3}'], markersize=10, color='r'),
                    mpf.make_addplot(self.df[f'DCL_{self.n_3}_{self.n_3}'], markersize=10, color='r'),
                    # 显示n3唐奇安通道
                    mpf.make_addplot(self.df['buy_mark'], scatter=True, markersize=50, marker='^', color='g'),
                    mpf.make_addplot(self.df['sell_mark'], scatter=True, markersize=50, marker='v', color='r'),
                    mpf.make_addplot(self.df['buy_close_mark'], scatter=True, markersize=50, marker='*', color='g'),
                    mpf.make_addplot(self.df['sell_close_mark'], scatter=True, markersize=50, marker='*', color='r'),
                    # 显示进场出场信号
                    ]
        title = '{0}\n{1}-{2}'.format(os.path.basename(__file__), self.start_date, self.end_date)
        style = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.family': 'SimHei'})
        graph = mpf.plot(self.df,
                         addplot=add_plot,
                         title=title,
                         ylabel='Price',
                         type='candle',
                         style=style,
                         show_nontrading=False)


df_rb = pd.read_csv(os.path.join(DATA_PATH, 'data_clean_rb.csv'), parse_dates=['date'], index_col='date')
test = FutureStrategy(20, 55, 6, 20, 0.01, '2014-1-2', '2021-7-30', data=df_rb, account=1000000, min_volume=1, max_add_times=3)
test.back_test()
test.turtle_plot()
