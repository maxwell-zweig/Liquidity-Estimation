from pathlib import Path
from tardis_client import TardisClient, Channel
import asyncio
import datetime as dt
import pandas as pd 
import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt 
import argparse
import time
import numba as nb
from scipy.stats import linregress as regress
from bisect import bisect_left

TARDIS_KEY = "TD.Q5XS-m6SRp-ogLN7.Nil-HWnnsa7ScQM.gEv0PZ7PKNOg1Vc.0pjQaUs47u3Ce7b.Yl75DeyhPNf4iGr.1rud"

class Intensity():
    def intensity_measurements(self, EXCHANGE, BACKTEST_BEGIN, BACKTEST_END, SESSION_LENGTH, MARKET, lookback, timetstamp, order_survival_length, order_sampling_freq):
        if not self.check_existance(MARKET, BACKTEST_BEGIN, BACKTEST_END, SESSION_LENGTH, EXCHANGE, lookback):
            print(f'{MARKET}, {EXCHANGE} data not found on system. Downloading...')
            self.fetch_data(market_name=MARKET, backtest_begin=BACKTEST_BEGIN, backtest_end=BACKTEST_END, period=SESSION_LENGTH, exchange=EXCHANGE, lookback=lookback, order_surival_length=order_survival_length, order_sampling_freq=order_sampling_freq)
        return self.get_params(market_name=MARKET, backtest_begin=BACKTEST_BEGIN, backtest_end=BACKTEST_END, lookback=lookback, period=SESSION_LENGTH, exchange=EXCHANGE, timestamp=timetstamp)


    def check_existance(self, market_name, backtest_begin, backtest_end, period, exchange, lookback):
        return Path(f"../data_storage/flow_info/{backtest_begin.replace(':', '-')}-{backtest_end.replace(':', '-')}-{exchange}-{period}-{lookback}-{market_name.replace('/', '-')}.csv").is_file()
        

    def fetch_data(self, market_name, backtest_begin, backtest_end, lookback, period, exchange, order_survival_length, order_sampling_freq):

        '''
        Currently only works if lookback == period. Can average a different number of period together to produce artifical lookbacks later.
        '''

      ##  print("reading...")
      ##  if self.check_existance(market_name, backtest_begin, backtest_end, period, exchange, lookback):
      ##      df = pd.read_csv(Path(f"../data_storage/flow_info/{backtest_begin.replace(':', '-')}-{backtest_end.replace(':', '-')}-{exchange}-{period}-{lookback}-{market_name.replace('/', '-')}.csv"))
      ##      As = list(df['As'])
      ##      Ks = list(df['Ks'])
      ##      timestamps =  list(df['Timestamps'])
      ##      avg_spreads = list(df['Spreads'])
      ##      avg_bids_d = list(df['Bid Density'])
      ##      avg_asks_d = list(df['Ask Density'])
       # else:
        As_sell = []
        Ks_sell = []

        As_buys = []
        Ks_buys = []
            
        timestamps = []
        avg_spreads = []
        avg_bids_d = []
        avg_asks_d = []
        avg_trades = []

        as_bids = []
        bs_bids = []
        as_asks = []
        bs_asks = []

        def intensity_func(x, A, K):
            return A * np.exp(-1 * K *x)

        async def replay():
            tardis_client = TardisClient(api_key = TARDIS_KEY)
            backtest_begin_timestamp = dt.datetime.timestamp(dt.datetime.strptime(backtest_begin,"%Y-%m-%dT%H:%M:%S"))        
            backtest_begin_timestamp -= period * lookback

            lookbacked = dt.datetime.fromtimestamp(backtest_begin_timestamp)

            beginning_of_day = dt.datetime(lookbacked.year, lookbacked.month, lookbacked.day)
            beginning_of_day_minus = beginning_of_day - dt.timedelta(seconds=5)

            lookbacked_formatted = dt.datetime.strftime(beginning_of_day_minus, "%Y-%m-%dT%H:%M:%S")

            lookbacked_formatted = dt.datetime.strftime(lookbacked, "%Y-%m-%dT%H:%M:%S")
            messages = tardis_client.replay(
                exchange=exchange,
                from_date=lookbacked_formatted,
                to_date=backtest_end,
                filters=[
                Channel(name="ticker", symbols=[market_name]),
                Channel(name="aggTrade", symbols=[market_name]),
                Channel(name="depth", symbols=[market_name]),
                Channel(name="depthSnapshot", symbols=[market_name])
                ],
            )
            print(dt.datetime.fromtimestamp(backtest_begin_timestamp))
            last_sampled = backtest_begin_timestamp + period
            print(dt.datetime.fromtimestamp(last_sampled))

            # Aggregating btcusd by tick - each x portion is 1 / 100th of a cent, up to 100 dollars
            sell_order_deltas = np.zeros(10000)
            buy_order_deltas = np.zeros(10000)

            price = 0
            price_initialized = False 
            last_sample_creation = backtest_begin_timestamp
            orders = []
            num_samples = 0 
            spreads = []

            standing_spreads = np.empty(0)
            standing_bids = np.empty(0)
            standing_asks = np.empty(0)
            standing_orders = np.empty(0)
            standing_spread_aggs = np.empty(0)
            standing_orderbooks_aggs = np.empty(0)
            num_standing_samples = 0

            standing_lob_bid = np.empty((0,5))
            standing_lob_ask = np.empty((0,5))


            sell_order_deltas_alternate = np.zeros(10000)
            buy_order_deltas_alternate = np.zeros(10000)

           # standing_sell_order_deltas = np.empty((0,0))
           # standing_buy_order_deltas = np.empty((0,0))
            


            # print(standing_spreads.shape)
            # print(standing_orderbooks.shape)


            orderbook_init = False
            orderbook_bids = {}
            orderbook_asks = {}

            bisect_bids_prices = []
            bisect_asks_prices = []

            bisect_bids_qts = []
            bisect_asks_qts = []


            bid_densities = []
            ask_densities = []

            bid_lobs = []
            ask_lobs = []
            
            trades = []

            last_timestamp_bruh = backtest_begin_timestamp - 10

            async for local_timestamp, message in messages:
                timestamp = dt.datetime.timestamp(local_timestamp)
                # assert timestamp >= last_timestamp_bruh
                # last_timestamp_bruh = timestamp
                if 'data' not in message.keys():
                    pass 
                else:
                    if message["stream"] == f'{market_name}@ticker':
                        price = (float(message['data']['b']) + float(message['data']['a'])) / 2
                       # spreads.append(float(message['data']['a']) - float(message['data']['b']))
                        standing_spreads += float(message['data']['a']) - float(message['data']['b'])
                        standing_spread_aggs += 1
                        '''
                        for order in orders:
                            order["spread_agg"] += float(message['data']['a']) - float(message['data']['b'])
                            order["spread_accs"] += 1

                        '''

                    elif message["stream"] == f'{market_name}@aggTrade':
                        exec_price = float(message['data']['p'])
                        exec_quantity = float(message['data']['q']) 
                        trades.append(exec_quantity)
                        is_sell = message['data']['m']
                        if is_sell:
                            
                            processed = standing_orders - exec_price
                            deltas = processed[processed >= 0]
                            d_indices = np.minimum(np.full(deltas.shape[0], sell_order_deltas_alternate.shape[0] - 1, dtype=int), np.floor(deltas * 100).astype(int)).astype(int)
                            np.add.at(sell_order_deltas_alternate, d_indices, 1)


                         #   assert np.array_equal(d_indices, (np.floor(deltas * 100).astype(int)))

                          #  sell_order_deltas_alternate[d_indices] += exec_quantity

                           # for order in orders:
                           #     delta = abs(exec_price - order["price"])
                           #     sell_order_deltas[:int(delta * 100) + 1] += exec_quantity
                            
                        else:

                            processed = exec_price - standing_orders
                            deltas = processed[processed >= 0]
                            d_indices = np.minimum(np.full(deltas.shape[0], buy_order_deltas_alternate.shape[0] - 1, dtype=int), np.floor(deltas * 100).astype(int)).astype(int)
                            np.add.at(buy_order_deltas_alternate, d_indices, 1)
                          #  buy_order_deltas_alternate[d_indices] += exec_quantity

                            #for order in orders:
                            #    delta = abs(exec_price - order["price"])
                            #    buy_order_deltas[:int(delta * 100) + 1] += exec_quantity

                    elif message["stream"] == f'{market_name}@depth@100ms':
                        if orderbook_init == False:
                            pass 
                        else:
                            bid_updates = message['data']['b']
                            ask_updates = message['data']['a']
                            
                            for bid in bid_updates:
                                bid_price = float(bid[0])
                                bid_qt = float(bid[1])

                                inserted_index = -1
                                action_index = bisect_left(bisect_bids_prices, bid_price)
                                if bid_price not in orderbook_bids:
                                    if action_index < len(bisect_bids_prices):
                                        bisect_bids_prices.insert(action_index, bid_price)
                                        bisect_bids_qts.insert(action_index, bid_qt)
                                        inserted_index = action_index
                                    else:
                                        bisect_bids_prices.append(bid_price)
                                        bisect_bids_qts.append(bid_qt)
                                        inserted_index = len(bisect_bids_prices) - 1
                                else:
                                    bisect_bids_qts[action_index] = bid_qt
                                    inserted_index = action_index


                                orderbook_bids[bid_price] = bid_qt

                                if bid_qt == 0:
                                    del orderbook_bids[bid_price]
                                    bisect_bids_prices.pop(inserted_index)
                                    bisect_bids_qts.pop(inserted_index)

                            for ask in ask_updates:
                                ask_price = float(ask[0])
                                ask_qt = float(ask[1])

                                inserted_index = -1
                                action_index = bisect_left(bisect_asks_prices, ask_price)
                                if ask_price not in orderbook_asks:
                                    if action_index < len(bisect_asks_prices):
                                        bisect_asks_prices.insert(action_index, ask_price)
                                        bisect_asks_qts.insert(action_index, ask_qt)
                                        inserted_index = action_index

                                    else:
                                        bisect_asks_prices.append(ask_price)
                                        bisect_asks_qts.append(ask_qt)
                                        inserted_index = len(bisect_asks_prices) - 1

                                else:
                                    bisect_asks_qts[action_index] = ask_qt
                                    inserted_index = action_index


                                orderbook_asks[ask_price] = ask_qt

                                if ask_qt == 0:
                                    del orderbook_asks[ask_price]
                                    bisect_asks_prices.pop(inserted_index)
                                    bisect_asks_qts.pop(inserted_index)

                            '''
                            
                            bid_keys = list(orderbook_bids.keys())
                            ask_keys = list(orderbook_asks.keys())

                            bid_keys_sorted = sorted(bid_keys, key = float)
                            ask_keys_sorted = sorted(ask_keys, key = float)

                            t25_bids_keys = bid_keys_sorted[-25:]
                            t25_asks_keys = ask_keys_sorted[:25]


                            cur_ask_levels = [orderbook_asks[key] for key in t25_asks_keys]
                            cur_bid_levels = [orderbook_bids[key] for key in t25_bids_keys]


                            assert cur_bid_levels == bisect_bids_qts[-25:]
                            assert cur_ask_levels == bisect_asks_qts[:25]
                            assert t25_bids_keys == bisect_bids_prices[-25:]
                            assert t25_asks_keys == bisect_asks_prices[:25]

                            print(cur_bid_levels)
                            print(bisect_bids_qts[-25:])

                            print(cur_ask_levels)
                            print(bisect_asks_qts[:25])
                            
                            print('bids')
                            print(t25_bids_keys)
                            print(bisect_bids_prices[-25:])

                            print('asks')
                            print(t25_asks_keys)
                            print(bisect_asks_prices[:25])

                     #   bid_densities.append(sum(bisect_bids_qts[-5:]) / 5)
                     #   ask_densities.append(sum(bisect_asks_qts[:5]) / 5)
                            for order in orders:
                                for idx in range(1, 6):
                                    order["bid_agg"][idx - 1] += bisect_bids_qts[-idx]
                                for idx in range(5):
                                    order["ask_agg"][idx] += bisect_asks_qts[idx]
                                order["orderbook_accs"] += 1
                            '''
                            standing_orderbooks_aggs += 1

                            standing_asks += sum(bisect_asks_qts[:5]) / 5
                            standing_bids += sum(bisect_bids_qts[-5:]) / 5

                            standing_lob_bid += np.array(bisect_bids_qts[-5:])
                            standing_lob_ask += np.array(bisect_asks_qts[:5])


                            
                    elif message['stream'] == f'{market_name}@depthSnapshot':
                        orderbook_init = True
                        bids = message['data']['bids']
                        asks = message['data']['asks']

                        bids = [[stri for stri in pair] for pair in bids]
                        asks = [[stri for stri in pair] for pair in asks]

                        bids = {float(pair[0]) : float(pair[1]) for pair in bids}
                        asks = {float(pair[0]) : float(pair[1]) for pair in asks}


                        orderbook_bids = bids
                        orderbook_asks = asks

                        bid_keys = orderbook_bids.keys()
                        ask_keys = orderbook_asks.keys()

                        bid_keys_sorted = sorted(bid_keys, key = float)
                        ask_keys_sorted = sorted(ask_keys, key = float)

                        bisect_asks_prices, bisect_asks_qts, bisect_bids_prices, bisect_bids_qts = [], [], [], []

                        for bid_key, ask_key in zip(bid_keys_sorted, ask_keys_sorted):
                            bisect_bids_prices.append(bid_key)
                            bisect_asks_prices.append(ask_key)
                            bisect_bids_qts.append(orderbook_bids[bid_key])
                            bisect_asks_qts.append(orderbook_asks[ask_key])

                        orderbook_init = True 
                        lastupdate = int(message['data']['lastUpdateId'])
                        print(f'received, {dt.datetime.fromtimestamp(timestamp)}')
                        

                    if (timestamp - last_sampled) // period > 0:

                        timestamps.append(timestamp)
                        print(f'Processed: {dt.datetime.fromtimestamp(timestamp)}')

                        X = np.linspace(0, 100, 10000)

                        ''' A is in orders per order survival length'''
                        @nb.jit(nopython=True)
                        def fast():
                            buy_order_deltas_alternate_calc = np.array([sum(buy_order_deltas_alternate[idx : ])  for idx in range(len(buy_order_deltas_alternate))])
                            sell_order_deltas_alternate_calc = np.array([sum(sell_order_deltas_alternate[idx : ])  for idx in range(len(sell_order_deltas_alternate))])
                            return buy_order_deltas_alternate_calc, sell_order_deltas_alternate_calc

                        buy_order_deltas_alternate_calc, sell_order_deltas_alternate_calc = fast()
                        Y_sell_alt = sell_order_deltas_alternate_calc / (num_standing_samples)
                        Y_buy_alt = buy_order_deltas_alternate_calc / (num_standing_samples)

                       # Y_sell = sell_order_deltas / (num_samples)
                       # Y_buy = buy_order_deltas / (num_samples)

                      #  assert np.array_equal(Y_buy, Y_buy_alt) and np.array_equal(Y_sell, Y_sell_alt)


                       # plt.plot(X, Y_sell)
                      #  plt.plot(X, Y_sell_alt)

                       # plt.plot(X, Y_buy)
                      #  plt.plot(X, Y_buy_alt)

                        

                        print('fitting')
                        res_sell = curve_fit(f = intensity_func, ydata=Y_sell_alt, xdata=X)
                        res_buy = curve_fit(f = intensity_func, ydata=Y_buy_alt, xdata=X)


                       # plt.plot(X, intensity_func(X, res_sell[0][0], res_sell[0][1]))
                       # plt.plot(X, intensity_func(X, res_buy[0][0], res_buy[0][1]))

                       # plt.show()
                        orderbook_func = lambda x, a, b : a * x + b 
                        


                        bid_data = np.array(bid_lobs).sum(axis=0) / num_standing_samples
                        ask_data = np.array(ask_lobs).sum(axis=0) / num_standing_samples


                        res_orderbook_shape_ask = regress(x=np.arange(5, dtype=int), y = ask_data)
                        res_orderbook_shape_bid = regress(x=np.arange(5, dtype=int), y = bid_data)


                        as_bids_stuff = res_orderbook_shape_bid[0]
                        bs_bids_stuff = res_orderbook_shape_bid[1]

                        as_asks_stuff = res_orderbook_shape_ask[0]
                        bs_asks_stuff = res_orderbook_shape_ask[1]

                        as_bids.append(as_bids_stuff)
                        bs_bids.append(bs_bids_stuff)

                        as_asks.append(as_asks_stuff)
                        bs_asks.append(bs_asks_stuff)


                        As_sell.append(res_sell[0][0])
                        Ks_sell.append(res_sell[0][1])

                        Ks_buys.append(res_buy[0][1])
                        As_buys.append(res_buy[0][0])

                        sell_order_deltas = np.zeros(10000)
                        buy_order_deltas = np.zeros(10000)
                        avg_spreads.append(sum(spreads) / len(spreads))
                        avg_trades.append(sum(trades) / len(trades))
                        



                        if orderbook_init and len(bid_densities) != 0 and len(ask_densities) != 0:
                            avg_bids_d.append(sum(bid_densities) / len(bid_densities))
                            avg_asks_d.append(sum(ask_densities) / len(ask_densities))
                        else:
                            avg_bids_d.append(-1)
                            avg_asks_d.append(-1)

                        print(As_sell[-1], Ks_sell[-1], As_buys[-1], Ks_buys[-1], avg_spreads[-1], avg_bids_d[-1], avg_asks_d[-1], avg_trades[-1], as_bids[-1], bs_bids[-1], as_asks[-1], bs_asks[-1])
                        
                        orders = []
                        num_samples = 0 
                        spreads = []
                        bid_densities =[]
                        ask_densities = []
                        trades =[]

                        standing_spreads = np.empty(0)
                        standing_bids = np.empty(0)
                        standing_asks = np.empty(0)
                        standing_orders = np.empty(0)
                        standing_spread_aggs = np.empty(0)
                        standing_orderbooks_aggs = np.empty(0)
                        num_standing_samples = 0
                        bid_lobs = []
                        ask_lobs = []

                        sell_order_deltas_alternate = np.zeros(10000)
                        buy_order_deltas_alternate = np.zeros(10000)

                        standing_lob_ask = np.empty((0,5))
                        standing_lob_bid = np.empty((0,5))


                        data = pd.DataFrame(np.array([As_sell, Ks_sell, As_buys, Ks_buys, avg_spreads, avg_bids_d, avg_asks_d, avg_trades, as_bids, as_asks, bs_bids, bs_asks]).T, index=timestamps, columns=["As_sell", "Ks_sell", "As_buys", "Ks_buys", "Spreads", "Bid Density", "Ask Density", "Sizes", "as_bids", "as_asks", "bs_bids", "bs_asks"])
                        data.index.name = "Timestamps"
                        data.to_csv(Path(f"../../data_storage/flow_info/{backtest_begin.replace(':', '-')}-{backtest_end.replace(':', '-')}-{exchange}-{period}-{lookback}-{market_name.replace('/', '-')}_final_2.csv"))
                        last_sampled = (timestamp // period) * period

                    if (timestamp - last_sample_creation) // order_sampling_freq > 0:
                        '''
                        
                        if len(orders) > order_survival_length / order_sampling_freq:
                            popped = orders.pop(0)
                            spreads.append(popped["spread_agg"] / popped["spread_accs"])
                            bid_densities.append(np.mean(popped["bid_agg"] / popped["orderbook_accs"]))
                            ask_densities.append(np.mean(popped["ask_agg"] / popped["orderbook_accs"]))


                        if not order_survival_length / order_sampling_freq > ((1 + timestamp // period) * period - timestamp) / order_sampling_freq:
                            orders.append({"price" : price, "spread_agg" : 0, "spread_accs" : 0, "bid_agg" : np.zeros(5), "ask_agg" : np.zeros(5), "orderbook_accs" : 0})
                            num_samples += 1
                        

                        if order_survival_length / order_sampling_freq > 1 + ((1 + timestamp // period) * period - timestamp) / order_sampling_freq:
                            popped = orders.pop(0)
                            spreads.append(popped["spread_agg"] / popped["spread_accs"])
                            bid_densities.append(np.mean(popped["bid_agg"] / popped["orderbook_accs"]))
                            ask_densities.append(np.mean(popped["ask_agg"] / popped["orderbook_accs"]))
                        '''
                        if standing_orders.shape[0] > order_survival_length / order_sampling_freq:
                            
                            popped = orders.pop(0)
                            
                            standing_orders = np.delete(standing_orders, 0)
                            spreads.append(standing_spreads[0] / standing_spread_aggs[0])
                            bid_densities.append(standing_bids[0] / standing_orderbooks_aggs[0])
                            ask_densities.append(standing_asks[0] / standing_orderbooks_aggs[0])
                            bid_lobs.append(standing_lob_bid[0] / standing_orderbooks_aggs[0])
                            ask_lobs.append(standing_lob_ask[0] / standing_orderbooks_aggs[0])
                            standing_lob_ask = np.delete(standing_lob_ask, 0, axis=0)
                            standing_lob_bid = np.delete(standing_lob_bid, 0, axis=0)

                            standing_spreads = np.delete(standing_spreads, 0)
                            standing_spread_aggs = np.delete(standing_spread_aggs, 0)
                            standing_bids = np.delete(standing_bids, 0)
                            standing_asks = np.delete(standing_asks, 0)
                            standing_orderbooks_aggs = np.delete(standing_orderbooks_aggs, 0)




                        if not order_survival_length / order_sampling_freq > ((1 + timestamp // period) * period - timestamp) / order_sampling_freq:
                            standing_orders = np.append(standing_orders, price)
                            num_standing_samples += 1
                            standing_spreads = np.append(standing_spreads, 0)
                            standing_spread_aggs = np.append(standing_spread_aggs, 0)

                            standing_bids = np.append(standing_bids, 0)
                            standing_asks = np.append(standing_asks, 0)
                            standing_orderbooks_aggs = np.append(standing_orderbooks_aggs, 0)
                            standing_lob_bid = np.append(standing_lob_bid, np.zeros((1, 5)), axis=0)
                            standing_lob_ask = np.append(standing_lob_ask, np.zeros((1, 5)), axis=0)




                            orders.append({"price" : price, "spread_agg" : 0, "spread_accs" : 0, "bid_agg" : np.zeros(5), "ask_agg" : np.zeros(5), "orderbook_accs" : 0})
                            num_samples += 1




                        if order_survival_length / order_sampling_freq > 1 + ((1 + timestamp // period) * period - timestamp) / order_sampling_freq:
                            standing_orders = np.delete(standing_orders, 0)
                            spreads.append(standing_spreads[0] / standing_spread_aggs[0])
                            bid_densities.append(standing_bids[0] / standing_orderbooks_aggs[0])
                            ask_densities.append(standing_asks[0] / standing_orderbooks_aggs[0])


                            bid_lobs.append(standing_lob_bid[0] / standing_orderbooks_aggs[0])
                            ask_lobs.append(standing_lob_ask[0] / standing_orderbooks_aggs[0])
                            standing_lob_ask = np.delete(standing_lob_ask, 0, axis=0)
                            standing_lob_bid = np.delete(standing_lob_bid, 0, axis=0)

                            standing_spreads = np.delete(standing_spreads, 0)
                            standing_spread_aggs = np.delete(standing_spread_aggs, 0)
                            
                            standing_bids = np.delete(standing_bids, 0)
                            standing_asks = np.delete(standing_asks, 0)
                            standing_orderbooks_aggs = np.delete(standing_orderbooks_aggs, 0)


                            popped = orders.pop(0)

                        #print(len(orders))
                        #print(dt.datetime.fromtimestamp(timestamp))
                        last_sample_creation = (timestamp // order_sampling_freq) * order_sampling_freq
                      #  print(len(orders))
                      #  print(timestamp)
        asyncio.run(replay())
       

    def get_params(self, market_name, backtest_begin, backtest_end, lookback, period, exchange, timestamp):
        df = pd.read_csv(Path(f"../data_storage/flow_info/{backtest_begin.replace(':', '-')}-{backtest_end.replace(':', '-')}-{exchange}-{period}-{lookback}-{market_name.replace('/', '-')}.csv"))
        df = df.set_index("Timestamps")
        closest_idx = df.iloc[df.index.get_indexer([timestamp], method='nearest')]
        return closest_idx.iloc[0][0], closest_idx.iloc[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange")
    parser.add_argument("--backtest_begin")
    parser.add_argument("--backtest_end")
    parser.add_argument("--session_length", help="Session Length in seconds. Imperative that it is an integer number of hours.")
    parser.add_argument("--market")
    parser.add_argument("--intensity_lookback")
    parser.add_argument("--survival_length")
    parser.add_argument("--sampling_freq")
    args = parser.parse_args()

    EXCHANGE = args.exchange
    BACKTEST_BEGIN = args.backtest_begin
    BACKTEST_END = args.backtest_end
    SESSION_LENGTH = int(args.session_length)
    MARKET = args.market
    intensity_lookback = int(args.intensity_lookback)
    survival_length = int(args.survival_length)
    sampling_Freq = int(args.sampling_freq)


    intensity = Intensity()
    intensity.fetch_data(MARKET, BACKTEST_BEGIN, BACKTEST_END, intensity_lookback, SESSION_LENGTH, EXCHANGE, survival_length, sampling_Freq)
