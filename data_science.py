import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if torch.cuda.is_available():
    device=torch.device('cuda:0')
    print('Running on GPU')
else:
    device=torch.device('cpu')
    print('Running on CPU')

OUTLIER_THRESHOLD = 0.01

df = pd.read_csv("2022-09-05T00-00-00-2022-12-31T23-59-59-binance-300-0-btcusdt_final_2.csv", index_col="Timestamps")

df = df.iloc[4:]


As_sell = np.array(df.loc[:, "As_sell"])
As_buy = np.array(df.loc[:, "As_buys"])
ks_sell = np.array(df.loc[:, "Ks_sell"])
ks_buy = np.array(df.loc[:, "Ks_buys"])
bid_density = np.array(df.loc[:, "Bid Density"])
ask_density = np.array(df.loc[:, "Ask Density"])
spread = np.array(df.loc[:, "Spreads"])
bid_a = np.array(df.loc[:, "as_bids"])
bid_b = np.array(df.loc[:, "bs_bids"])
ask_a = np.array(df.loc[:, "as_asks"])
ask_b = np.array(df.loc[:, "bs_asks"])

# Removing outliers from collected data

As_sell_upper = np.percentile(As_sell, 99)
As_sell_lower = np.percentile(As_sell, 0.01)

As_buy_upper = np.percentile(As_buy, 99)
As_buy_lower = np.percentile(As_buy, 0.01)


ks_sell_upper = np.percentile(ks_sell, 99)
ks_sell_lower = np.percentile(ks_sell, 0.01)

ks_buy_upper = np.percentile(ks_buy, 99)
ks_buy_lower = np.percentile(ks_buy, 0.01)

bid_density_upper = np.percentile(bid_density, 99)
bid_density_lower = np.percentile(bid_density, 0.01)

ask_density_upper = np.percentile(ask_density, 99)
ask_density_lower = np.percentile(ask_density, 0.01)

spread_upper = np.percentile(spread, 99)
spread_lower = np.percentile(spread, 0.01)

bid_a_upper = np.percentile(bid_a, 99)
bid_a_lower = np.percentile(bid_a, 0.01)

bid_b_upper = np.percentile(bid_b, 99)
bid_b_lower = np.percentile(bid_b, 0.01)

ask_a_upper = np.percentile(ask_a, 99)
ask_a_lower = np.percentile(ask_a, 0.01)

ask_b_upper = np.percentile(ask_b, 99)
ask_b_lower = np.percentile(ask_b, 0.01)


ks_sell_mask = np.logical_and(ks_sell > ks_sell_lower, ks_sell < ks_sell_upper)
ks_buy_mask = np.logical_and(ks_buy > ks_buy_lower, ks_buy < ks_buy_upper)
bid_density_mask = np.logical_and(bid_density > bid_density_lower, bid_density < bid_density_upper)
ask_density_mask = np.logical_and(ask_density > ask_density_lower, ask_density < ask_density_upper)
spread_mask = np.logical_and(spread > spread_lower, spread < spread_upper)
bid_a_mask = np.logical_and(bid_a > bid_a_lower, bid_a < bid_a_upper)
bid_b_mask = np.logical_and(bid_b > bid_b_lower, bid_b < bid_b_upper)
ask_a_mask = np.logical_and(ask_a > ask_a_lower, ask_a < ask_a_upper)
ask_b_mask = np.logical_and(ask_b > ask_b_lower, ask_b < ask_b_upper)
As_sell_mask = np.logical_and(As_sell > As_sell_lower, As_sell < As_sell_upper)
As_buy_mask = np.logical_and(As_buy > As_buy_lower, As_buy < As_buy_upper)


total_mask = np.logical_and(ks_sell_mask, ks_buy_mask)
total_mask = np.logical_and(total_mask, bid_density_mask)
total_mask = np.logical_and(total_mask, ask_density_mask)
total_mask = np.logical_and(total_mask, spread_mask)
total_mask = np.logical_and(total_mask, bid_a_mask)
total_mask = np.logical_and(total_mask, bid_b_mask)
total_mask = np.logical_and(total_mask, ask_a_mask)
total_mask = np.logical_and(total_mask, ask_b_mask)
total_mask = np.logical_and(total_mask, As_sell_mask)
total_mask = np.logical_and(total_mask, As_buy_mask)


ks_sell = ks_sell[total_mask]
ks_buy = ks_buy[total_mask]
bid_density = bid_density[total_mask]
ask_density = ask_density[total_mask]
spread = spread[total_mask]
bid_a = bid_a[total_mask]
bid_b = bid_b[total_mask]
ask_a = ask_a[total_mask]
ask_b = ask_b[total_mask]
As_sell = As_sell[total_mask]
As_buy = As_buy[total_mask]


# Calculating aggregate statistics


ks_sell_std, ks_sell_mean = np.std(ks_sell), np.mean(ks_sell)
ks_buy_std, ks_buy_mean = np.std(ks_buy), np.mean(ks_buy)
bid_density_std, bid_density_mean = np.std(bid_density), np.mean(bid_density)
ask_density_std, ask_density_mean = np.std(ask_density), np.mean(ask_density)
spread_std, spread_mean = np.std(spread), np.mean(spread)
bid_a_std, bid_a_mean = np.std(bid_a), np.mean(bid_a)
bid_b_std, bid_b_mean = np.std(bid_b), np.mean(bid_b)
ask_a_std, ask_a_mean = np.std(ask_a), np.mean(ask_a)
ask_b_std, ask_b_mean = np.std(ask_b), np.mean(ask_b)

As_sell_std, As_sell_mean = np.std(As_sell), np.mean(As_sell)
As_buy_std, As_buy_mean  = np.std(As_buy), np.mean(As_buy)


np.savetxt("params/ks_sell_std.csv", [ks_sell_std])
np.savetxt("params/ks_sell_mean.csv", [ks_sell_mean])
np.savetxt("params/ks_buy_std.csv", [ks_buy_std])
np.savetxt("params/ks_buy_mean.csv", [ks_buy_mean])
np.savetxt("params/ask_density_std.csv", [ask_density_std])
np.savetxt("params/ask_density_mean.csv", [ask_density_mean])
np.savetxt("params/bid_density_std.csv", [bid_density_std])
np.savetxt("params/bid_density_mean.csv", [bid_density_mean])
np.savetxt("params/spread_std.csv", [spread_std])
np.savetxt("params/spread_mean.csv", [spread_mean])
np.savetxt("params/bid_a_std.csv", [bid_a_std])
np.savetxt("params/bid_a_mean.csv", [bid_a_mean])
np.savetxt("params/bid_b_std.csv", [bid_b_std])
np.savetxt("params/bid_b_mean.csv", [bid_b_mean])
np.savetxt("params/ask_a_std.csv", [ask_a_std])
np.savetxt("params/ask_a_mean.csv", [ask_a_mean])
np.savetxt("params/ask_b_std.csv", [ask_b_std])
np.savetxt("params/ask_b_mean.csv", [ask_b_mean])
np.savetxt("params/As_sell_std.csv", [As_sell_std])
np.savetxt("params/As_sell_mean.csv", [As_sell_mean])
np.savetxt("params/As_buy_std.csv", [As_buy_std])
np.savetxt("params/As_buy_mean.csv", [As_buy_mean])


bid_density = np.reshape(bid_density, (bid_density.shape[0], 1))
ask_density = np.reshape(ask_density, (ask_density.shape[0], 1))
spread = np.reshape(spread, (spread.shape[0], 1))
bid_a = np.reshape(bid_a, (bid_a.shape[0], 1))
bid_b = np.reshape(bid_b, (bid_b.shape[0], 1))
ask_a = np.reshape(ask_a, (ask_a.shape[0], 1))
ask_b = np.reshape(ask_b, (ask_b.shape[0], 1))
As_sell = np.reshape(As_sell, (As_sell.shape[0], 1))
As_buy = np.reshape(As_buy, (As_buy.shape[0], 1))

X_data = np.concatenate([bid_density, ask_density, spread, bid_a, bid_b, ask_a, ask_b], axis=1).T

# Fitting our model to the measured data 

def func(x, a0, a1, a2, a3, a4, a5, a6, a7, a8):

    bd = x[0]
    ad = x[1]
    sp = x[2]
    ba = x[3]
    bb = x[4]
    aa = x[5]
    ab = x[6]
  #  AB = x[7]
  #  AS = x[8]
   

    spread_qt = a0 / (a1 + sp)
    intercept = a8
    density_qt = bd * a2  + ad * a3 
    bid_qt = ba * a4 + bb * a5 
    ask_qt = aa * a6  + ab * a7 
  #  lambda_qt = a9 * AS + a10 * AB

    return spread_qt + intercept + density_qt + bid_qt + ask_qt #+ lambda_qt


ask_params = curve_fit(f=func, xdata=X_data, ydata=ks_sell, maxfev=10000)[0]
bid_params = curve_fit(f=func, xdata=X_data, ydata=ks_buy, maxfev=10000)[0]


ask_params_arr = np.array([ask_params[idx] for idx in range(9)])
bid_params_arr = np.array([bid_params[idx] for idx in range(9)])

np.savetxt("ask_params.csv", ask_params_arr)
np.savetxt("bid_params.csv", bid_params_arr)


ask_res = [func(X_data[:, idx], ask_params[0], ask_params[1], ask_params[2], ask_params[3], ask_params[4], ask_params[5], ask_params[6], ask_params[7], ask_params[8]) for idx in range(X_data.shape[1])]
ask_res = np.array(ask_res)

ask_residuals = ask_res - ks_sell

bid_res = [func(X_data[:, idx], bid_params[0], bid_params[1], bid_params[2], bid_params[3], bid_params[4], bid_params[5], bid_params[6], bid_params[7], bid_params[8]) for idx in range(X_data.shape[1])]
bid_res = np.array(bid_res)

bid_residuals = bid_res - ks_buy


rss = ((ask_res - ks_sell)**2).sum()
tss = ((ks_sell - ks_sell.mean())**2).sum()
print(f'{1 - rss/tss=}')


rss = ((bid_res - ks_buy)**2).sum()
tss = ((ks_buy - ks_buy.mean())**2).sum()
print(f'{1 - rss/tss=}')



ask_residuals_std, ask_residuals_mean = np.std(ask_residuals), np.mean(ask_residuals)
bid_residuals_std, bid_residuals_mean = np.std(bid_residuals), np.mean(bid_residuals)

np.savetxt("params/ask_residuals_std.csv", [ask_residuals_std])
np.savetxt("params/ask_residuals_mean.csv", [ask_residuals_mean])
np.savetxt("params/bid_residuals_std.csv", [bid_residuals_std])
np.savetxt("params/bid_residuals_mean.csv", [bid_residuals_mean])


#Creates a network with 5 convolutional layers, followed by a fully connected layer. 
#1 layer of zero-padding is applied to the right and bottom side of the board.
# Fitting a residual neural network to predict the residual of our easily-interpretable model. Going to see if this improves the R2 of the model. 
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.fc1 = nn.Linear(7, 14)
        self.fc2 = nn.Linear(14, 14)
        self.fc3 = nn.Linear(14, 28)
        self.fc4 = nn.Linear(28, 1)

        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
    
bid_density -= bid_density_mean
bid_density_mean /= bid_density_std

ask_density -= ask_density_mean
ask_density_mean /= ask_density_std

spread -= spread_mean
spread /= spread_std

bid_a -= bid_a_mean
bid_a /= bid_a_std

bid_b -= bid_b_mean
bid_b /= bid_b_std

ask_a -= ask_a_mean
ask_a /= ask_a_std

ask_b -= ask_b_mean
ask_b /= ask_b_std

X_data_norm = np.concatenate([bid_density, ask_density, spread, bid_a, bid_b, ask_a, ask_b], axis=1)
X_data_norm = torch.from_numpy(X_data_norm).to(device).float()

ask_residuals -= ask_residuals_mean
ask_residuals /= ask_residuals_std

bid_residuals -= bid_residuals_mean
bid_residuals /= bid_residuals_std

ask_residuals = torch.from_numpy(ask_residuals).to(device).float()
bid_residuals = torch.from_numpy(bid_residuals).to(device).float()


def train(net: ResNet, side):
    optimizer=optim.Adam(params=net.parameters(), lr=0.001)
    objective=nn.MSELoss()


    
    losses=[]
    iters = 0
    while iters < 20000:
        optimizer.zero_grad()
        preds = net(X_data_norm)
        preds = preds.squeeze()

        if side == "ask":
            loss=objective(preds, ask_residuals)
        else:
            loss=objective(preds, bid_residuals)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        iters += 1

    plt.plot(losses)
   # plt.show()

net = ResNet().to(device)
train(net, "ask")
with torch.no_grad():
    ask_residuals_preds = net(X_data_norm).squeeze()
ask_residuals_preds *= ask_residuals_std
ask_residuals_preds += ask_residuals_mean

ask_residuals_preds = ask_residuals_preds.cpu().numpy()

ask_res -= ask_residuals_preds

rss = ((ask_res - ks_sell)**2).sum()
tss = ((ks_sell - ks_sell.mean())**2).sum()
print(f'{1 - rss/tss=}')

torch.save(net.state_dict(), 'params/net_params_ask.pt') 

net = ResNet().to(device)
train(net, "bid")
with torch.no_grad():
    bid_residuals_preds = net(X_data_norm).squeeze()
bid_residuals_preds *= bid_residuals_std
bid_residuals_preds += bid_residuals_mean

bid_residuals_preds = bid_residuals_preds.cpu().numpy()

bid_res -= bid_residuals_preds

rss = ((bid_res - ks_buy)**2).sum()
tss = ((ks_buy - ks_buy.mean())**2).sum()
print(f'{1 - rss/tss=}')

torch.save(net.state_dict(), 'params/net_params_bid.pt') 





