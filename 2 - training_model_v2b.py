import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os


# Fijar semilla para replicabilidad
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



############################################
# Carga de data preprocesada
############################################

symbols = ['AAPL', 'AMZN', 'GOOG', 'MSFT', 'META']
stock_data = {}
for symbol in symbols:
    df = pd.read_csv(f"{symbol}_final.csv", index_col=0, parse_dates=True)
    stock_data[symbol] = df

macro_data = pd.read_csv("macro_final.csv", index_col=0, parse_dates=True)

# Obtener el conjunto de fechas comunes
all_dates = sorted(list(set.intersection(*[set(df.index) for df in stock_data.values()],
                                           set(macro_data.index))))
n = len(all_dates)
# División temporal: 70% train, 15% validación, 15% test
train_dates = all_dates[:int(n * 0.7)]
valid_dates = all_dates[int(n * 0.7):int(n * 0.85)]
test_dates  = all_dates[int(n * 0.85):]

# Calcular dimensión del estado: suma de features de cada acción + features macro
state_dim = sum(df.shape[1] for df in stock_data.values()) + len(macro_data.columns)
print(f"Dimensión del estado: {state_dim}")

############################################
# Definición del modelo DRLTrader
############################################

class DRLTrader(nn.Module):
    def __init__(self, state_dim, n_stocks):
        super(DRLTrader, self).__init__()
        self.n_stocks = n_stocks
        self.fc1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, n_stocks * 3)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.ln1(torch.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.ln2(torch.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        # Reestructurar la salida y aplicar softmax por grupo de 3 acciones (Buy, Sell, Hold)
        x = x.reshape(-1, self.n_stocks, 3)
        x = torch.softmax(x, dim=2)
        return x.reshape(-1, self.n_stocks * 3)

############################################
# Definición del entorno de trading
############################################

class TradingEnv:
    def __init__(self, stock_data, macro_data, date_range=None, initial_balance=100000, 
                 transaction_fee=0.005, cooldown_period=5, atr_window=14, 
                 atr_stop_loss_multiplier=1.5, atr_take_profit_multiplier=3, 
                 trailing_stop_multiplier=1.5, correlation_threshold=0.8):
        self.stock_data = stock_data
        self.macro_data = macro_data
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        if date_range is None:
            self.dates = sorted(list(set.intersection(*[set(df.index) for df in stock_data.values()],
                                                       set(macro_data.index))))
        else:
            self.dates = date_range
            
        self.stocks = list(stock_data.keys())
        self.n_stocks = len(self.stocks)
        
        # Parámetros para cooldown y gestión de riesgo
        self.cooldown_period = cooldown_period
        self.atr_window = atr_window
        self.atr_stop_loss_multiplier = atr_stop_loss_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier
        self.trailing_stop_multiplier = trailing_stop_multiplier
        self.correlation_threshold = correlation_threshold
        
        # Inicializar cooldown para cada activo
        self.cooldown = {stock: 0 for stock in self.stocks}
        
        # Calcular matriz de correlaciones entre activos (usando precios de cierre en las fechas comunes)
        prices_df = pd.DataFrame({stock: stock_data[stock].loc[self.dates]['Close'] for stock in self.stocks})
        self.correlation_matrix = prices_df.corr()
        
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio = {stock: 0 for stock in self.stocks}
        # Inicializar registros de trades y trade abierto para cada stock
        self.trade_records = {stock: [] for stock in self.stocks}
        self.current_trade = {stock: None for stock in self.stocks}
        self.portfolio_value_history = [self.initial_balance]
        # Reiniciar cooldown
        self.cooldown = {stock: 0 for stock in self.stocks}
        return self._get_state()

    def _get_state(self):
        state = []
        for stock in self.stocks:
            row = self.stock_data[stock].loc[self.dates[self.current_step]]
            state.extend(row.values)
        state.extend(self.macro_data.loc[self.dates[self.current_step]].values)
        return np.array(state, dtype=np.float32)
    
    def compute_atr(self, stock, window=None):
        """Calcula el ATR para el activo usando un período determinado (por defecto atr_window)."""
        if window is None:
            window = self.atr_window
        current_date = self.dates[self.current_step]
        df = self.stock_data[stock]
        df_window = df.loc[:current_date].tail(window)
        if len(df_window) < 2:
            return 0.0
        df_window = df_window.copy()
        df_window['prev_close'] = df_window['Close'].shift(1)
        df_window.dropna(inplace=True)
        tr1 = df_window['High'] - df_window['Low']
        tr2 = (df_window['High'] - df_window['prev_close']).abs()
        tr3 = (df_window['Low'] - df_window['prev_close']).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean().iloc[-1]
        return atr

    def step(self, actions):
        # Actualizar cooldown: decrementar el contador para cada activo
        for stock in self.stocks:
            if self.cooldown[stock] > 0:
                self.cooldown[stock] -= 1

        current_value = self.get_portfolio_value()
        actions = actions.reshape(self.n_stocks, 3)
        
        # Gestión de riesgo: revisar trades abiertos para aplicar stop loss, take profit y trailing stop
        for stock in self.stocks:
            if self.current_trade[stock] is not None:
                price = self.stock_data[stock].loc[self.dates[self.current_step]]['Close']
                current_atr = self.compute_atr(stock)
                trade = self.current_trade[stock]
                # Actualizar el máximo alcanzado y el trailing stop
                if price > trade['max_price']:
                    trade['max_price'] = price
                trailing_stop = trade['max_price'] - self.trailing_stop_multiplier * current_atr
                trade['trailing_stop'] = trailing_stop
                # Si se cumple alguna condición de riesgo, forzar la venta
                if price <= trade['stop_loss'] or price >= trade['take_profit'] or price <= trailing_stop:
                    shares = self.portfolio[stock]
                    sell_value = shares * price * (1 - self.transaction_fee)
                    self.balance += sell_value
                    pnl = sell_value - trade['entry_cost']
                    self.trade_records[stock].append({
                        'entry_price': trade['entry_price'],
                        'exit_price': price,
                        'shares': shares,
                        'pnl': pnl,
                        'entry_step': trade['entry_step'],
                        'exit_step': self.current_step,
                        'reason': 'risk management'
                    })
                    self.portfolio[stock] = 0
                    self.current_trade[stock] = None
                    self.cooldown[stock] = self.cooldown_period

        # Procesar las acciones del agente
        for i, stock in enumerate(self.stocks):
            price = self.stock_data[stock].loc[self.dates[self.current_step]]['Close']
            action = actions[i].argmax()  # 0: Buy, 1: Sell, 2: Hold
            if action == 0:  # Comprar
                # Solo comprar si no hay posición abierta, no está en cooldown y hay saldo disponible
                if self.current_trade[stock] is None and self.cooldown[stock] == 0 and self.balance > 0:
                    # Diversificación: calcular asignación objetivo
                    total_value = self.get_portfolio_value()
                    target_allocation = total_value / self.n_stocks
                    current_investment = self.portfolio[stock] * price  # normalmente 0 si no se tiene posición
                    if current_investment >= target_allocation:
                        continue  # ya se alcanzó la asignación deseada para este activo
                    # Diversificación: evitar comprar si hay alta correlación con otros activos en cartera
                    correlated = False
                    for other in self.stocks:
                        if other != stock and self.portfolio[other] > 0:
                            if self.correlation_matrix.loc[stock, other] > self.correlation_threshold:
                                correlated = True
                                break
                    if correlated:
                        continue  # se omite la compra por alta correlación
                        
                    max_investment = min(self.balance * 0.2, target_allocation - current_investment)
                    shares = int(max_investment / price) if price > 0 else 0
                    cost = shares * price * (1 + self.transaction_fee)
                    if cost <= self.balance and shares > 0:
                        self.portfolio[stock] = shares
                        self.balance -= cost
                        current_atr = self.compute_atr(stock)
                        self.current_trade[stock] = {
                            'entry_price': price,
                            'shares': shares,
                            'entry_cost': cost,
                            'entry_step': self.current_step,
                            'stop_loss': price - self.atr_stop_loss_multiplier * current_atr,
                            'take_profit': price + self.atr_take_profit_multiplier * current_atr,
                            'trailing_stop': price - self.trailing_stop_multiplier * current_atr,
                            'max_price': price
                        }
            elif action == 1:  # Vender (acción manual)
                if self.current_trade[stock] is not None and self.portfolio[stock] > 0:
                    shares = self.portfolio[stock]
                    sell_value = shares * price * (1 - self.transaction_fee)
                    self.balance += sell_value
                    trade = self.current_trade[stock]
                    pnl = sell_value - trade['entry_cost']
                    self.trade_records[stock].append({
                        'entry_price': trade['entry_price'],
                        'exit_price': price,
                        'shares': shares,
                        'pnl': pnl,
                        'entry_step': trade['entry_step'],
                        'exit_step': self.current_step,
                        'reason': 'manual sell'
                    })
                    self.portfolio[stock] = 0
                    self.current_trade[stock] = None
                    self.cooldown[stock] = self.cooldown_period
            # Si la acción es Hold, no se realiza acción para ese activo.
        self.current_step += 1
        new_value = self.get_portfolio_value()
        reward = (new_value - current_value) / current_value
        self.portfolio_value_history.append(new_value)
        done = self.current_step >= len(self.dates) - 1
        return self._get_state(), reward, done

    def get_portfolio_value(self):
        total = self.balance
        for stock in self.stocks:
            price = self.stock_data[stock].loc[self.dates[self.current_step]]['Close']
            total += self.portfolio[stock] * price
        return total

############################################
# Creación de ambientes: Train, Validación, Test
############################################

train_env = TradingEnv(stock_data, macro_data, date_range=train_dates)
valid_env = TradingEnv(stock_data, macro_data, date_range=valid_dates)
test_env  = TradingEnv(stock_data, macro_data, date_range=test_dates)

############################################
# Entrenamiento del modelo en el ambiente de TRAIN
############################################

model = DRLTrader(state_dim, train_env.n_stocks)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_episodes = 30
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.95
memory = deque(maxlen=10000)

train_returns = []
train_portfolio_values = []

for episode in range(n_episodes):
    state = train_env.reset()
    total_reward = 0
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    
    while True:
        if random.random() < epsilon:
            action = np.random.rand(train_env.n_stocks * 3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action = model(state_tensor.unsqueeze(0)).squeeze(0).numpy()
        next_state, reward, done = train_env.step(action)
        total_reward += reward
        memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            states = torch.FloatTensor(np.array(states))
            actions = torch.FloatTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))
            
            current_q = model(states)
            next_q = model(next_states).detach()
            target_q = rewards + gamma * torch.max(next_q.reshape(batch_size, -1, 3), dim=2)[0].sum(dim=1) * (1 - dones)
            loss = nn.MSELoss()(torch.sum(current_q * actions, dim=1), target_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    train_returns.append(total_reward)
    train_portfolio_values.append(train_env.portfolio_value_history[-1])
    print(f"[TRAIN] Episodio {episode+1}: Retorno total: {total_reward:.4f}, Valor final: ${train_env.portfolio_value_history[-1]:.2f}, Epsilon: {epsilon:.4f}")

# Guardar el modelo entrenado
model_file = os.path.join("trained_drl_model_v2b.pth")
torch.save(model.state_dict(), model_file)
print(f"Modelo guardado en: {model_file}")

############################################
# Evaluación en los ambientes de VALIDACIÓN y TEST
############################################

def evaluate_env(env, model):
    state = env.reset()
    rewards = []
    while True:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action = model(state_tensor.unsqueeze(0)).squeeze(0).numpy()
        state, reward, done = env.step(action)
        rewards.append(reward)
        if done:
            break
    final_value = env.portfolio_value_history[-1]
    total_return = (final_value - env.initial_balance) / env.initial_balance
    annual_return = (1 + total_return) ** (252 / len(env.dates)) - 1
    sharpe_ratio = np.mean(rewards) / np.std(rewards) * np.sqrt(252)
    
    # Cálculo del Máximo Drawdown
    portfolio_values = np.array(env.portfolio_value_history)
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cumulative_max) / cumulative_max
    max_drawdown = abs(drawdowns.min())
    
    # Retornamos además el capital inicial
    initial_capital = env.initial_balance
    return initial_capital, total_return, annual_return, sharpe_ratio, final_value, max_drawdown

(valid_initial_capital, valid_total_return, valid_annual_return, 
 valid_sharpe, valid_final_value, valid_max_dd) = evaluate_env(valid_env, model)
(test_initial_capital, test_total_return, test_annual_return, 
 test_sharpe, test_final_value, test_max_dd) = evaluate_env(test_env, model)

print("\n[VALIDACIÓN] Resultados:")
print(f"Capital Inicial: ${valid_initial_capital:.2f}")
print(f"Capital Final: ${valid_final_value:.2f}")
print(f"Retorno total: {valid_total_return*100:.2f}%")
print(f"Retorno anualizado: {valid_annual_return*100:.2f}%")
print(f"Ratio de Sharpe: {valid_sharpe:.2f}")
print(f"Máximo Drawdown: {valid_max_dd*100:.2f}%")

print("\n[TEST] Resultados:")
print(f"Capital Inicial: ${test_initial_capital:.2f}")
print(f"Capital Final: ${test_final_value:.2f}")
print(f"Retorno total: {test_total_return*100:.2f}%")
print(f"Retorno anualizado: {test_annual_return*100:.2f}%")
print(f"Ratio de Sharpe: {test_sharpe:.2f}")
print(f"Máximo Drawdown: {test_max_dd*100:.2f}%")

############################################
# Función para calcular métricas de performance por acción
############################################

def compute_performance_metrics(env):
    performance = {}
    for stock in env.stocks:
        trades = env.trade_records[stock]
        trades_count = len(trades)
        if trades_count > 0:
            pnl_total = sum(trade['pnl'] for trade in trades)
            win_count = sum(1 for trade in trades if trade['pnl'] > 0)
            win_rate = win_count / trades_count
        else:
            pnl_total = 0.0
            win_rate = 0.0
        performance[stock] = {'win_rate': win_rate, 'pnl_total': pnl_total, 'trades_count': trades_count}
    return performance

# Calcular y mostrar las métricas por acción para los ambientes de VALIDACIÓN y TEST
valid_performance = compute_performance_metrics(valid_env)
test_performance = compute_performance_metrics(test_env)

print("\n[VALIDACIÓN] Métricas por acción:")
for stock, metrics in valid_performance.items():
    print(f"{stock} - Win Rate: {metrics['win_rate']*100:.2f}%, PnL Total: ${metrics['pnl_total']:.2f}, Trades: {metrics['trades_count']}")

print("\n[TEST] Métricas por acción:")
for stock, metrics in test_performance.items():
    print(f"{stock} - Win Rate: {metrics['win_rate']*100:.2f}%, PnL Total: ${metrics['pnl_total']:.2f}, Trades: {metrics['trades_count']}")
