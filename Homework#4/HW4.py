import numpy as np

class InventoryEnv:
  def __init__(self, T = 100, I_max = 50, Q_max = 25, K = 100, C = 3, H = 5, P = 20, L = 15):

    self.T = T            #study period
    self.I_max = I_max    #maximum inventory level
    self.Q_max = Q_max    #maximum daily order
    self.K = K            #fixed order cost
    self.C = C            #variable order cost
    self.H = H            #holding cost
    self.P = P            #penalty cost for lost sales
    self.L = L            #demand rate (Poisson distribution)

    #Define state space
    self.state_dim = 2
    self.state_space = [(t, I) for t in range(self.T + 1) for I in range(self.I_max + 1)]
    self.terminal_states = [(self.T, I) for I in range(self.I_max + 1)]

    #Define action space
    self.action_dim = self.Q_max + 1
    self.action_space = [q for q in range(self.Q_max + 1)]

  def reset(self):
    self.state = (0, 0)

    return np.array(self.state, dtype=np.float32)

  def step(self, action):
    #current state
    t, I = self.state

    #order quantity cannot exceed the maximum inventory level
    q = min(action, self.I_max - I)

    #inventory level before demand occurs
    I_opening = I + q

    #demand realization
    D = np.random.poisson(self.L)

    #sales, lost sales, final inventory level
    sales = min(I_opening, D)
    lost_sales = max(D - I_opening, 0)
    I_closing = max(I_opening - D, 0)

    #cost computation
    if q > 0:
      order_cost = self.K + self.C * q
    else:
      order_cost = 0

    holding_cost = self.H * I_closing
    penalty_cost = self.P * lost_sales

    total_cost = order_cost + holding_cost + penalty_cost
    reward = -total_cost

    self.state = (t + 1, I_closing)

    if self.state in self.terminal_states:
      done = True
    else:
      done = False

    '''
    Use the transition information if necessary
    info = {"day": t,
        "inventory": I,
        "order_quantity": q,
        "opening_inventory": I_opening,
        "demand": D,
        "sales": sales,
        "lost_sales": lost_sales,
        "closing_inventory": I_closing,
        "order_cost": order_cost,
        "holding_cost": holding_cost,
        "penalty_cost": penalty_cost,
        "total_cost": total_cost}
    '''

    return np.array(self.state, dtype=np.float32), reward, done, #info