import math

d_i = 2*0.02
d_o = 2*0.1
pi = math.pi

def get_rho(T):
    if T <= 422:
        rho = 2137.7 - 0.8487*T
    elif T > 422 and T < 462:
        rho = 21125 - 129.29*T+ 0.2885*(T**2) - 0.00021506*(T**3)
    else:
        rho = 2050.8 - 0.6240*T
    return rho

def get_Ac():
    Ac = (pi/4) * (d_o**2 - d_i**2) # Cross-sectional area
    return Ac

def get_m(T, Ac):
    # Get mass for a cross-sectional area of pipe, assuming unit length
    rho = get_rho(T)
    m = Ac * rho
    return m

def get_Cp(T):           
    if T <= 392:                  
        Cp = 996.458/1000
    elif T > 392 and T <= 431.2:
        Cp = ( 3.636e-7*math.exp(1.925/(T - 440.4)) ) + ( 0.002564*T )
    elif T>431.2 and T<718:
        Cp= 1.065 + ( 2.599/(T- 428) ) - ( 0.3092/((T - 428)**2) ) + ( 5.911e-9*((T - 428)**3) )
    elif T>=718:
        Cp = 1215.535/1000;
    Cp = Cp*1000
    return Cp

def get_As():
    As = pi * d_i # Heat transfer surface area, assuming unit length
    return As

def get_dT_dt(T_list, i, timestep):
    ######################
    # Formula (Centered difference for discrete differentiation):
    #
    #           T_(t+1) - T_(t-1)
    # dT/dt = --------------------
    #              2 * timestep
    #
    ######################
    T_prev = T_list[i - 1][0]
    T_next = T_list[i + 1][0]
    dT_dt = (T_next - T_prev) / (2*timestep)
    return dT_dt

def get_h(df):
    ######################
    # Formula:
    #
    #     m * Cp * dT/dt
    # h = ---------------
    #      As * (Tw - T)
    #
    ######################
    # This formula assumes it is given a dataframe (df) from a single dataset
    # e.g. the data from a single test run with Tw = 550 and Ti = 440.
    
    T_list = df[["Tavg"]].to_numpy()
    time = df[["flow-time"]].to_numpy()
    timestep = time[1][0] - time[0][0]
    Tw = df[["Tw"]].to_numpy()[0][0]
    As = get_As()
    Ac = get_Ac()

    h = list()
    for i, T in enumerate(T_list):
        T = T[0]
        # Cannot get centered diff derivative for T at first and last datapoints
        if i == 0 or i == len(T_list) - 1: 
            continue
        m = get_m(T, Ac)
        Cp = get_Cp(T)
        dT_dt = get_dT_dt(T_list, i, timestep)
        h_i = (m * Cp * dT_dt) / (As * (Tw-T))
        h.append(h_i)
    return h

def get_T(T_prev, h, Tw, timestep):
    Ac = get_Ac()
    m = get_m(T_prev, Ac)
    Cp = get_Cp(T_prev)
    As = get_As()
    slope = ( h*As*(Tw - T_prev) )/( m*Cp )
    T = slope*timestep + T_prev
    return T