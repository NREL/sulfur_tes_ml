import math

d_i = 0.01
d_o = 0.02
pi = math.pi

def get_rho(Tc):
    if Tc <= 422:
        rho = 2137.7 - 0.8487*Tc
    elif Tc > 422 and Tc < 462:
        rho = 21125 - 129.29*Tc+ 0.2885*(Tc**2) - 0.00021506*(Tc**3)
    else:
        rho = 2050.8 - 0.6240*Tc
    return rho

def get_Ac():
    Ac = (pi/4) * (d_o**2 - d_i**2) # Cross-sectional area
    return Ac

def get_m(Tc, Ac):
    # Get mass for a cross-sectional area of pipe, assuming unit length
    rho = get_rho(Tc)
    m = Ac * rho
    return m

def get_Cp(Tc):           
    if Tc <= 392:                  
        Cp = 996.458/1000
    elif Tc > 392 and Tc <= 431.2:
        Cp = ( 3.636e-7*math.exp(1.925/(Tc - 440.4)) ) + ( 0.002564*Tc )
    elif Tc>431.2 and Tc<718:
        Cp= 1.065 + ( 2.599/(Tc- 428) ) - ( 0.3092/((Tc - 428)**2) ) + ( 5.911e-9*((Tc - 428)**3) )
    elif Tc>=718:
        Cp = 1215.535/1000;
    Cp = Cp*1000
    return Cp

def get_As():
    As = pi * d_i # Heat transfer surface area, assuming unit length
    return As

def get_dTc_dt(Tc_list, i, timestep):
    ######################
    # Formula (Centered difference for discrete differentiation):
    #
    #           Tc_(t+1) - Tc_(t-1)
    # dTc/dt = --------------------
    #              2 * timestep
    #
    ######################
    Tc_prev = Tc_list[i - 1][0]
    Tc_next = Tc_list[i + 1][0]
    dTc_dt = (Tc_next - Tc_prev) / (2*timestep)
    return dTc_dt

def get_h(df):
    ######################
    # Formula:
    #
    #     m * Cp * dTc/dt
    # h = ---------------
    #      As * (Tw - Tc)
    #
    ######################
    # This formula assumes it is given a dataframe (df) from a single dataset
    # e.g. the data from a single test run with Tw = 550 and Ti = 440.
    
    Tc_list = df[["Tc"]].to_numpy()
    time = df[["Time"]].to_numpy()
    timestep = time[1][0] - time[0][0]
    Tw = df[["Tw"]].to_numpy()[0][0]
    As = get_As()
    Ac = get_Ac()

    h = list()
    for i, Tc in enumerate(Tc_list):
        Tc = Tc[0]
        # Cannot get centered diff derivative for Tc at first and last datapoints
        if i == 0 or i == len(Tc_list) - 1: 
            continue
        m = get_m(Tc, Ac)
        Cp = get_Cp(Tc)
        dTc_dt = get_dTc_dt(Tc_list, i, timestep)
        h_i = (m * Cp * dTc_dt) / (As * (Tw-Tc))
        h.append(h_i)
    return h