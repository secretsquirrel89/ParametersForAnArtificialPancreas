using DifferentialEquations, Flux, DiffEqFlux, Plots, LinearAlgebra


function ODEsForParametersForArtificialPancreasController(du, u, p, t)
    
    # Parameter vector
    
    
    
    t_Max_IA, t_Max_G, A_G, S_I, K, G_b = p
    
    x_1, x_2, a_1, a_2, G = u
    
    
    # W, Weight of patient 
    # [kilograms]
    
    
    
    W = 75 # [kg]
    
    
    
    # MCR_I, Metabolic clearance rate of effective insulin, 
    # [Liters/kilogram/minute] or [L/kg/min]
    
    # Source: A. Haidar et al., “Pharmacokinetics of insulin aspart in pump-treated
    # subjects with type 1 diabetes: reproducibility and effect of age, weight, and
    # duration of diabetes,” Diabetes Care, vol. 36, pp. e173–e174, Oct. 2013.

    
    
    MCR_I = 0.017 # [L/kg/min]
    
    
    
    # V_G, Plasma glucose pool size,
    # [Liters/kilogram] or [L/kg]
    
    # Source: R. Hovorka et al., “Partitioning glucose distribution/transport, disposal,
    # and endogenous production during IVGTT,” Amer. J. Physiol. Endocrinol.
    # Metabolism, vol. 282, pp. E992–E1007, May 2002.

    
    
    V_G = 0.16 # [L/kg]
    
    
    
    # X_b, the basal effective insulin concentration,
    # [milliUnits/Liter] or [mU/L]
    
    # Source: Estimated from MODELING DAY-TO-DAY VARIABILITY OF GLUCOSE–INSULIN REGULATION, 
    # on page 1415, part C. in steady-state [with near-constant parameters and 
    # variables] conditions, when X_t equals X_b, G_t equals G_b, and u_1 equals
    # u_2 which also equals x_ss. x_ss represents a constant insulin infusion rate
    # of 0.017 Units/kilogram/hour [U/kg/h] or 1.2 Units/hour [U/h] at a 70 kilogram
    # [kg] body weight
    
    
    
    X_b = 12.9 # [mU/L]
    
    
    
    # U_I, insulin infusion rate from the insulin pump used by patient,
    # [Units/hour] or [U/h]
    
    # Source: from user, via data upload of insulin pump to computer
    # Dummy value, for now [will be a vector of values represented at time points
    # with if/else if/else statements later]
    
    
    
    U_I = 7.2 # [U/h]
    
    
    
    # CHO, carbohydrates consumed
    # [grams] or [g]
    
    # Source: from user (counted prior to consumption of food)
    # Dummy value, for now [will be a vector of values represented at time points
    # with if/else if/else statements later]
    
    
    
    CHO = 75 # [g]
    
    
    
    #############################################################################
    #                                                                           #
    #                                                                           #
    #                                                                           #
    #              SYSTEM OF ORDINARY DIFFERENTIAL EQUATIONS                    #
    #                                                                           #
    #                                                                           #
    #                                                                           #
    #############################################################################
    
    
    #############################################################################
    #                INSULIN ABSORPTION AND ACTION SUBMODEL                     #
    #############################################################################
    
    
    
    
    # Original Equation:
    # dx_1(t)/dt = ((-1/t_Max_IA)*x_1(t)) + (U_I(t)/60)
    # [Units] or [U]
    
    
    du[1] = (-(1/t_Max_IA)*x_1) + (U_I/60)
    
    
    
    # Original Equation:
    # dx_2(t)/dt = (1/t_Max_IA)*(x_1(t) - x_2(t))
    
    
        
    du[2] = (1/t_Max_IA)*(x_1 - x_2)
    
    
    
    # Original Equation:
    # X(t) = (1000*x_2(t))/(t_Max_IA*MCR_I*W)
    
    
            
    X_t = (1000*x_2)/(t_Max_IA*MCR_I*W)
    
    
    
    
    #############################################################################
    #                      MEAL ABSORPTION DYNAMICS SUBMODEL                    #
    #############################################################################
    
    # Original Equation:
    
        
    du[3] = (-(1/t_Max_G)*a_1) + CHO
            
    du[4] = (1/t_Max_G)*(a_1-a_2)
        
    U_M = (5.556*A_G*a_2)/(t_Max_G*V_G*W)
    
    
    #############################################################################
    #                          GLUCOSE DYNAMICS SUBMODEL                        #
    #############################################################################
            
    du[5] = (-S_I*(X_t - X_b)) + (U_M - (K*(G-G_b)))
        
end


t_Max_IA = Float32[78.0]

t_Max_G = Float32[48.0]

A_G = Float32[0.84]

S_I = Float32[0.0050]

K = Float32[0.0039]

G_b = Float32[6.6]

p = [t_Max_IA; t_Max_G; A_G; S_I; K; G_b]

tspan = (0.0f0, 5.0f0)

ts = range(0.0f0, 60.0f0, length=61)

u0 = Float32[4.2, 2.56, 30, 30, 10]

prob = ODEProblem(ODEsForParametersForArtificialPancreasController, u0, tspan, p)
     
sol = solve(prob, Tsit5(), saveat = ts)

# Neural Network for Ordinary Differential Equation

# Model

model = Chain(
    Dense(5, 50, swish),
    Dense(50, 5))

p = param([78.0, 48.0, 0.84, 0.0050, 0.0039, 6.6]) # Initial Parameter Vector

params = Flux.Params([p])

function predict_rd() # 1-Layer Neural Network
    
    Flux.Tracker.collect(diffeq_rd(p, prob, Tsit5(), saveat = ts))
    
end

loss_rd() = sum(abs2, G - 10 for G in predict_rd()) # Loss Function

# Callback

cb = function()
    
    display(loss_rd())
    display(params)
    display(predict_rd())
    display(plot(solve(remake(prob,p=Flux.data(p)),Tsit5(),saveat=0.0:0.1:5.0),ylim=(0,12)))
end

# Training

data = Iterators.repeated((), 100000000)

opt = ADAM(0.1)
    
cb()

Flux.train!(loss_rd, params, data, opt, cb = Flux.throttle(cb, 1))
