#run on full sample 

 # load the UV beta measurements
beta_calculator = UV_Beta_Calculator(
     aper_diam, 
    SED_fitter.label
)
beta_calculator(cat)