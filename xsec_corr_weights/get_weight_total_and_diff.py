import numpy as np
import pickle, os, sys

pth = '/scratch/mliubar/jupyter_notebooks/xsec_corr_weights/'
extrap_dict = pickle.load(open(os.path.join(pth, 'tot_xsec_corr_Q2min1_isoscalar.pckl'), 'rb'), encoding='latin1')

pth_spl = pth+'splines_flat/'
wf_nucc = pickle.load(open(pth_spl+'NuMu_CC_flat.pckl','rb'), encoding='latin1') 
wf_nubarcc = pickle.load(open(pth_spl+'NuMu_Bar_CC_flat.pckl','rb'), encoding='latin1') 
wf_nunc = pickle.load(open(pth_spl+'NuMu_NC_flat.pckl','rb'), encoding='latin1') 
wf_nubarnc = pickle.load(open(pth_spl+'NuMu_Bar_NC_flat.pckl','rb'), encoding='latin1') 

#coef = 0. -- Genie -> Genie
#coef = 1. -- Genie -> Nugen

def get_weight_total(E, nu='Nu', current='CC', ext_type='constant', lgE_min_cust=2., coef=1.):
    
    if nu not in ['Nu', 'NuBar']:
        print("Nu type is`t correctly specified! Possible values are: 'Nu', 'NuBar'.\n Returning weight value = 1.")
        return 1.
    if current not in ['CC', 'NC']:
        print("Current is`t correctly specified! Possible values are: 'CC', 'NC'.\n Returning weight value = 1.")
        return 1.
    if ext_type not in ['constant', 'linear', 'higher']:
        print("Extrapolation type is`t specified correctly! Possible values are: 'constant', 'linear', 'higher'.\n Returning weight value = 1.")
        return 1.    
    
    if ext_type == 'constant':
        lgE_min = lgE_min_cust
    else:
        lgE_min = 1.68
    
    lgE = np.log10(E)

    fit_lgE_reg = np.transpose(np.argwhere(lgE > lgE_min))[0]
    ext_lgE_reg = np.transpose(np.argwhere(lgE <= lgE_min))[0]
    
    w = np.ones(len(lgE))
    
    poly_coef = extrap_dict[nu][current]['poly_coef']
    w[fit_lgE_reg] = np.polyval(poly_coef, lgE[fit_lgE_reg])
    
    if ext_type == 'constant':
        poly_coef = extrap_dict[nu][current]['poly_coef']
        w[ext_lgE_reg] = np.polyval(poly_coef, lgE_min*w[ext_lgE_reg]) 
#         w[ext_lgE_reg] = extrap_dict[nu][current]['c']*w[ext_lgE_reg]
    elif ext_type == 'linear':
        lin_coef = extrap_dict[nu][current]['linear']
        w[ext_lgE_reg] = np.polyval(lin_coef, lgE[ext_lgE_reg])
    else:
        poly_coef = extrap_dict[nu][current]['poly_coef']
        w[ext_lgE_reg] = np.polyval(poly_coef, lgE[ext_lgE_reg]) 
    
       
    w_coef = w*(1. + (1./w - 1)*(1. - coef))
        
    return w_coef

def get_weight_diff(E, y, xs_type, coef=1., lgE_min=2.): 
    
    lgE = np.log10(E)
    
    spl_lgE_reg = np.transpose(np.argwhere(lgE > lgE_min))[0]
    ext_lgE_reg = np.transpose(np.argwhere(lgE <= lgE_min))[0]
    
    w = np.ones(len(lgE))
    
    if xs_type == 'NuCC':
        weight_func = wf_nucc
    elif xs_type == 'NuBarCC':
        weight_func = wf_nubarcc
    elif xs_type == 'NuNC':
        weight_func = wf_nunc
    elif xs_type == 'NuBarNC':
        weight_func = wf_nubarnc
    else:
        print ('specify xs_type correctly!')
    
    w[spl_lgE_reg] = weight_func.ev(lgE[spl_lgE_reg],y[spl_lgE_reg])
    w[ext_lgE_reg] = weight_func.ev(w[ext_lgE_reg]*lgE_min,y[ext_lgE_reg])
    w_coef = w*(1. + (1./w - 1)*(1. - coef))
    
    return w
