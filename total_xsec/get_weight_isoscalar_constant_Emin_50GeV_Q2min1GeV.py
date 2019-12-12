#import pickle
import numpy as np
import pickle, os, sys
# extrap_dict = np.load('constant_isoscalar_Emin_50GeV.npy')[0]
pth = '/scratch/mliubar/jupyter_notebooks/total_xsec'
extrap_dict = pickle.load(open(os.path.join(pth, 'Q2min1GeV_constant_isoscalar_Emin_50GeV'), 'rb'), encoding='latin1')

def get_weight_iso(E, nu='Nu', current='CC', ext_type='constant', lgE_min_cust=1.68):
    
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
#     print ( 'lgE_min = ', lgE_min )
    
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
    
    
#     if lgE > lgE_min:
#         poly_coef = extrap_dict[nu][current]['poly_coef']
#         w = np.polyval(poly_coef, lgE)
#     else:
#         if ext_type == 'constant':
#             w = extrap_dict[nu][current]['c']
#         elif ext_type == 'linear':
#             lin_coef = extrap_dict[nu][current]['linear']
#             w = np.polyval(lin_coef, lgE)
#         else:
#             poly_coef = extrap_dict[nu][current]['poly_coef']
#             w = np.polyval(poly_coef, lgE)
        
    return w
