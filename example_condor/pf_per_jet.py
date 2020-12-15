import pandas as pd
import numpy as np

folder = '/nfs/dust/cms/group/cms-llp/dataframes_final_Julia/v3_calo_AOD_2018_dnn__v5_SUSY/'

npf=25
npf_count = 0

store = pd.HDFStore(folder+"AK4jets_val_rel.h5")
df = store.select("df",start=0,stop=50000)
#store_backg = pd.HDFStore(folder+"back_val.h5")
#df_backg = store_backg.select("df",start=0,stop=10000)

for i in df.index:
    for j in range(npf):        
        value = df['pt_'+str(j)][i]
        if value > 1:
            npf_count +=1
    #print("Jet "+str(i)+": "+str(npf_count)+" pf candidates!")        
    #npf_count = 0
    
mean = npf_count/len(df)
print("The average number of pf candidates per jet is: "+str(mean))
    
# back_val: 14.84
#sign_val: 10.03
#val: 20.09

#AK4jets_test_rel: 17.7