import pandas as pd
from statsmodels.tsa.stattools import acf
from sys import argv

def auto_corr(row):
    res =  acf(row, fft=True, nlags=500, missing='drop')
    quarterly = res[120] if len(res) > 120 else 0
    yearly = res[365] if len(res) > 365 else 0
    return pd.Series({'quarterly' : quarterly, 'yearly' : yearly})

def translate(element, translation_dict):
    return translation_dict.index(element)

original_file = argv[1]
outfile = argv[2]
nrows = 10000

raw = pd.read_csv(original_file)

t1 = raw
t1 = t1.set_index('Page')

t1 = t1.dropna(axis=0, how='any')

# calculate correlation values
t1 = t1.merge(t1.apply(auto_corr, axis=1), left_index=True, right_index=True)

# split page column into parts
t1 = t1.reset_index()
t1[['d1', 'name', 'country_code', 'type', 'access', 'd2']] = t1.Page.str.split('(.*)_(.*?)\..*?.org_(.*)_(.*)', expand=True)
t1 = t1.drop(['d1', 'd2'], axis=1)

t1 = t1.set_index('Page')

# convert categorical columns from string to integer
cols_to_convert = ['name', 'country_code', 'type', 'access']
translation_dicts = {}
for col in cols_to_convert:
    this_col = t1[col]
    translation_dicts[col] = this_col.unique().tolist()
    t1[col] = this_col.apply(lambda x: translate(x, translation_dicts[col]))

# drop any rows with NaN
t1 = t1.dropna(axis=0, how='any')

# write to file
t1.to_csv(outfile)

print "SAMPLE OUTPUT : "
print t1.head()
