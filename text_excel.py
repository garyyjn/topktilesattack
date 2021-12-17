import pandas as pd

a = pd.read_excel('Table_S1.2017_08_05.xlsx', 'Master table')

print(a[a['Case ID'] == 'TCGA-4Z-AA7Q'])