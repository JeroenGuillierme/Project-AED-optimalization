import pandas as pd

aed = pd.read_parquet('DATA/aed_locations.parquet.gzip')

print(aed)

aed1 = aed.head(50)
print(aed1)