import json, sys
from imageBackgroundColor import getImageColors
import pandas as pd


df = pd.read_csv(sys.argv[1])
imageUrls = df.Image_ID.tolist()
result = []
cnt = 0
for url in imageUrls:
	try:
		cnt = cnt+1
		result.append(json.loads(getImageColors(url)))
		if cnt > 100:
			break
	except:
		None
print json.dumps(result)
