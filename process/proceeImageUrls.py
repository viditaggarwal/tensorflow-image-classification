import json

fl = open('image_urls_list_3.txt', 'r')
out = open('images_handbags_list_3.csv', 'w+')
for line in fl.readlines():
	line = line.strip()
	data = json.loads(line)
	for v in data:
		out.write("=IMAGE(\"" + v + "\";2)\n")
	