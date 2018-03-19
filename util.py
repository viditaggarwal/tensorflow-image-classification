import sys

fl = open(sys.argv[1], 'r+')
out = open(sys.argv[2], 'w+')

for line in fl.readlines():
	line = line.strip()
	spl = line.split(',')
	spl[2] = "=IMAGE(\"" + str(spl[2]) + "\";2)"
	out.write(",".join(spl) + "\n")
fl.close()
out.close()