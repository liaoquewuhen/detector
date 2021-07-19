import os 

print(os.path.abspath(__file__))
print(os.path.dirname(os.path.abspath(__file__)))

with open("test.txt","r") as f:
	data = f.read()
	print(data)