import os

print("Enter the folder name:")
name = input()

for file in os.listdir(name):
    if ' 2.' in file:
        os.remove(os.path.join(name, file))