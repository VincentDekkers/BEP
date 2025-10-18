# File to generate te relation between the branches
name = "m/grondmetingen5-73x500ns2025-05-27_14-25-10/grondmetingen5-73x500ns2025-05-27_14-25-10"
with open(f'{name}.txt', 'r') as file: data = [eval(i[:-1]) for i in file.readlines()] # read the branches' file
allrelations = []
for j in data:
    starts, ends, relations= [],[],[] # initiate empty lists
    for i in j: starts.append(i[0]); ends.append(i[-1]) #make a list of start and end coordinates
    for i,start in enumerate(starts[1:],1): relations.append([ends.index(start),i]) # makes a list of the form [[parent, daughter]]
    allrelations.append(relations)
with open(f'{name}-relations.txt','w') as file: #to write the relations
    for relations in allrelations: file.write(str(relations)+'\n')

