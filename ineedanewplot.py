from matplotlib import pyplot as plt
mainbranch = [[1,0],[2,1],[4,2],[5,3],[6,5],[8,6]]
subbranch = [[3,4],[1,6],[0,8]]
dottedline = [[.7,.9,.8],[5.4,5.3,5.1]]
dottedline2 = [[.7-1.6,.9-1.6,.8-1.6],[5.4+.8,5.3+.8,5.1+.8]]
plt.plot(*zip(*mainbranch),c='black')
plt.arrow(3,4,-4,2,color='red',width=0.05)
plt.plot([5,3],[3,4],'r:')
plt.text(-1,4.3,"General direction", rotation=-26.5651)
plt.text(3.1,4.1,'A')
plt.text(1.1,6.1,'B')
plt.text(.1,8.1,'C')
plt.plot([1,0.6],[6,5.2],'b:')
plt.plot([0,-1],[8,6],'b:')
plt.plot(*dottedline,color='black')
plt.plot(*dottedline2,color='black')
plt.scatter(*zip(*subbranch), c=['blue','green','red'])
plt.scatter(*zip(*mainbranch),c='black')
ax = plt.gca()
ax.set_aspect('equal')
ax.set_axis_off()
plt.show()