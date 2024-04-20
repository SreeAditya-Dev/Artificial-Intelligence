##Basic Plot

##import matplotlib.pyplot as plt
##import numpy as np
##
##x=np.array([0,6,10,16])
##y=np.array([0,100,25,150])
##plt.plot(x,y)
##plt.show()


####Marker 
####import matplotlib.pyplot as plt
####import numpy as np
####x=np.array([0,6,20,40])
####y=np.array([0,100,200,450])
####plt.plot(x,y,"S")   #s,p,d,P
####plt.show()



####Dotted Plot with line
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,'s:r')
##plt.show()



####Make a solid Line
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,'s-b')
##plt.show()


####Change the size of the plot
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,'s-.b',ms=20)
##plt.show()



##Make colour around & Inside the marker(plot)
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,'s-.b',ms=20,mec="b",mfc="y")
##plt.show()



####Change the size of the plotline
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,ls="-",color="y",linewidth="10")
##plt.show()
##





####Title Assigning
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,ls="-",color="y",linewidth="20")
##plt.title("Intro to Matplotlib")
##plt.ylabel("ylabel")
##plt.xlabel("xlabel")
##plt.show()


####Make Grid Format
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,ls="-",color="b")
##plt.title("Title")
##plt.ylabel("ylabel")
##plt.xlabel("xlabel")
##plt.grid()
##plt.show()
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.plot(x,y,ls="-",color="b")
##plt.title("Title")
##plt.ylabel("ylabel")
##plt.xlabel("xlabel")
##plt.grid(axis="x")
##plt.show()

##
####Making Subplots
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.subplot(2,1,1)
##plt.plot(x,y)
##
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.subplot(2,1,2)
##plt.plot(x,y)
##plt.title("Title")
##plt.ylabel("ylabel")
##plt.xlabel("xlabel")
##plt.show()
##
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.subplot(2,1,1)
##plt.plot(x,y)
##
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.subplot(2,1,2)
##plt.plot(x,y)
##plt.title("Title")
##plt.ylabel("ylabel")
##plt.xlabel("xlabel")
##plt.show()


####Align Supertitle
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.subplot(2,1,2)
##plt.plot(x,y)
##
##
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([0,6,20,40])
##y=np.array([3,8,1,10])
##plt.subplot(2,1,1)
##plt.plot(x,y)
##plt.title("Title")
##plt.ylabel("ylabel")
##plt.xlabel("xlabel")
##plt.suptitle("Common Title")
##plt.show()


####Bar chart
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array(["A","B","C","D","E"])
##y=np.array([10,20,35,60,90])
##plt.bar(x,y)
##plt.show()
##
##
####Bar chart(Horizontal and Colour)
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array(["A","B","C","D","E"])
##y=np.array([10,20,35,60,90])
##plt.barh(x,y, color="g")
##plt.show()


####Create a Histogram
##
##import matplotlib.pyplot as plt
##import numpy as np
##
##x=np.random.normal(160,10,250)
##print(x)
##plt.hist(x)
##plt.show()


##Pie chart

##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([10,20,35,60,90])
##plt.pie(x)
##plt.show()
##
####To give a name
##import matplotlib.pyplot as plt
##import numpy as np
##x=np.array([10,20,35,60,90])
##
##mylabel=["Blueberry","Orange","Green Apple","Apple","Grapes"]
##seperate=[0.15,0.15,0.15,0.15,0.15]
##plt.pie(x,labels=mylabel, explode=seperate)
##plt.show()
