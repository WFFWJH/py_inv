import mph
import numpy as np
import time
# model.component("comp1").physics("solid").feature("disp1").set("U0", new double[][]{{0}, {-0.5}, {0}});
# model.component("comp1").physics("solid").feature("disp2").set("U0", new double[][]{{0}, {0.5}, {0}});
# model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("pos", new String[]{"4e3", "0"});
# U1 = model.component("comp1").physics("solid").feature("disp1").getStringArray("U0")
# print(list(U1))

# model.sol("sol1").runAll();


# build data
nx, ny = 100, 140  # 点数可改
x = np.linspace(-50, 50, nx)
y = np.linspace(-20, 120, ny)
X, Y = np.meshgrid(x, y)

xe = X.ravel() * 1000.0  # km -> m
yn = Y.ravel() * 1000.0
nobs = xe.size
data_coord = np.column_stack([
    xe,
    yn,
    np.zeros(nobs),  #z
])

# data2 = [[2696.7,9178.9,0.0000],
# [-1391.2,3217.0,0.0000],
# [3126.0,2677.2,0.0000],
# [-2470.8,9524.1,0.0000],
# [2955.5,8842.2,0.0000]]
# data3 = [[2696.7,-1391.2,3126.0,-2470.8,2955.5],
# [9178.9,3217.0,2677.2,9524.1,8842.2],
# [0.0000,0.0000,0.0000,0.0000,0.0000]]
# # 2696.7	9178.9	0.0000	0.034881
# # -1391.2	3217.0	0.0000	0.037900
# # 3126.0	2677.2	0.0000	-0.034328
# # -2470.8	9524.1	0.0000	-0.035475
# # 2955.5	8842.2	0.0000	0.033606



# model.result().numerical().create("uvw","Interp")
# interp = model.result().numerical("uvw")
# interp.setInterpolationCoordinates(data3)
# interp.set("expr",["u","v","w"])
# result = interp.getData()
# result_np = np.asarray(result)
# result_u = result_np[0,0,:]
# result_v = result_np[1,0,:]
# result_w = result_np[2,0,:]
# print(result_np)
# print(result_np.shape)


data1 = np.array([xe,yn,np.zeros(nobs)])

width = 20e3
length = 100e3
len_top = 4e3

n_layer = int(length/len_top)
layers = 5
wid_top = float(width/layers)
n_patch = n_layer*layers
Green = np.zeros((nobs*3,n_patch*2))
# np.empty

start = time.perf_counter()

client = mph.start()
model_java = client.load("top0.mph")

model = model_java.java

model.result().numerical().create("uvw","Interp")
for i in range(0,1):
    for j in range(n_layer):
        model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("pos", [str(len_top*j),str(wid_top*i)])
        model.component("comp1").physics("solid").feature("disp1").set("Direction", [["free"], ["prescribed"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp1").set("U0", [[0],[0.5],[0]])
        model.component("comp1").physics("solid").feature("disp2").set("Direction", [["free"], ["prescribed"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp2").set("U0", [[0],[0.5],[0]])
        model.sol("sol1").runAll();

        interp = model.result().numerical("uvw")
        interp.setInterpolationCoordinates(data1)
        interp.set("expr",["u","v","w"])
        result = model.result().numerical("uvw").getData()
        result_np = np.asarray(result)
        result_u = result_np[0,0,:]
        result_v = result_np[1,0,:]
        result_w = result_np[2,0,:]
        Green[:,i*n_layer+j] = np.concatenate([result_u,result_v,result_w])
        print(i,j,i*n_layer+j)

        model.component("comp1").physics("solid").feature("disp1").set("Direction", [["prescribed"], ["free"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp1").set("U0", [[-0.5],[0],[0]])
        model.component("comp1").physics("solid").feature("disp2").set("Direction", [["prescribed"], ["free"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp2").set("U0", [[0.5],[0],[0]])
        model.sol("sol1").runAll();
        # model.result().numerical().create("uvw","Interp")
        interp.setInterpolationCoordinates(data1)
        interp.set("expr",["u","v","w"])
        result = model.result().numerical("uvw").getData()
        model_java.save("dip")
        result_np = np.asarray(result)
        result_u = result_np[0,0,:]
        result_v = result_np[1,0,:]
        result_w = result_np[2,0,:]
        Green[:,n_patch+i*n_layer+j] = np.concatenate([result_u,result_v,result_w])
        print(i,j,n_patch+i*n_layer+j)

top_time = time.perf_counter() 

client.remove(model_java)
del model
client.clear()
model_java = client.load("top1.mph")

model = model_java.java
model.result().numerical().remove("uvw")
model.result().numerical().create("uvw","Interp")
for i in range(1,layers):
    for j in range(n_layer):
        model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("pos", [str(len_top*j),str(wid_top*i)])
        model.component("comp1").physics("solid").feature("disp1").set("Direction", [["free"], ["prescribed"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp1").set("U0", [[0],[0.5],[0]])
        model.component("comp1").physics("solid").feature("disp2").set("Direction", [["free"], ["prescribed"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp2").set("U0", [[0],[0.5],[0]])
        model.sol("sol1").runAll();

        interp = model.result().numerical("uvw")
        interp.setInterpolationCoordinates(data1)
        interp.set("expr",["u","v","w"])
        result = model.result().numerical("uvw").getData()
        result_np = np.asarray(result)
        result_u = result_np[0,0,:]
        result_v = result_np[1,0,:]
        result_w = result_np[2,0,:]
        Green[:,i*n_layer+j] = np.concatenate([result_u,result_v,result_w])
        print(i,j,i*n_layer+j)

        model.component("comp1").physics("solid").feature("disp1").set("Direction", [["prescribed"], ["free"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp1").set("U0", [[-0.5],[0],[0]])
        model.component("comp1").physics("solid").feature("disp2").set("Direction", [["prescribed"], ["free"], ["free"]]);
        model.component("comp1").physics("solid").feature("disp2").set("U0", [[0.5],[0],[0]])
        model.sol("sol1").runAll();
        # model.result().numerical().create("uvw","Interp")
        interp.setInterpolationCoordinates(data1)
        interp.set("expr",["u","v","w"])
        result = model.result().numerical("uvw").getData()
        model_java.save("dip")
        result_np = np.asarray(result)
        result_u = result_np[0,0,:]
        result_v = result_np[1,0,:]
        result_w = result_np[2,0,:]
        Green[:,n_patch+i*n_layer+j] = np.concatenate([result_u,result_v,result_w])
        print(i,j,n_patch+i*n_layer+j)

client.clear()
over = time.perf_counter()
print("top time: ",top_time-start)
print("bottom time: ",over-top_time)
print("total time: ",over-start)
np.save("Green_comsol.npy",Green)  
exit()