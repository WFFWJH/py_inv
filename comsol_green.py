import mph
import numpy as np
import time

from paragram import xe, yn, n_patch, n_layer, len_top, wid_top, layers, top0_model, top1_model

nobs = xe.size
data_coord = np.column_stack([
    xe,
    yn,
    np.zeros(nobs),  #z
])

data1 = np.array([xe,yn,np.zeros(nobs)])

Green = np.empty((nobs*3,n_patch*2))

start = time.perf_counter()

client = mph.start()
model_java = client.load(top0_model)

model = model_java.java

model.result().numerical().create("uvw","Interp")
for i in range(0,1):
    for j in range(n_layer):
        model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("size", [str(len_top),str(wid_top)])
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
model_java = client.load(top1_model)

model = model_java.java
model.result().numerical().remove("uvw")
model.result().numerical().create("uvw","Interp")
for i in range(1,layers):
    for j in range(n_layer):
        model.component("comp1").geom("geom1").feature("wp2").geom().feature("r1").set("size", [str(len_top),str(wid_top)])
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