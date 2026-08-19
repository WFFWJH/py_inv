n_layer = 5
l_top = 4e3
w = 4e3

G = np.array(n_point*3,n_patch*2)
G_u = np.array(n_point,n_patch*2)
G_v
G_w

for i in range(n_layer):
    for j in range(n_every_layer):
        set_wp(l_top*j,-w*i)
        (-0.5,0,0)(0.5,0,0)
        runall()
        u,v,w = eveluate(u,v,w,points)
        G_u(i*n_every_layer+j,:) = u;
        G_v
        G_w
        (0,0.5,0)(0,0.5,0)
        runall()
        uvw = eveluate(u,v,w,points)
        G_u(n_point+i*n_every_layer+j,:) = u;
        G_v
        G_w


