using HDF5
using NearestNeighbors

pickle_fil = "D:/Tonstad/Q40_20s.pickle"

filnamn = "D:/Tonstad/utvalde/Q40.hdf5"

function norm(v)
    return v ./ .√( sum(v.^2) )
end

function ranges()
    y_range = 1:114
    x_range = 1:125

    piv_range = (x_range, y_range)
    return piv_range

end

function lag_tre(t_max = 1) 
    (I,J)= h5open("../../Q40.hdf5", "r") do file
             (read(file, "I"), read(file, "J"))
    end
    
    steps = t_max * 20
    piv_range = ranges()

    
    Umx = h5read("../../Q40.hdf5", "Umx", (:,1:steps))
    Umx_reshape = permutedims(reshape(Umx, (I,J,steps)), [2,1,3])[piv_range,:]

    nonan = .!isnan.(Umx_reshape)

    x = permutedims(reshape(h5read("../../Q40.hdf5", "x"), (I,J)), [2,1])[piv_range]
    y = permutedims(reshape(h5read("../../Q40.hdf5", "y"), (I,J)), [2,1])[piv_range]
    t= 0:0.05:t_max-0.05

    t_3d = permutedims(t .* ones(steps,J,I), [2,3,1])
    x_3d = x[1,:]' .* ones(J,I,steps)
    y_3d = y[:,1] .* ones(J,I,steps)

    xyt = [x_3d[nonan] y_3d[nonan] t_3d[nonan]]' 
   
    tre = KDTree(xyt) 

    return tre
end 

function get_velocity_data(t_max=1, one_dimensional = True)
	steps = t_max * 20
	piv_range = ranges()

	Umx = h5read("../../Q40.hdf5", "Umx", (:,1:steps))
	Vmx = h5read("../../Q40.hdf5", "Vmx", (:,1:steps))
	Umx_reshape = permutedims(reshape(Umx, (I,J,steps)), [2,1,3])[piv_range,:]
	Vmx_reshape = permutedims(reshape(Vmx, (I,J,steps)), [2,1,3])[piv_range,:]

        dx = 1.4692770000000053
        dy = 1.4692770000000053
        dt = 1/20	   

        nonan = .!isnan.(Umx_reshape)

        return Umx_reshape[nonan], Vmx_reshape[nonan]
end

ckdtre = lag_tre(20)
U = get_velocity_data(20)

function get_u(t, x_inn, ckdtre, U, linear=true)
	append!(x,t)

	kd_index = nn(ckdtre, x)

	return U[1][kd_index], U[2][kd_index]
end

function rk3(f, t, y0)
	resultat = 
