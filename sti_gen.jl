pickle_fil = "D:/Tonstad/Q40_20s.pickle"

filnamn = "D:/Tonstad/utvalde/Q40.hdf5"

function norm(v)
    return v ./ .√( sum(v.^2) )
end

function ranges()
    y_range = 