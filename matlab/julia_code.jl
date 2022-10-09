# Copyright Brain Engineering Lab at Dartmouth. All rights reserved.
# Please feel free to use this code for any non-commercial purpose under the CC Attribution-NonCommercial-ShareAlike license: https://creativecommons.org/licenses/by-nc-sa/4.0/
# If you use this code, cite ___.
import Distances

function crop_least_co_occurring_edges(edge_states::Array{UInt8,2}, edges::Array{UInt8,2}, nToRemove::Array{Float64,2})::Array{UInt8,2}
	nToRemove = dropdims(nToRemove; dims=2) # it comes from matlab as a 2d array, even though it's a 1d (2nd dim will be size 1)
    
    cmpIdx = findall(x -> x>0, nToRemove);

	Threads.@threads for i in cmpIdx # don't forget to start julia with --threads=auto
        while nToRemove[i] > 0
            edgeIdx = findall(x -> x>0, edge_states[:,i]) # indices of all the non-0 edges

            code = convert(Array{Float32,2}, edges[edgeIdx,:] .== edge_states[edgeIdx,i])
            
            pDist = Distances.pairwise(Distances.CosineDist(), code; dims=1) # 1 minus cosine similarity
            val,idx = findmax(sum(pDist; dims=2))

            edge_states[edgeIdx[idx],i] = 0
            nToRemove[i] = nToRemove[i] - 1
        end
    end
	return edge_states
end