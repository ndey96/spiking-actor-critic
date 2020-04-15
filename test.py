import numpy as np
import matplotlib.pyplot as plt
import nengo
import nengo_ocl

# define the model
with nengo.Network() as model:
    stim = nengo.Node(np.sin)
    a = nengo.Ensemble(100, 1)
    b = nengo.Ensemble(100, 1)
    nengo.Connection(stim, a)
    nengo.Connection(a, b, function=lambda x: x**2)

    probe_a = nengo.Probe(a, synapse=0.01)
    probe_b = nengo.Probe(b, synapse=0.01)

# build and run the model
with nengo_ocl.Simulator(model) as sim:
    sim.run(10)

# plot the results
#plt.plot(sim.trange(), sim.data[probe_a])
#plt.plot(sim.trange(), sim.data[probe_b])
#plt.show()
