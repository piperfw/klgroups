using QuantumCumulants, ModelingToolkit, OrdinaryDiffEq, Plots

# Define parameters
@cnumbers w g kappa gamup gamdown

# Define Hilbert space
hf = FockSpace(:cavity)
ha = NLevelSpace(:atom, (:1, :2))
# equivalently hf \otimes<tab> ha
h = tensor(hf, ha)

# Operators acting on part 1/2 of total space h
a = Destroy(h, :a, 1) # a' (apostrophe) gives adjoint
sigma = Transition(h, :σ, 2) # sigma(2,1) is raising operator

# Hamiltonian
H = w*a'*a + g*(a'*sigma(1,2) + a*sigma(2,1))

# Collapse operators
J = [a, sigma(1,2), sigma(2,1)]
rates = [kappa, gamdown, gamup]

# operators to derive equations for
ops = [a'*a, sigma(2,2)]

# Derive equations for initial_ops at order 2 (note ; separates optional args)
eqs = meanfield(ops, H, J; rates=rates, order=2)

# Complete the set of equations
eqs_complete = complete(eqs)

# Generate an ODESystem
@named sys = ODESystem(eqs_complete) # view sys.states etc.

# Solve with trivial initial conditions and some parameters
u0 = zeros(ComplexF64, length(eqs_complete));
p0 = (w, g, kappa, gamup, gamdown) .=> (0, 1.5, 1, 4, .25);
t0 = 0.0;
tend = 10.0;
prob = ODEProblem(sys, u0, (t0, tend), p0);
sol = solve(prob, Tsit5());

# Extract and plot solution (note index using operators, '.' element-wise operator)
n = real.(sol[a'*a])
pe = real.(sol[sigma(2,2)])
plot(sol.t, n, label="n", xlabel="t")
plot!(sol.t, pe, label="pex")

# Running with different parameters now very quick after first run
#p0 = (w, g, kappa, gamup, gamdown) .=> (0, 1.5, 1, 2, .25); # changed gamup
#prob = ODEProblem(sys, u0, (t0, tend), p0);
#sol = solve(prob, Tsit5());
#n = real.(sol[a'*a]);
#pe = real.(sol[sigma(2,2)]);
#plot(sol.t, n, label="n", xlabel="t");
#plot!(sol.t, pe, label="pex")
