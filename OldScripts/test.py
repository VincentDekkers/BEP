# from wolframclient.evaluation import WolframLanguageSession
# from wolframclient.language import wl, wlexpr
# import time

# session = WolframLanguageSession()
# session.evaluate(r"g[\[Theta]_]:=Integrate[Sqrt[1 + Tan[\[Theta]]^2 Cos[\[Phi]]^2]/\[Pi], {\[Phi], 0, \[Pi]}] // FullSimplify")
# session.evaluate(r"inverse[x_] := Piecewise[{{Abs[Mod[InverseFunction[g][x], \[Pi]/2] - \[Pi]/4] + \[Pi]/4, x > g[\[Pi]/4]}, {Abs[Mod[InverseFunction[g][x], \[Pi]/2]], 1 <= x <= g[\[Pi]/4]}}]")
# start = time.time()
# for i in range(1,10):
#     print(i,session.evaluate(f"inverse[{i}]//N"))
# stop = time.time()
# print(session.evaluate(r"Table[inverse[i] // N, {i, 10}]"))
# print(stop-start)
# print(time.time()-stop)
# session.terminate()

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr
import time

# Start session once
session = WolframLanguageSession()

# Define g[θ] and inverse[x] only once in the Wolfram kernel
session.evaluate(wlexpr(r"""
g[θ_] := Integrate[Sqrt[1 + Tan[θ]^2 Cos[ϕ]^2]/π, {ϕ, 0, π}] // FullSimplify;

inverse[x_] := Piecewise[{
  {Abs[Mod[InverseFunction[g][x], π/2] - π/4] + π/4, x > g[π/4]},
  {Abs[Mod[InverseFunction[g][x], π/2]], 1 <= x <= g[π/4]}
}];
"""))

# Optional: compile inverse if using many evaluations
# session.evaluate(wlexpr(r"""
# compiledInverse = Compile[{{x, _Real}}, 
#   Piecewise[{
#     {Abs[Mod[InverseFunction[g][x], π/2] - π/4] + π/4, x > g[π/4]},
#     {Abs[Mod[InverseFunction[g][x], π/2]], 1 <= x <= g[π/4]}
#   }]
# ];
# """))
print("starting")
# Benchmarking: time the call from Python
start = time.time()

# Evaluate the full table in one call
results = session.evaluate(wlexpr("Table[inverse[i/1.] // N, {i, 1, 9}]"))
# For compiled version:
# results = session.evaluate(wlexpr("Table[compiledInverse[i/1.], {i, 1, 9}]"))

stop = time.time()

# Display results
print("Results:", results)
print("Wolfram evaluation time:", stop - start, "seconds")

# End session cleanly
session.terminate()