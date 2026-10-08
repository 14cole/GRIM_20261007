# Numerical methods and verification scope

The September 2026 changes preserve the outgoing-wave and amplitude-version-2
conventions. Independent cylinder and sphere references and complex-field
comparisons are necessary when changing a boundary representation: a small
linear residual alone does not establish physical accuracy.

## 2D transmission

A single dielectric body uses an exterior `D + i*k0*S` potential and an
interior single-layer potential. The exterior field projection includes both
terms. With `W = -normal_derivative(D)`, the exterior matrix blocks are
`K + M/2 + i*k0*S` and `W - i*k0*(K' - M/2)`. This removes the fictitious
Neumann resonances of the former exterior double-layer-only representation.

Closed Robin contours, including standalone PEC and IBC bodies in both
polarizations, and closed continuous regional PEC/transmission contours use side-oriented
`S + gamma*D` regional potentials, including the corresponding trace jumps,
hypersingular flux terms, and exterior field projection. Dense and compressed
coefficient queries implement the same equations. Regions with open material
junctions retain the previous representation; they are not claimed to have the
same resonance protection.

The closed-contour law may vary along the contour: PEC plus impedance zones,
several impedance zones, tapers, and separate bodies with different laws. The
boundary equation is `dn(u) + alpha(x) u` (or `u` on TM PEC nodes) for one
continuous closed-contour density. W is never weighted by `alpha`, so its Maue
form keeps no endpoint terms, and the uniqueness argument for `S + gamma*D`
(an interior impedance problem with `1/gamma`) does not involve `alpha`. The D/W
correction is therefore routed per node exactly as the base S/K' assembly:
TM PEC nodes receive `gamma*D`; every other node receives `-gamma*W` plus
`gamma*alpha*D`. The +-1/2 trace jump of a Robin row uses the element-weighted
mass `sum_e alpha_e M_e`, matching the element-weighted `alpha*S`; weighting
that jump by the nodal average alone left a second-order error proportional
to the steepness of the law (0.81% instead of 0.09% against the former equation
for a 20+5j to 377 ohm TM taper at 48 panels per wavelength).

A change of impedance flag along a closed PEC/impedance loop no longer splits
the mesh node: it is a change of boundary law, not a break in the contour, and a
hypersingular operator has no meaning across a split density. Open chains and
every other signature difference (type, materials, regions, boundary kind) keep
separate nodes. The former equation and the combined one converge to the same
fields under refinement (0.001% to 0.09% apart at 3,328 panels); at the former
discrete resonance of a PEC|75-20j ohm circle the 25 dB defect, a condition
estimate of 7e5 and a passing quality gate become a smooth response with a
condition estimate of 30. In TM every node that touches a PEC element carries
the Dirichlet row (`u = 0` on that element), in the regional layout as in the
standalone Robin assembler; the nodal coefficient is set to zero there. The
first round-5 draft classified a PEC/impedance junction node by its averaged
coefficient instead, which applies the flux equation over the PEC half of the
test function: 2.5% against 0.9% for the former equation at 24 panels per
wavelength. With the shared rule the two equations agree to 0.025% / 0.014% /
0.007% at 104 / 208 / 416 panels. At the junction the TM boundary condition
changes type (`u = 0` to a Robin law) and the density behaves like `r**(-1/2)`,
so uniform linear panels converged at first order in both equations (0.97%,
0.49%, 0.24%, 0.12% at 104 to 832 panels against a graded 3328-panel
reference). `_build_panels` now splits the panel on each side of such a
junction of an opaque conductor (TYPE 2 and TYPE 4 segment ends)
geometrically, four levels of ratio 1/2: eight extra panels per junction,
adjacent length ratios of at most 2 (inside the calibrated far-rule range). The
same input then gives 0.24%, 0.058%, 0.016%, 0.0055%, the level of the smooth
contour and of TE (0.23%, 0.057%, 0.014%, 0.0034%). Four fixed levels only
shrink the first-order term (the innermost panel is `h/16`): against a finer
reference the TM error ratio per doubling decays 4.1, 3.5, 2.4, 2.0 from 104 to
1664 panels (0.0071% and 0.0036% at the fine end). Since round 12 the mesher
adds a level for every doubling of the density beyond 32 panels per wavelength
(`_junction_grading_levels`; the innermost panel stays two decades above the
1e-9 m node snap): 0.238 / 0.060 / 0.0148 / 0.0035 / 0.0007 % at 104 to 1664
panels, ratios 3.9 to 5.1 and the TE values at the fine end, for 4 to 16 more
panels. Meshes of up to 32 panels per wavelength are unchanged.

The junction is classified from the boundary condition the field solve
evaluates, not from the flag (Codex's review of round 10: a positive flag
carrying 0 ohms was left ungraded, 0.97% instead of 0.24% at 104 panels and
fields 1.04% away from the identical flag-0 input). `_boundary_condition_junctions`
evaluates each conductor segment end (`arc_s` 0 or 1, as a taper is sampled) at
every frequency the mesh serves, with the solver's own `|Zs| <= EPS` meaning
PEC, and grades where the complex jump `|Z1 - Z2|` exceeds `1 - 1/4` of the
larger magnitude (`impedance_jump_is_graded`, `JUNCTION_GRADING_CONTRAST = 4`):
a factor of four for laws in phase, about 44 degrees for laws of equal
magnitude. PEC next to any impedance always does; two PEC
definitions meeting do not; a table that reaches zero grades the meshes of the
frequencies where it does (every frequency of a fixed reference mesh). The
contrast rule exists because exact zero is not the only case: a Robin
coefficient that is large at the panel scale acts as `u = 0` until the panels
resolve `1/|alpha|`. Uniform against graded error at 104 / 208 / 416 panels,
law | `75-20j` on the same circle:

| Law, polarization | Uniform | Graded |
|---|---|---|
| 0 ohm law, TM (as flag 0) | 0.97 / 0.48 / 0.24 % | 0.24 / 0.058 / 0.015 % |
| 1 ohm, TM | 0.36 / 0.17 / 0.089 % | 0.24 / 0.056 / 0.013 % |
| 5 ohm, TM | 0.29 / 0.13 / 0.061 % | 0.25 / 0.063 / 0.015 % |
| 20 ohm, TM (contrast 3.9) | 0.20 / 0.057 / 0.015 % | 0.26 / 0.065 / 0.016 % |
| 1e5 ohm, TE | 1.98 / 1.07 / 0.57 % | 0.13 / 0.031 / 0.010 % |
| 3770 ohm, TE | 1.04 / 0.32 / 0.087 % | 0.20 / 0.046 / 0.010 % |
| 200 ohm, TE (contrast 2.6) | 0.24 / 0.059 / 0.014 % | 0.23 / 0.058 / 0.014 % |

Against a `377` ohm partner, `20` ohm (contrast 19) loses 1.17 / 0.42 / 0.13 %
to 0.27 / 0.065 / 0.015 % in TM, `50` ohm (7.5) 0.53 / 0.16 / 0.043 %, and
`100` or `1500` ohm (3.8, 4.0) are within 1.5 times the graded error in both
polarizations. Below a contrast of four the uniform mesh is second order, so
the steps of a staircase that approximates a taper are not graded. Phase counts
since round 12, because a phase-only contrast costs a constant factor although
not the order: laws of magnitude 77.6 ohms against `75-20j`, TM, uniform over
graded error at 416 panels, 1.2 at 30 degrees apart, 1.3 at 45, 1.5 at 60, 1.8
at 75 and 2.4 at 105 (`+77.6j`: 0.59 / 0.15 / 0.038 % against 0.26 / 0.065 /
0.016 %); TE is indifferent there and mirrors it against a 377 ohm partner.
Combined contrasts (a factor of three with 45 degrees) gain up to 1.9 and are
graded too. The measure is a threshold on one number, not a model of the
singularity: at equal jump, `25.9` ohms at +30 degrees gains nothing and at -60
degrees 1.9. Above the threshold
grading never lost more than the 0.06 percentage points it costs at 104 panels
(it breaks the uniform circle's error cancellation). Both polarizations share
the graded mesh, so their systems stay interchangeable. Solves, backend
selection and HPC scheduling pass the same material library and frequencies to
`_build_panels` and count the same panels; a caller without a library has only
inline laws evaluated, and a table is then graded only against PEC. The rule is
a threshold: a table crossing it between two frequencies of a sweep changes
that junction's mesh by eight panels there. Free strip ends, triple junctions
and TYPE 1 sheet joints are not graded. The real HPC driver was run end to end
on such bodies (plan without submission, then the worker in an interpreter
where importing Qt raises): for a rectangle with two PEC/impedance junctions (24 and
36 panels per side at 1 GHz, 180 to 540 per wavelength, so seven to nine levels)
the schedule forecasts 126 base and 176 certification nodes, the worker solves
and attests 176 panels, and a zero-ohm law gives 96 and 144 in both (the
certification mesh gives each chain `max(B + 1, ceil(1.5 B))` panels, longest
panels first, never fewer than the base on any primitive).
The BoR mesher applies the same rule to the chains of a conductor (below).
Monostatic, bistatic and boundary-density entry points use this same selection.
The discrete 104-panel TE circle resonance at `ka=5.13718385499639` is a
regression case; merely checking the nearby smooth-circle eigenvalue misses
the former narrow error band.

Production regional assembly evaluates S, K', D and W in one traversal per
wavenumber and scatters directly into the system matrix or compressed tile.
The fused W contribution uses a calibrated far rule of 8 points for degree 1
and 9 for degrees 2 and 3 (10 before the 22 September audit) for passive
wavenumbers with `1e-4 <= |k| Lmax <= 3`, panel length ratio at least 0.001,
separation at least three maximum panel lengths; the worst W block error is
1.5e-14 to 4.6e-14. Outside that range it retains the 16-point floor; explicit
higher orders and `far_grading=False` retain the conservative rule. Calibration
compares the completed W block, including cancellation, with independent
40/48-point rules. The graded far orders of S, K' and D depend on the
polynomial degree, with a separate table for strongly lossy wavenumbers
(`arg k < -45 degrees`); every bin is within 1e-12 (S relative to the block
maximum, K' and D to the integral of the kernel magnitude). A panel end that
hovers within half a panel length over another panel's interior uses the
adaptive near rule (worst K' 1.3e-8 to 7.4e-12). Self blocks with `|k l| > 8`
are built from sub-intervals (1.5e-14; the generic fallback was 4.4e-2 at
`|k l| = 8.1`). Kernel tables accept a check tolerance of
`max(2e-13, 8 eps |k| r)` and stop at their 4096-interval budget (about 377
wavelengths), beyond which the far expansion applies; they used to be refused
beyond about 163 wavelengths.
Near pairs reuse the same checked Green and derivative integrals.
The separate operator implementation remains an independent assembly reference.
This removes repeated 64-row correction queries without reducing quadrature
accuracy. Resource estimates include the additional routes and basis moments.

The combined equation needs S, K', D and W where the former equations needed a
single operator, and a conductor's TM system cannot be derived from its TE
system (TE uses K' and W, TM uses S and D), so the co-polarized solve of a
conductor had paid two full traversals. The regional layout takes nothing
polarization-specific from the element records, and the fused engine applies
masks and routes to finished tile accumulators, so the TE step of a
co-polarized dense solve now assembles both systems in one traversal
(`assemble_pair`) wherever TM would otherwise be assembled fresh (combined
systems whose conductor rows exceed one eighth of the unknowns). The TM step
finds its finished matrix as the retained one. The second matrix is resident
during the TE solve, so the pairing is admitted only when it fits on top of that
solve's own estimate; otherwise, and for single-polarization calls, bistatic
entries and the compressed backend, nothing changes. Both matrices equal the
separately assembled ones to rounding (1e-13 relative), and the retained matrix
no longer needs restoring from the residual spool. Measured on the whole
co-polarized solve at 4,320 panels (`ka = 60`): PEC 33.8 to 27.7 s, uniform
impedance 50.7 to 31.7 s, PEC|impedance 50.4 to 31.7 s; a coated PEC core
(1,440 panels, 23 % conductor rows) 24.0 to 15.4 s; a dielectric cylinder,
whose TM system is still derived from the TE one, 6.0 s both ways. A conductor
gains less than an impedance body because each of its polarizations already
needs both Hankel functions and only two of the four operators. The combined
equation therefore remains more expensive than the unprotected one it replaced;
what is left is the four-operator contraction and W's far rule.

Plain dielectric assembly shares the exterior S/K'/D/W traversal. Combined
regional TE-to-TM reuse retains trace rows and rescales the appropriate flux
blocks, while independently rebuilding conductor/Robin equations through
bounded row-strip queries. Those queries cost several times more per row than
the fused assembly, so reuse is attempted only while conductor rows are at most
one eighth of the unknowns (measured break-even between 9% and 20% on coated
cylinders); otherwise TM is assembled fresh into the same owned buffer. A
standalone PEC/IBC body has nothing to reuse: routing it through the row-strip
path made the co-polarized solve slower than its two polarizations run
separately (4,320 nodes: PEC 22.6 s against 12.1 s, 75-20j ohm IBC 50.9 s
against 17.7 s, identical fields). The combined equation itself still costs
about twice the former single-operator assembly for such bodies. Unequal
observer/source rules use the original assembly path. These optimizations
preserve the previously introduced combined-potential formulation.

The outer tile scheduler owns 2D assembly concurrency. Native far integration
does not start a nested four-thread pool. Endpoint-touching pairs (one
1e-9 m touching tolerance, the node-snap width, plus a same-node rule) use a
graded `t^4` corner rule with 20 points per direction and corner-stable
distances: per block 2.6e-13 collinear and 1.3e-10 (K') at a 170 degree reflex
corner, where the former Duffy batches erred by up to 6.3e-4; on a circle the
quadrature error was O(h) (5e-5 to 2.4e-6) and total convergence is now clean
O(h^2). Far tiles are computed on threads but scattered in a fixed tile order,
so thread counts give bitwise identical matrices, and weighted tile
preparation occurs outside the matrix scatter lock.

Dense double-precision LU can spool an exclusively owned original matrix to
temporary disk storage and overwrite its RAM buffer with LU. Every residual
product still uses every original row; condition scaling is computed before
overwrite. Original TE coefficients needed for TM assembly are restored from
the spool into the same buffer after solving. `dense_residual_storage` accepts `auto`, `memory`, or `disk` in the
2D execution options. The admission estimates price the original next to its
LU, so `auto` keeps it in memory whenever that copy fits: it spools an eligible
matrix of at least 512 MiB only when the memory available as the matrix is
factored cannot hold the copy (plus an eighth, at least 256 MiB), and falls
back to memory if the spool cannot be created. Every residual product of a
spooled matrix re-reads the whole file (on the certified 10 GHz airfoil, 64 s
of 207 s for a 7.8 GB system; kept in memory the run took 145 s within its
16.7 GB estimate). Non-owned arrays, mixed precision, and hierarchical factors
retain their existing storage. Spools close on completion or failure. BoR modal
factors follow the same policy: a mode worker whose copy does not fit factors a
system of 512 MiB or more in place and computes its residuals from the spool.

A dense double-precision system of at least 10,000 unknowns
(`linalg.hierarchical.HIERARCHICAL_MIN_UNKNOWNS`; the environment variable
`GHOST_HIERARCHICAL_MIN_UNKNOWNS` overrides it, 0 keeps LU) is factored as a
HODLR inverse under the `dense` and `auto` factorizations. Its off-diagonal
blocks are compressed by an adaptive randomized range finder (products of the
block with 32 Gaussian probes at a time, re-orthogonalized, until fresh
samples leave less than `1e-10 ||A||_inf`; then a truncated SVD through the QR
of the projected block), and the factor is accepted exactly as before: every
solve is refined against the original matrix to a normwise backward error of
3e-15, the result must then pass the dense backward-error gate, and a factor
that does not converge is rebuilt once at `1e-12`, then replaced by LU. On the
certified airfoil's systems (three batches of 256 right-hand sides, eight
cores) the factor tied LU at 9,082 unknowns (4.2 against 4.7 s) and was 1.7
times faster at 13,618 (8.0 against 13.7 s, factor 182 MB against a 2,967 MB
LU); below the threshold LU is faster (7,348 unknowns: 3.4 against 2.8 s).
The block tolerance sets the cost: at `1e-9` refinement needed two steps, at
`1e-6` seven, each a product with the original matrix per batch. A cluster
tree over the coordinates keeps coincident unknowns of different densities in
the same leaf; the natural order would split them at the top level, where one
block then reached rank 1,024. Memory forecasts price such a system as the
matrix and its factor budget (0.65 of it) instead of the matrix and an LU
copy; an LU fallback that finds no room for its copy spools the original
(`dense` and `auto`). End to end, the certified airfoil at 6 GHz takes 40.5 s
instead of 51.7 s with LU (factorizations 12.9 against 25.3 s, peak 5.1
against 7.6 GB, RCS equal to 1.7e-12 of the largest amplitude); at 10 GHz,
where the LU copy did not fit the planner's margin on a 31 GB workstation and
the compressed backend ran (90-92 s, 4.3 GB), the dense path is now admitted
and takes 88 s (11.5 GB).

## BoR formulations and geometry

Closed uniform IBC dispatch selects CFIE for reactive and resistive impedance.
The direct `solve_bor` API also defaults to automatic selection: CFIE for
closed opaque bodies, EFIE for open shells and transmitting sheets.
Physical resistance does not remove EFIE representation resonances. Closed
bodies with a spatially varying impedance (a PEC zone next to an impedance
band, several impedance zones, tapers) use the same CFIE. Its magnetic-field
part contains the EFIE operator acting on `M = -Zs n x J`, whose charge term
needs a continuous source: a jump of `Zs` between two elements would be a
magnetic line charge that the Gauss-point tables cannot represent. `M` is
therefore expanded in the nodal basis with the average impedance of the two
adjacent elements at each node (the convention the partial-coating bare pieces
already use), which spreads a jump over two elements and reduces to the uniform
law exactly (a uniform array reproduces the scalar result bit for bit). The
electric-field part and the far field keep the element-wise `Zs`.

No exact reference exists for a varying law, so the CFIE was compared with the
EFIE away from its resonances on a 0.1 m sphere at `ka = 3.3`: a smooth taper
agrees to 0.0009 dB at 240 elements and 0.0065 dB at 60; a PEC half next to
`100+50j` to 0.015 dB (0.09 dB at 60); two zones `50 | 300+100j` to 0.048 dB
(0.42 dB at 60). At an abrupt jump both equations converge at first order on
uniform elements and the CFIE needs about 1.5 times the EFIE's density. The
dispatcher therefore grades the generatrix of a conductor toward every chain
junction where the evaluated impedance jumps, with the 2D rule
(`impedance_jump_is_graded`, the law at the chain end at that frequency) and
the 2D levels (four, ratio 1/2: eight more elements per junction).
`_mark_impedance_junctions` marks the chain ends; one breakpoint function
(`_primitive_breaks`) serves the mesher and the element count, so preview,
near-pair count and solve see the same mesh, and each frequency record carries
`graded_impedance_junctions`. CFIE, `ka = 3.3`, against graded 240-element
references, 30 / 60 / 120 elements: PEC half | `100+50j` 0.20 / 0.11 / 0.055 dB
uniform, 0.058 / 0.015 / 0.004 dB graded (three levels 0.072 / 0.021 / 0.007,
five 0.050 / 0.011 / 0.002); zones `50 | 300+100j` 0.97 / 0.48 / 0.23 dB
uniform and 0.17 / 0.066 / 0.024 dB with three levels; the EFIE behaves alike
(0.15 / 0.074 / 0.036 to 0.057 / 0.015 / 0.004 dB). This had been refused by
the near angular rule, whose failure was a rounding floor (see Certification).
Only the conductor kind is graded: bare pieces of partial and banded coatings,
material junctions and sheets are not. The BoR HPC driver was run end to end on
a graded sphere (plan, then the headless worker): 32 elements, one graded
junction, IBC-CFIE, and 24 with a zero-ohm law. Around the first cavity resonance
(`ka = 2.70..2.80`, 60 elements) the EFIE departs from its neighbours by 1.5 dB
(PEC half) and 16.3 dB (taper); the CFIE stays within 0.0012 dB. Explicit
`formulation='efie'` remains a diagnostic option with no resonance guarantee.

Full PEC cores use a combined electric/magnetic boundary equation in the
coating medium, including corresponding cross-surface terms. In the direct
single-coating notation, the core row is
`[Tco + etaL*R*Pco, -eta0*Pco + eta0/etaL*R*Tco, -Tcc - etaL*Kcc]`,
where `R` rotates tangential test rows and `Kcc` includes the MFIE jump.
The full multilayer path uses the same core equation. Partial core surfaces
meeting material junctions retain their existing junction formulation.

Preview, certified dispatch and direct full-coating APIs require every inner
interface/core to lie strictly inside its enclosing surface, without crossing
or touching it. Same-surface separated pairs route to checked near integration
when the gap is below twice the larger panel length, as well as the existing
angular-resolution threshold. Preview memory includes these geometric pair
counts; a fixed index stencil alone cannot price tight grooves accurately.

## TE sheets

Adjacent impedance flags and virtual sheet regions assigned to segment names
no longer split a TE sheet trace. Each nonbranching sheet is oriented before
nodes are shared and before polynomial enrichment. Real material regions remain
distinct, and TM interface splitting is unchanged.

A TE sheet attached to a PEC body requires a coupled junction condition that
is not implemented. Such a joint now raises a specific error instead of
pinning both coincident nodes and returning a misleading certified field.
Model a physical fin as a closed, finite-thickness PEC contour. Branching TE
sheets are also rejected. Separated sheets and PEC bodies remain supported.
The desktop co-polarized workflow requires both VV (TE) and HH (TM), so the
attached-sheet guard prevents that entire run, even though a standalone TM
solve is supported. The solver selection tooltip explains this limitation.

## Certification

Mesh refinement checks discretization changes; it cannot detect an equation
that converges to the wrong model. Quality gates check the solved linear system
and mesh evidence, not general correctness of every supported physical model.
The default certification tolerances have not been tightened as a substitute
for the formulation fixes. Informational mesh-cache/junction notices are
separate from warnings counted by the quality gate.

A reference mesh frequency can only maintain or refine the discretization:
the single-frequency path now includes both the solve and reference frequencies
when choosing the shortest material wavelength.

The near ANGULAR rule (sinh core plus Gauss-Legendre tail, orders raised until
two results agree to 2e-8) used to stop with "did not converge at the maximum
order" on very small elements, which is what had made BoR mesh grading
impossible. The cause was not the order: the sampled MFIE/IBC brackets form
`rho_p - rho_q*cos(xi)` and products such as `tr_p*tz_q - tz_p*tr_q*cos(xi)`,
and for two points `d` apart on one element those differences are of order
`rho*xi**2` and `d` while their operands are of order `rho`. Rounding then
leaves 2e-9 of the kernel at `d = 1e-5 rho`, 1e-8 at `5e-6 rho` and 3e-7 at
`1e-6 rho`, above the tolerance, in the native sampler and in NumPy alike.
`kernels._stable_brackets` writes every such term with `h = 2 sin(xi/2)**2`
(the `h**2` parts cancel analytically; `R**2 = d**2 + 2 rho_p rho_q h`), agrees
with the sampled forms to 1e-13 where those are accurate, and stays within 4e-10
down to `d = 1e-6 rho`. At 20 elements per wavelength the self and adjacent
terms of an equatorial element still converge at `ka = 400`; elements of `h/64`
now solve.

Since the 22 September audit the near rule is graded (a sinh core, ratio-2
geometric panels up to `s = 0.25` and one tail panel, each sized from its own
phase) and runs natively (`near_green_rule`, `near_brackets_rule`); the native
bracket rule evaluates every point with the stable forms at no extra cost, and
the NumPy fallback switches to them below `d = 1e-4 max(rho)`. A coarse/fine
comparison is accepted only when every order actually grew: the former check
could accept two identical saturated tail orders (4096) and returned kernels
wrong by up to 1.1e-3 at `d/a = 1e-8` without an error (526 of 920 points of a
tiny-element self cell above tolerance, worst 2.3e-2). The kernels now agree
with independent references to 4e-14 for `d/a` from 1e-9 to 0.5.

BoR self and adjacent meridian quadrature uses a fixed graded rule (quadtree
cells toward the singular set, 4 x 5 Gauss points per cell, depth `near_depth`).
Its self-term error was measured in the 22 September audit: the rule misses the
model integral of `log|s - s'|` by 8e-3 and leaves about 6e-4 relative
far-field error at 27 elements per wavelength, decreasing as O(h). A rule
graded in `|s - s'|` only (each triangle mapped to `u = |s - s'|` and the
along-diagonal coordinate, geometric intervals in `u`) removes that error (5e-6
on the model integral, 2-3e-6 in the far field) with a third of the points,
but against exact series for PEC (CFIE and EFIE, `ka` 1 to 10), impedance and
dielectric spheres at 10 to 80 elements per wavelength its end-to-end error was
not smaller at practical densities, and mostly slightly larger (ka = 3 CFIE:
0.0248 against 0.0199 dB at 20 elements per wavelength): the cell rule's
integration error partly offsets the flat-segment geometry error. It also cost
about 10% more, so the cell rule was kept.

Far element pairs (at least two element lengths apart), the plane-wave
excitation and the far-field projection use `kernels.FAR_GAUSS_ORDER = 3`
Gauss-Legendre points per element since 24 September (four before; the
`gauss_order` argument still selects any order). Against four points the
amplitudes moved 2e-9 to 2e-8 relative on PEC (CFIE and EFIE), impedance
(`75 - 20j`), lossy and lossless dielectric (`4 - 0.1j`, `2.56`) and coated
spheres at `ka = 5` and 20 elements per (material) wavelength, and the errors
against the exact series were unchanged to 1e-4 dB (0.005 to 0.02 dB); the
4 GHz ogive moved 2e-7 against a 9e-3 discretization error. Far tables shrink
by (3/4)^2 and the far build by about a third. Mirror-symmetric bodies stay
symmetric to 3e-8 (4e-10 with four points): element pairs on the near/far
routing boundary can be routed differently on the two sides by rounding, and
the far rule's error on those pairs is what shows.

Element pairs of two different surfaces that touch at a junction point have
their own rule (`JUNCTION_CELL_ORDER = 10`, `JUNCTION_CELL_DEPTH = 7`). The
order-4, depth-4 cells they used before left 1.2e-3 relative far-field error at
partial, layered-patch and banded junctions against an order-12, depth-8
reference; the refined rule is within 2e-5 and costs nothing measurable (a
junction has a few such pairs). A vacuum coating, which must be invisible,
reproduces the exact PEC sphere as well as or better than before (ka = 3, 24
elements per hemisphere: 0.0095 dB, the plain PEC mesh's error; ka = 1:
unchanged). Reverse cross-surface operators are derived from the forward ones
by reciprocity, which is exact only when both directions use the same rule;
comparing the derived and independently integrated reverse operators is how
the junction error was found.

Included meridian corners below 15 degrees are rejected before solving; a
corner drawn at the threshold passes (1e-6 degree rounding tolerance).
Subdividing straight panels does not change this angle. This guard does not
establish convergence for every corner at or above the threshold. The same
limit applies to every stitched surface of partial, layered and banded layouts,
including the whole PEC core of a partial coating. The included angle BETWEEN
two surfaces at a junction (a feathered coating termination is a dielectric
wedge) is not limited. The repository's own partial-coating fixture ends in a
12.5 degree wedge whose regressions pass; refined junction rules now run there
(see above), but no evidence-based wedge threshold has been established yet. On-axis tips are not
guarded: production and refined rules differ by at most 0.012 dB down to a 10
degree included tip angle (0.10 dB at 5 degrees), against 0.28 dB for a 10
degree rim.
Sphere benchmarks support its performance on those meshes, but do not replace
an independent near-integral convergence test. That limitation remains open.

## BoR material interfaces: weighted region equations

Every BoR material interface combines the equations of its two regions. PMCHWT
adds them with equal weights. The continuous PMCHWT equations have no real
resonance, but on a lossless interface the coarse discrete sum had spurious
singularities at isolated real frequencies: a 20-element
`eps_r = 3` sphere of radius 0.1 m (10 elements per medium wavelength) reached
a condition estimate of 2e8 at 0.95127 and 1.09777 GHz, with +6.7 dB axial and
10 dB off-axis error, a 3e-13 residual and a band about 1e-4 wide; the same
mesh cost a coated sphere 8.4 dB. The near-null vector is a `phi`-directed
node-to-node alternation at the two poles in `|m| = 1`. Its charge,
`j*m*J_phi/rho`, involves no derivative, so its discrete charge is as small as
a smooth current's; the exterior operator is then net capacitive on it and the
interior one net inductive (`-4.815e-3j` and `+4.823e-3j`), and equal weights
let them cancel. Using the exact axis relation (`t_rho` instead of its sign) or
refining the self/adjacent quadrature only moves the crossings by about 0.1%.
That rules those two repairs out as cures; it does not show that no other
discretization of the pole would remove the mode. The artifact weakens under
refinement (largest condition estimate 2e8 at 20 elements, 4e6 at 40), as a
property of the discretization should. What follows is therefore a numerical
stabilization of the discrete system, not a correction of the formulation.

The regions' equations are therefore added with unit-modulus weights
`c_R = exp(-j*15 deg * rank_R)`, regions ranked by `Re(eps_r*mu_r)` and the
exterior normalized to 1 (`_region_equation_weights`). Mautz and Harrington's
uniqueness condition constrains the coefficients `alpha` and `beta` that
multiply ONE region's electric and magnetic equations: `alpha*conj(beta)` real
and positive (PMCHWT has `alpha = beta = 1`). Both equations of a region share
its weight here, so that product is `|c_R|**2 = 1` whatever the phase; the
condition holds, and the phase between regions is free to make the reactive
cancellation impossible. (An earlier version of this section stated the
condition for the product of two regions' weights, which a relative phase would
violate; Codex's review corrected it.) Unequal weights no longer cancel the identity terms, so each
interface row receives `+sigma_R*c_R/2 * <W, n x M>` (electric) and
`-sigma_R*c_R/2 * <W, n x J>` (magnetic). The denser region takes the negative
phase because loss rotates its term the same way: with the opposite sign a
lossy medium can restore the cancellation (condition 1e5 at `2-2j`, +30 deg),
with this one loss only improves it. Regions of equal `Re(eps_r*mu_r)` keep
equal weights; their reactive parts have the same sign.

Against the exact series over 0.8 to 1.6 GHz (19 frequencies, 5 aspects, both
polarizations; maximum / median error in dB and largest condition estimate):

| Body | PMCHWT | Weighted |
|---|---|---|
| dielectric 3, 20 elements | 10.24 / 0.077 / 2e8 | 0.17 / 0.064 / 3e4 |
| dielectric 3, 40 elements | 0.045 / 0.016 / 4e6 | 0.043 / 0.015 / 8e4 |
| dielectric 3-0.3j, 40 | 0.186 / 0.030 / 1e5 | 0.173 / 0.031 / 6e4 |
| dielectric 10, 40 | 0.470 / 0.072 / 2e7 | 0.472 / 0.077 / 7e4 |
| dielectric 0.5, 40 | 0.023 / 0.006 / 4e7 | 0.023 / 0.006 / 2e5 |
| coated 3, 20 | 8.44 / 0.185 / 3e8 | 0.37 / 0.161 / 3e4 |
| coated 3, 40 | 0.083 / 0.034 / 4e6 | 0.085 / 0.035 / 9e4 |
| two layers 4 over 2, 40 | 0.079 / 0.038 / 4e6 | 0.077 / 0.039 / 2e5 |
| two layers 2 over 4, 40 | 0.093 / 0.033 / 4e6 | 0.097 / 0.034 / 2e5 |

The condition column is the largest estimate over each sweep: the weights
bound the worst case, they do not lower the estimate at every frequency.
Frequency by frequency on 20-element spheres (`eps_r` from 0.5 to 10, five
frequencies each, 45 cases) the weighted estimate is lower in 31 cases and
higher in 14, all but one of those at `|eps_r - 1| <= 0.1` and by up to 11
times (`eps_r = 0.99` at 0.7 GHz: 3.9e4 to 4.3e5, found by Codex), with the
error unchanged there (0.282 to 0.277 dB). The largest weighted estimate of
that survey is 4.3e5 against 2.3e8.

The identity terms give the rows a second-kind part, which costs a few
thousandths of a dB in the median where PMCHWT was healthy (15 degrees is the
smallest angle tried; 90 degrees doubles that cost). The same weights apply to
the dielectric, coated, partial-coating and multi-region solvers and to the
dense, streamed and compressed paths. A coarse mesh still has coarse-mesh
accuracy (0.17 dB at 10 elements per wavelength); the default 20 per medium
wavelength is unchanged.

Neither form resolves a vanishing contrast. The scattered field is then a small
difference of order-one equivalent currents, and the relative error grows like
`1/|eps_r*mu_r - 1|`: on the 20-element sphere at 0.7 GHz, 0.04 dB at
`eps_r = 1.1`, 0.28 dB at 1.01 and 12 dB at 1.0001 (exact RCS 8.6e-11 m^2;
10.3 and 8.5 dB at 40 and 80 elements, 9.6 and 8.0 dB weighted). The sphere
comparisons above do not establish accuracy in that regime.
Codex's review added magnetic and lossy media (`mu_r` up to 3, `3-3j | 1-1j`:
0.006 to 0.056 dB at 48 elements) and three-layer coatings (0.005 and 0.008 dB
at 40 elements per surface) to the evidence for the weighted form.

## October 2 solver review changes

Fixed-reference 2-D sweeps now retain the complete request's sizing-frequency
set across frequency-local execution and checkpoint resume. Both polarizations
and base/fine certification phases retain their own mesh topology; matrices
and factors still live for one frequency. Low references and dispersive
materials can therefore produce a finer mesh than earlier canonical sweeps.

Run forecasts use digest keys and a 16 MiB byte budget. Reused checkpoints
verify their digest/header before solving and decode their sample arrays once
at final merge. Failed checkpoint writes preserve computed results within a
bounded allowance; if further work cannot fit, the GUI marks the result
partial and does not automatically export it. BoR desktop sweeps now use
frequency checkpoints with BoR source, material and option identities.

LU fallback checks remaining host and solve-reservation memory before copying
the original matrix. Failed factor construction tracebacks are released before
replacement. Batched compressed GMRES retires completed columns independently.
Original-matrix residual gates remain unchanged.

BoR `quadrature_check='refine'` is an optional same-mesh comparison, exposed as
"Compare refined integration" in the desktop BoR options. It increases the
self/adjacent and cross-surface junction grading depth by one, compares each
frequency/channel's complex fields at requested angles, and returns the
refined result only when maximum and RMS normalized changes are at most
0.002 and 0.001. It adds a second solve and is off by default. This checks
integration sensitivity, not continuous angular interpolation or the accuracy
of the drawn piecewise-linear shape. Refined integration need not reduce total
error when integration and faceting errors previously offset each other.

Far-block ACA now checks spread rows and up to three random unused rows before
accepting its stopping estimate. A failed probe becomes a new pivot. This is
stronger sampled evidence, not a deterministic full-block error certificate.

Backend timing reuse retains the conservative initial cost prior. Repeated
paired measurements may also inform a nearby request on the same geometry,
materials, source, host and execution settings, subject to tight frequency,
DOF and angular-grid bounds, stable timing samples and an uncertainty margin.
Normal solves do not perform speculative extra backend solves for calibration.

Earlier component benchmarks were measured separately from complete solves;
their speedups are not claims of the same improvement in complete solve time.
The published [measurement summary](../../experiments/solver_review_20261002/results.json)
records the later airfoil and subtraction comparisons, their scope and the
status of research prototypes. Local baseline copies and raw benchmark logs
are not part of the published repository.

The subsequent 2-D sweep improvements retain admission-sampled compressed
tiles within a 32 MiB run cache. Reuse requires identical geometry, evaluated
materials, polarization, quadrature settings and row/column DOF lists. Cached
tiles enter the normal assembly queue so error-bound sums keep their original
order. A completed polarization partner supplies an exact payload size instead
of being sampled again. Run-owned spawned assembly workers refresh their
coefficient data between operators and close on completion or cancellation;
idle worker allowances remain in the memory forecast during factorization.

Desktop checkpointed sweeps use up to two frequency workers when CPU, RAM and
predicted work permit, with nested assembly pools disabled. Expensive units
run first and final samples retain request order. CPU reservations and worker
interpreter memory are included in admission. This changes scheduling, not
mesh certification or the equations solved.

The stage cost model separates near/far assembly, compression, factorization
and RHS work. Nearby calibration requires the same geometry, materials,
source, host, options and factorization regime, compatible observed meshes,
at least three stable samples for each backend and an uncertainty margin.
Bounds are 15% in frequency, 20% in DOFs and angle count, and 50% in stage work.
Fallback or repeated-factorization runs do not train the stage model.

`compressed_far_method='verified_cur'` is experimental and off by default.
It proposes factors from selected exact rows and columns only for separated
support boxes. Every coefficient is still evaluated to validate the proposal;
its complete error enters the existing row/column error bounds. Rejected
proposals fall back to ordinary QR compression. This is not a sampled-only
error certificate or an asymptotically faster production assembly path.

`frequency_preconditioner='reuse'` is also experimental and off by default.
The run-local cache is limited to the smaller of 128 MiB and 5% of the solve
RAM budget and stores inverse data only. A compatible mesh, DOF ordering,
polarization and formulation are required, with frequency ratio at most 1.10.
The operator is always assembled at the requested frequency. A reused inverse
gets two correction steps and up to eight GMRES iterations before rebuilding;
the original-coefficient backward-error gate is unchanged. Mesh-changing
adaptive sweeps generally cannot use this cache.

Native fixed-width far-kernel loops preserve the arithmetic order. The earlier
frequency-scheduling comparison is included in the published measurement
summary. Native-only component measurements and rejected local experiments
do not establish complete-solve improvements.

On the measured Windows host, certified `airfoil.geo` sweeps at 1, 1.5, 2,
2.5 and 3 GHz with 181 angles and the same four-CPU/12-GiB budget averaged
40.765 seconds sequentially and 32.845 seconds with two frequency workers
(two runs each, 19.4% less elapsed time). Meshes matched and the largest
complex-field change, normalized by each channel's peak, was 4.92e-13.
One certified 10 GHz before/after pair measured 94.279 and 91.296 seconds
(3.2% less time); memory sampling fell from 3.668 to 2.017 seconds and the
largest normalized field change was 3.96e-14. That pair used the same optimized
native binary on both sides to isolate the Python/backend changes. The 3 GHz
dense solve's internal time was essentially unchanged. Parent-process peak
working set was essentially unchanged; simultaneous worker-inclusive peak RAM
was unavailable, so these measurements establish no RAM saving. The complete
1-18 GHz sweep has not been timed, and component or small-sweep gains should
not be extrapolated to it.

The optional inverse-reuse experiment did not improve the measured cases.
A fixed-mesh coated case at 1 and 1.01 GHz (384 panels, 181 angles) avoided two
factorizations but required extra GMRES work: 4.244 seconds without reuse and
4.692 seconds with reuse, with 16.1 MiB more peak working set. Its largest
peak-normalized field change was 3.14e-12 and physical backward error remained
below 2.86e-15. The fixed-mesh airfoil case at 3-3.1 GHz produced inverses
larger than the 128 MiB cache; the redundant reuse run was stopped and is not a
timing comparison. This evidence supports keeping the option off by default.

The latest production changes widen the sampled column-space proposals used
by ordinary QR tile compression and trim their projected rank before storage.
Every accepted proposal is checked against the complete original tile at the
existing tolerance; its full coefficient error still enters the operator's
error bounds. Rejected proposals fall back to the existing compression path.
This enabled optimization is separate from the optional `verified_cur` path.
Regional coefficient assembly also prepares shared index maps once per
operator and reuses selections within each tile query, avoiding redundant
whole-mesh mapping work. It does not reuse meshes or coefficients across
frequencies or change material routing, geometry or certification thresholds.

Fresh-process certified airfoil comparisons used 181 angles, both
polarizations, four CPU threads and a 12 GiB budget. Relative to the already
optimized backend, the 3 GHz median changed from 15.31 to 15.21 seconds
(essentially unchanged), the 10 GHz median from 92.12 to 79.37 seconds
(13.8% lower), and the 18 GHz measurement from 236.75 to 196.12 seconds
(17.2% lower). The 3 and 10 GHz figures use two runs per version; 18 GHz uses
one paired measurement. Assembly time fell about 23-29% at 10 and 18 GHz,
while sampled aggregate process-tree peak working set fell about 1.5%.
Summed working sets may double-count shared pages; this is a sampled process
metric, not a measurement of unique physical RAM consumption.
All certification checks passed, and the largest whole-airfoil complex-field
change was below 1.3e-13 of the corresponding pattern peak. These are separate
single-frequency comparisons; their gains must not be added to earlier
measurements or extrapolated to a complete frequency sweep.

A one-time coherent-subtraction comparison found quadratic/cubic changes
below 0.066% of the feature-response peak for standard PEC grooves at 3, 10
and 15 GHz, and 0.683% for a synthetic 0.01-inch groove at 10 GHz. A finer
cubic mesh changed the latter response by 0.236%. After the production
optimizations, that narrow-gap subtraction was unchanged in VV and changed
by at most 8.04e-11 of its feature peak in HH. These are observed comparisons,
not error bounds for other geometries or individual deep nulls. The standard
coupons have frequency-designed remote backing, so their three frequencies
do not constitute a sweep of one fixed coupon. The published
[results](../../experiments/solver_review_20261002/results.json) and
[subtraction plot](../../experiments/solver_review_20261002/subtraction_comparison.png)
record this evidence; the production subtraction workflow is unchanged.

Same-frequency quadratic/cubic assembly reuse remains a small dense research
prototype: its setup-inclusive comparison measured a 27.1% reduction, but
does not establish a gain for the full compressed, certified airfoil solver.
Directional far-kernel interpolation is also a prototype, compared with
direct SciPy kernels rather than GHOST's native assembled operator.
Higher-order BoR remains a design proposal without an implementation or
measured saving. None of these three research directions is enabled in the
production solver.

## Running tests

Install the `test` and (for desktop coverage) `gui` extras, then run
`python ghost_backend/tests/run_suite.py`. This runs the headless suite together
and Qt-related modules in separate interpreters, continues after a failed group,
and exits nonzero if any group fails. This separates Qt module lifetimes from
numerical workers and completed successfully on Windows. The root cause of the
original order-dependent native abort has not been proven.

The new audit regressions cover the exact narrow cylinder-resonance inputs,
sheet segmentation/orientation, compressed/dense coefficients, conservative
mesh references, BOR storage and planning, mode retries, and grouped FFT
coefficients and complex fields.
