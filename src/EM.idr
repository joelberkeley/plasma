module EM

import Tensor
import Literal

||| The trajectory of `n` charged particles through a uniform electromagnetic field.
|||
||| @ex The x-component of the electric field.
||| @ey The y-component of the electric field.
||| @bz The z-component of the magnetic field.
||| @r0 The starting coordinate of each particle.
||| @v0 The starting velocity of each particle.
||| @q The charge, in units of e, of each particle.
||| @m The mass, in eV, of each particle.
||| @interval The time interval between steps.
||| @steps The number of time steps.
export
uniformEM :
  (ex, ez, bz : Tensor [] F64) ->
  {n : _} ->
  (r0 : Tensor [n, 3] F64) ->
  (v0 : Tensor [n, 3] F64) ->
  (q, m : Tensor [] F64) ->
  (interval : Tensor [] F64) ->
  (steps : Nat) ->
  Tag $ Tensor [n, steps, 3] F64
uniformEM ex ez bz r0 v0 q m dt steps = do
  let v0x = slice [all, 0.to 1] v0
      v0y = slice [all, 1.to 2] v0
      v0orth = broadcast {to = [1, n, steps]} $ sqrt (v0x ^ fill 2.0 + v0y ^ fill 2.0)
      w = abs q * bz / m
      sign = select (q > 0.0) 1.0 (-1.0)
  t <- tag $ dt * iota {shape = [1, n, steps]} 2
  let x : Tensor [1, n, steps] F64 = reshape $ v0orth / w * sin (w * t)
      y : Tensor [1, n, steps] F64 = reshape $ sign * v0orth / w * cos (w * t) - t * broadcast (ex / bz)
      z : Tensor [1, n, steps] F64 = reshape $
        q * ez * (t ^ fill 2.0) / (2.0 * m) + t * broadcast (slice [all, 2.to 3] v0)
  pure (broadcast (expand 1 r0) + transpose [1, 2, 0] (concat 0 x $ concat 0 y z))
