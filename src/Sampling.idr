module Sampling

import Tensor
import Literal
import Data.Vect

||| Generate `n` three-dimensional, IID points ~N(0, 1).
|||
||| @key The RNG key.
export
normal : {n : Nat} -> (key : Tensor [] U64) -> Rand $ Tensor [n, 3] F64
normal key = do
  let norm = Tensor.normal {shape = [n, 1]} key
  [x, y, z] <- sequence $ Vect.replicate 3 norm
  pure (Tensor.concat 1 x $ Tensor.concat 1 y z)
