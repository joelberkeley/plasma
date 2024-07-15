module Main

import Data.String
import Data.Vect
import System
import System.File

import Tensor
import Literal
import PjrtPluginXlaCpu

import Sampling
import EM

toCSV : {m, n : _} -> Vect n String -> Literal [m, n] Double -> String
toCSV headers xs =
  let strs = map tostr xs
      lines = map (joinBy "," . toList) $ cast @{toArray} strs
      headers = joinBy "," $ toList headers
   in joinBy "\n" (headers :: toList lines)

  where
  tostr : Double -> String
  tostr x =
    let inf = 1.0 / 0.0 in
    if x == inf then "inf" else
    if x == -inf then "-inf" else
    if x /= x then "nan" else show x

product2 : (a, b : Nat) -> (rest : List Nat) ->
           product (the (List _) (a :: b :: rest)) === product (the (List _) (a * b :: rest))
product2 Z _ _ = Refl
product2 (S aa) _ _ = ?product2Succ

main : IO ()
main = do
  let key = tensor {dtype = U64} 1
      temp : Tensor [] F64 = 1.0e4  -- eV
      me : Tensor [] F64 = 0.511e6  -- eV
      mp : Tensor [] F64 = 938.272e6  -- eV
      e : Tensor [] F64 = 1.0
      steps = 1000
      particleCount := 20
      paths : Tag $ TensorList [[particleCount * steps, 4], [particleCount * steps, 4]] _ = do
        let rand = Sampling.normal {n = particleCount} key
        (r0e, r0p, v0e, v0p) <- evalStateT (tensor {dtype = U64} [Scalar 1]) $ do
          let r0e = Tensor.(*) (broadcast (tensor [1.0e-2, 1.0e-2, 1.0])) !rand
              r0p = Tensor.(*) (broadcast (tensor [1.0e-2, 1.0e-2, 1.0])) !rand
              v0e = sqrt (3.0 * temp / me) * !rand
              v0p = sqrt (3.0 * temp / mp) * !rand
          pure (r0e, r0p, v0e, v0p)
        -- these are unrealistic field strengths, but they show paths more clearly
        -- https://en.wikipedia.org/wiki/Orders_of_magnitude_(magnetic_field)
        let applyEM = uniformEM {ex = 1.0e6, ez = 1.0e3, bz = 1.0e9}
        ions <- applyEM {r0 = r0p, v0 = v0p, q = e, m = mp, interval = 0.01, steps = steps}
        electrons <- applyEM {r0 = r0e, v0 = v0e, q = -e, m = me, interval = 0.01, steps = steps}
        let count = reshape {sizesEqual = product2 particleCount steps [1]}
              {to = [particleCount * steps, 1]} $ iota {shape = [particleCount, steps, 1]} 0
            ions = reshape {sizesEqual = product2 particleCount steps [3]}
              {to = [particleCount * steps, 3]} ions
            electrons = reshape {sizesEqual = product2 particleCount steps [3]}
              {to = [particleCount * steps, 3]} electrons
        pure [concat 1 count ions, concat 1 count electrons]

  device <- eitherT (die . show) pure device
  [p, e] <- TensorList.Tag.eval device paths
  either (const $ die {io = IO} "file error") pure =<< writeFile "ions.csv" (toCSV ["p", "x", "y", "z"] p)
  either (const $ die {io = IO} "file error") pure =<< writeFile "electrons.csv" (toCSV ["p", "x", "y", "z"] e)
