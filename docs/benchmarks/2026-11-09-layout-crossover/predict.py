"""Broad resident 1D layout estimate, matching Dioptrin Automatic mode."""
import argparse,json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pixels',type=int,required=True)
p.add_argument('--bins',type=int,required=True)
p.add_argument('--threads',type=int,default=8)
p.add_argument('--poisson',action='store_true')
p.add_argument('--batch',action='store_true')
a=p.parse_args()
threshold=8_000_000 if a.poisson else 6_000_000 if 512<=a.bins<=2000 else 4_000_000
layout='csc' if a.batch or a.pixels<=0 or a.bins<=0 or a.threads<=0 or a.pixels>=threshold else 'csr'
print(json.dumps(dict(layout=layout,approximateSwitchPixels=threshold,
 note='Ballpark estimate from the M4 Pro sweep. CSC remains the desktop default; Automatic is opt-in. Parallel/streamed workers retain CSC. Manual overrides are available.'),indent=2))
