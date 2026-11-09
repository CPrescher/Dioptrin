"""Query the fitted prior. Standard-library Python only; no detector I/O."""
from pathlib import Path
import argparse, json, math
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pixels',type=int,required=True)
p.add_argument('--bins',type=int,required=True)
p.add_argument('--threads',type=int,default=8)
p.add_argument('--poisson',action='store_true')
p.add_argument('--batch',action='store_true')
p.add_argument('--model',type=Path,default=Path(__file__).with_name('predictor.json'))
a=p.parse_args();m=json.loads(a.model.read_text())
supported=(a.threads==m['threads'] and m['pixelRange'][0]<=a.pixels<=m['pixelRange'][1] and m['binRange'][0]<=a.bins<=m['binRange'][1])
if supported:
    x,z,e,b=math.log2(a.pixels/1e6),math.log2(a.bins/1000),int(a.poisson),int(a.batch)
    ratio=math.exp(sum(c*v for c,v in zip(m['coefficients'],[1,x,z,z*z,x*z,e,e*x,e*z,b,b*x,b*z])))
else:ratio=None
print(json.dumps(dict(layout='csr' if ratio is not None and ratio>1.3 else 'csc',predictedCscOverCsr=ratio,
    withinMeasuredDomain=supported,needsLocalTiming=True,
    note='CSC baseline; CSR prior requires >30% predicted gain. Automatic confirms >15% actual gain in at least four of five paired resident-frame probes. Batch and streamed workers retain CSC.'),indent=2))
