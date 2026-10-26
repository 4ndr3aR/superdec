import torch

PATH = "/mnt/shared/superdec-pngs/diag_e80.pt"
D = torch.load(PATH)
CL = ["box", "sphere", "torus", "cone", "capsule", "cylinder", "pyramid"]
HDR = ("%8s %4s %4s %6s %6s %7s %7s %7s %6s %6s" %
       ("class", "n", "P", "e1", "e2", "d1:GtP", "d2:PtG", "p95", "predH", "gthalf"))
print(HDR)
for c in CL:
    R = D[c]
    print("%8s %4d %4.2f %6.3f %6.3f %7.5f %7.5f %7.5f %6.3f %6.3f" %
          (c, R["n"], R["P"], R["e1"], R["e2"], R["d1"], R["d2"], R["p95"], R["s3"], R["g3"]))
