import ee.PES as ChPES
from collections import OrderedDict
import torch
def jit_pes():
    state_dict = torch.load("REANN.pth",map_location='cpu',weights_only=False)
    new_state_dict = OrderedDict()
    for k, v in state_dict['reannparam'].items():
        if k[0:7]=="module.":
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    init_pes=ChPES.ChPES(new_state_dict['gausswidth'],new_state_dict['hardness'],new_state_dict['polarizability'],new_state_dict['slatR'])
    init_pes.load_state_dict(new_state_dict)
    scripted_pes=torch.jit.script(init_pes)
    for params in scripted_pes.parameters():
        params.requires_grad=False
    scripted_pes.to(torch.double)
    scripted_pes.save("REANN_CHA_DOUBLE.pt")
    scripted_pes.to(torch.float32)
    scripted_pes.save("REANN_CHA_FLOAT.pt")
