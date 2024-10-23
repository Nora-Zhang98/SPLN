import torch
save_vec = torch.load('/home/stormai/userfile/zrn/PENET/tools/output/relation_baseline/vis_protos.pt')
bg_vec = torch.normal(0,1,(1,4096))
save_vec[0] = bg_vec
torch.save(save_vec, '/home/stormai/userfile/zrn/PENET/tools/output/relation_baseline/vis_protos.pt')