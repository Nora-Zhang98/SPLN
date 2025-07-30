import clip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14', device, download_root='/home/stormai/userfile/zrn/CLIP')
con_dict = []
# background
con_dict.append([clip.tokenize("A less important feature of scenery, as opposed to foreground."),
                 clip.tokenize("background noise."),
                 clip.tokenize("There was tons of noise in the background.")])
# above
con_dict.append([clip.tokenize("Positioned at the upper surface of, touching from above."),
                 clip.tokenize("Positioned at or resting against the outer surface of; attached to."),
                 clip.tokenize("Expressing figurative placement, burden, or attachment."),]
                 
vectors = torch.Tensor(51, 768*3)
vectors.normal_(0, 1)

with torch.no_grad():
    for id, text_input in enumerate(con_dict):
        text_inputs = torch.cat(text_input).to(device)
        text_features = model.encode_text(text_inputs)
        rec_features = text_features.flatten(0)
        vectors[id] = rec_features

torch.save(vectors, 'path_to_save/clip_con_protos.pt')
print('ready saving concept prorotype')
