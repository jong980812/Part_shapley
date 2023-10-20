def unfreeze_block(model, block_list):
    unfreeze_list = []
    for name, param in model.named_parameters():
        for block in block_list:#if block in block_list
            if block in name:
                param.requires_grad = True
                unfreeze_list.append(name)
                break
            else:
                param.requires_grad = False
    return model, unfreeze_list