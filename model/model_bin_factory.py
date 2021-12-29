from model.bin.normalbin_res_prelu_avgpool import NormalBinResPreluAvgpool


def model_bin_factory(model_type, binary, max_disp=192):
    net = None
    if model_type == 'normalbin_res_prelu_avgpool':
        net = NormalBinResPreluAvgpool(max_disp, binary)
    else:
        raise NameError(f'Model {model_type} not supported')
    return net
