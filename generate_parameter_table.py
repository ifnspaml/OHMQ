mnist = False

if mnist:
    from src.systems import MnistAutoencoder as Autoencoder
else:
    from src.systems import OpenimagesAutoencoder as Autoencoder
from torch import nn
import pandas as pd


def main():
    bn_channels_list = [8, 16, 32, 480]
    num_symbols_list = [2, 4, 8, 16, 32, 64, 256]
    df = pd.DataFrame()
    for bn_channels in bn_channels_list:
        for no_quant_symbols in num_symbols_list:
            for is_int in [False, True]:
                if no_quant_symbols == 256 and bn_channels == 480:
                    continue
                approach = 'int' if is_int else 'ohm'
                print(f'Cz={bn_channels}\tS={no_quant_symbols}\tapproach={approach}')
                system = instantiate_system(bn_channels, no_quant_symbols, is_int)
                enc_params = get_num_params(system.encoder)
                dec_params = get_num_params(system.decoder)
                quant_params = get_num_quant_params(system, is_int)
                entropy_params = get_num_entropy_params(system, is_int)

                # last layer of encoder
                last_enc = get_num_params(list(system.encoder.modules())[-1])
                enc_params -= last_enc
                quant_params += last_enc

                # fist layer of decoder
                if mnist:
                    dec_begin_params = get_num_params(system.decoder.first_part)
                else:
                    dec_begin_params = get_num_params(system.decoder.begin)
                dec_params -= dec_begin_params
                quant_params += dec_begin_params
                row = dict(
                    Cz=bn_channels,
                    S=no_quant_symbols,
                    approach=approach,
                    encoder=enc_params,
                    decoder=dec_params,
                    quantizer=quant_params,
                    entropy_model=entropy_params,
                )
                df = df.append(row, ignore_index=True)
    df = df.applymap(lambda x: x if isinstance(x, str) else int(x))
    print(df.to_string())
    df.to_csv('parameters.csv', index=False)


def get_num_params(model: nn.Module):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def instantiate_system(bn_channels, no_quant_symbols, is_int):
    enc_channels = bn_channels
    if not is_int:
        enc_channels *= no_quant_symbols
    if mnist:
        return Autoencoder(
            0, 0, 0,  # does not matter
            no_quant=False,
            integer_quant=is_int,
            encoder_channels=enc_channels,
            bottleneck_channels=[bn_channels],
            no_quant_symbols=no_quant_symbols,
            ste=False,
            gauss_entropy_model=False,
        )
    else:
        return Autoencoder(
            0, 0, 0,  # does not matter
            no_quant=False,
            integer_quant=is_int,
            encoder_channels=enc_channels,
            bottleneck_channels=[bn_channels],
            no_quant_symbols=no_quant_symbols
        )


def get_num_quant_params(system, is_int):
    if is_int:
        return get_num_quant_params_int(system)
    else:
        return get_num_quant_params_ohm(system)


def get_num_entropy_params(system, is_int):
    if is_int:
        return get_num_entropy_params_int(system)
    else:
        return get_num_entropy_params_ohm(system)


def get_num_quant_params_ohm(system):
    return get_num_params(system.quant)

def get_num_quant_params_int(system):
    return 0


def get_num_entropy_params_ohm(system: Autoencoder):
    return get_num_params(system.prior_model)


def get_num_entropy_params_int(system):
    return get_num_params(system.prior_model)


if __name__ == "__main__":
    main()