-- Gerado automaticamente a partir do notebook de treino/quantização.
-- Constantes de dimensionamento e requantização de cada camada da CNN1D_EEG quantizada em INT8.
library ieee;
use ieee.std_logic_1164.all;

package pkg_cnn_eeg is

    -- Comprimento da janela de entrada (2s a 173.61 Hz)
    constant C_LEN_ENTRADA : integer := 347;

    -- Camada 1: Conv1d(1 -> 32, K=21, pad=10)
    constant C1_IN_CH  : integer := 1;
    constant C1_OUT_CH : integer := 32;
    constant C1_K       : integer := 21;
    constant C1_PAD      : integer := 10;
    constant C1_IN_LEN  : integer := 347;
    constant C1_OUT_LEN : integer := 173;
    constant C1_M0      : integer := 85;
    constant C1_SHIFT   : integer := 15;

    -- Camada 2: Conv1d(32 -> 64, K=11, pad=5)
    constant C2_IN_CH  : integer := 32;
    constant C2_OUT_CH : integer := 64;
    constant C2_K       : integer := 11;
    constant C2_PAD      : integer := 5;
    constant C2_IN_LEN  : integer := 173;
    constant C2_OUT_LEN : integer := 86;
    constant C2_M0      : integer := 33;
    constant C2_SHIFT   : integer := 15;

    -- Camada 3: Conv1d(64 -> 128, K=7, pad=3)
    constant C3_IN_CH  : integer := 64;
    constant C3_OUT_CH : integer := 128;
    constant C3_K       : integer := 7;
    constant C3_PAD      : integer := 3;
    constant C3_IN_LEN  : integer := 86;
    constant C3_OUT_LEN : integer := 43;
    constant C3_M0      : integer := 41;
    constant C3_SHIFT   : integer := 15;

    -- Camada FC1: Linear(achatado -> 64), com ReLU
    constant FC1_IN_LEN  : integer := 5504;
    constant FC1_OUT_LEN : integer := 64;
    constant FC1_M0      : integer := 8;
    constant FC1_SHIFT   : integer := 15;

    -- Camada FC2: Linear(64 -> 3), sem ReLU (logits)
    constant FC2_IN_LEN  : integer := 64;
    constant FC2_OUT_LEN : integer := 3;
    constant FC2_M0      : integer := 51;
    constant FC2_SHIFT   : integer := 15;

end package pkg_cnn_eeg;
