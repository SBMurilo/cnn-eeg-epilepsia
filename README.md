# CNN 1D para classificação de EEG (epilepsia)

Estudo de redes neurais convolucionais 1D em PyTorch para classificar sinais de EEG de duas bases públicas:

- **Dataset de Bonn** (raiz do repositório): saudável, interictal e crise epiléptica, com evolução em 15 notebooks e, no final, **quantização INT8 e exportação dos pesos para FPGA** (arquivos `.mif` e pacote VHDL).
- **CHB-MIT** (pasta `chb-mit/`): EEG de 23 canais de pacientes pediátricos, avaliado em **paciente nunca visto** no treino.

## Bonn: dataset

[Dataset de Bonn](https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/) (Andrzejak et al., 2001): 5 conjuntos de 100 segmentos de EEG de canal único, com 4097 amostras a 173,61 Hz (23,6 s) cada.

| Pasta | Conjunto | Descrição |
|-------|----------|-----------|
| `Z` | A | Saudável, olhos abertos |
| `O` | B | Saudável, olhos fechados |
| `N` | C | Interictal, fora da zona epileptogênica |
| `F` | D | Interictal, zona epileptogênica |
| `S` | E | Crise epiléptica (ictal) |

Os dados **não estão neste repositório**. Baixe no site acima e coloque as pastas `Z`, `O`, `N`, `F` e `S` na mesma pasta dos notebooks.

## Bonn: resultados

Acurácia no conjunto de teste (20% das janelas), conforme as saídas salvas em cada notebook.

| Notebook | Tarefa | Variação testada | Acurácia |
|----------|--------|------------------|---------:|
| 01 | Saudável (A+B) × crise (E) | janelas de 1 s | 99,13% |
| 02 | 3 classes: saudável / interictal / crise | janelas de 1 s | 94,65% |
| 03 | 3 classes | janelas de 2 s | 94,82% |
| 04 | 3 classes | + data augmentation | 95,36% |
| 05 | 3 classes | + BatchNorm | 90,73% |
| 06 | 3 classes | BatchNorm sem dropout | 94,55% |
| 07 | 3 classes | dropout + kernels ajustados | 96,55% |
| 08 | 3 classes | 3 camadas convolucionais (kernels 21, 11, 7) | 97,00% |
| 09 | 3 classes | + weight decay (433 mil parâmetros) | 97,00% |
| 10 | Interictal (C+D) × crise (E) | classificador binário | 98,03% |
| 11 | A × B (olhos abertos × fechados) | classificador binário | 85,23% |
| 12 | C × D (fora × dentro da zona epileptogênica) | classificador binário | 76,36% |
| 13 | 5 classes | cascata hierárquica de modelos | 90,73% |
| 14 | 5 classes | modelo único | 83,36% |
| 15 | 3 classes | **INT8 emulando o hardware** | 96,82% (float: 97,00%) |

A cascata (notebook 13) usa uma primeira etapa que separa a categoria ampla (98,27% de acerto) e modelos especializados nas folhas; a tabela mostra a acurácia final nas 5 classes.

## Bonn: quantização e FPGA

O notebook `15_cnn_eeg_bonn_quantizado_INT8_export.ipynb`:

1. Quantiza pesos e bias de cada camada para INT8/INT32, com multiplicador de requantização em ponto fixo (`M0` e `shift`).
2. **Emula em Python, de forma bit-exata, o pipeline inteiro** que o hardware executará, para validar a perda por quantização (97,00% → 96,82%).
3. Exporta para `export_vhdl/`: ROMs de pesos e bias em `.mif`, o pacote `pkg_cnn_eeg.vhd` com as constantes de cada camada, e vetores de teste para a testbench.

Arquitetura: 3 camadas convolucionais 1D (32, 64 e 128 canais; kernels 21, 11 e 7; max pooling 2) seguidas de 2 camadas lineares (64 neurônios e 3 classes).

## CHB-MIT: avaliação em paciente nunca visto

Pasta `chb-mit/`. Base [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) (PhysioNet; Shoeb, 2009): registros contínuos de EEG em formato EDF, 23 canais a 256 Hz, com marcação do início e fim de cada crise. Os dados **não estão no repositório**: baixe da PhysioNet e coloque as pastas `chbXX` dentro de `chb-mit/`.

Pipeline dos notebooks:

1. Leitura dos arquivos EDF com MNE e das marcações de crise nos arquivos `-summary.txt`.
2. Extração de janelas de 2 s (512 amostras × 23 canais): todas as janelas de crise e até 50 janelas normais por arquivo, com margem de 10 s ao redor de cada crise.
3. **Divisão por paciente**: treino com vários pacientes e teste em um paciente que a rede nunca viu (`chb08`).
4. CNN 1D multicanal (3 camadas convolucionais + 2 lineares, 290.146 parâmetros), 50 épocas.

| Notebook | Treino | Variação | Acurácia no teste (chb08) | Recall de crise | Precisão de crise |
|----------|--------|----------|--------------------------:|----------------:|------------------:|
| 01 | chb01–07 (3.318 janelas) | dropout 0,2 | 61,68% | 35% | 94% |
| 02 | chb01–03, 05–07 (3.268 janelas) | data augmentation + dropout 0,5 | 57,23% | 27% | 93% |

O conjunto de teste tem 809 janelas (350 normais e 459 de crise). Nos dois casos a acurácia de treino chega a ~100%, mas no paciente novo a rede acerta quase todas as janelas normais (recall de 97%) e **perde a maior parte das crises**.

Leitura dos resultados: o desempenho de 97% a 99% obtido em Bonn não se repete quando o teste é feito em outro paciente. Há forte sobreajuste ao treino e grande variabilidade entre pacientes, o que torna a divisão por paciente a avaliação mais realista.

## Como executar

```bash
pip install -r requirements.txt
jupyter notebook
```

Os notebooks são independentes e numerados pela ordem dos experimentos. Os arquivos `modelo_*.pth` são usados pela cascata (notebook 13). Os notebooks de `chb-mit/` devem ser executados de dentro dessa pasta, onde ficam as pastas `chbXX`.

## Limitações

- Os sinais são divididos em janelas e **as janelas são separadas aleatoriamente entre treino e teste**. Janelas da mesma gravação podem aparecer nos dois conjuntos, então os números podem ser otimistas e não são comparáveis a avaliações separadas por gravação ou por paciente.
- O dataset de Bonn tem canal único e segmentos curtos, e é bem mais simples que registros clínicos contínuos. Por isso os resultados de Bonn e do CHB-MIT não são comparáveis diretamente.
- No CHB-MIT foram usados apenas 7 a 8 pacientes e um único paciente de teste, então os números têm alta incerteza.

## Tecnologias

Python, PyTorch, MNE, NumPy, scikit-learn, Matplotlib, Jupyter, VHDL (exportação).
