import math

class Constants:
    eta = 1e-8
    log2 = math.log(2)
    log2pi = math.log(2 * math.pi)
    logceilc = 88  # largest cuda v s.t. exp(v) < inf
    logfloorc = -104  # smallest cuda v s.t. exp(v) > 0
    clamp_min = 1e-4
    clamp_max = 1e+4

class DataDefault(object):
    # for cite-seq data
    genes_NK = [ "CHST2", "PPP1R15A", "CTSW", "ARPC2", "FCRL6", "CD53", "CNOT6L", "BPGM", "CEBPB", "CREM", "LY6E", "PSAP", "HSP90AB1" ]


    genes_CD8pT = [ 'CXCR4', 'IRF1', 'SNHG25', 'IL32', 'NBEAL1', 'RPS29', 'FOS', 'RP1-313I6.12', 'MYL12B', 'SOCS1', 'CORO1A', 'SH2D1A', 'ATP5EP2', 'VIM', 'RPS27', 'THYN1', 'SRSF3', 'RFNG' ]


    genes_CD14pMono = [ 'MCL1', 'FOSB', 'MYADM', 'CD55', 'SAMSN1', 'CDC42EP3', 'NAMPT', 'C16orf54', 'IRS2', 'PIM3', 'SRGN', 'EGR1', 'AHR', 'JMJD1C', 'HIF1A', 'IER2', 'HHEX', 'HCST', 'STK17B', 'CEBPD', 'B2M', 'RARA', 'LILRB3', 'SRSF7', 'CTD-3252C9.4', 'CD7', 'LRRFIP1', 'KLF4', 'SLC12A7', 'RP6-159A1.4', 'RAB11FIP1', 'SIGIRR', 'GABPB1-AS1', 'RFX2', 'GNG2', 'CCT8', 'ACTB', 'FPR1' ]

    genes_B = [ 'HLA-DQA1', 'VPREB3', 'CD83', 'CALM2', 'STX7', 'ATP6V0E1', 'DUSP1', 'MT-ND3', 'PTPN6', 'MMADHC', 'FNIP1', 'CCND3', 'HSPE1', 'PDHB', 'NDUFB2', 'MGEA5', 'C1orf43', 'QRSL1', 'CTDSP1', 'NDUFV2', 'CCDC50', 'FGR', 'IMP3', 'ABHD17A', 'SELT', 'ODC1', 'PDCD6', 'LAPTM5', 'TRA2B', 'DDIT3', 'MRPL13', 'MRPL22', '43525', 'SCAF4', 'ZNF706', 'IFI16', 'SIRT7', 'ARV1', 'PEX2', 'NDUFA3', 'MALSU1', 'PRELID1', 'FCRL3', 'ACTR3', 'RPS27L', 'SPON2', 'ALDH9A1', 'IGHM', 'RBX1', 'SMCHD1', 'CRIP1', 'TNFSF10', 'SAMD9L', 'PDZD8', 'CD24', 'EIF4A2', 'CHST12', 'RP5-1171I10.5', 'TINF2', 'LINC00926', 'STK17A', 'FCGR3A', 'EMG1', 'KIAA0040', 'LYSMD2', 'BCL2', 'RP11-693J15.5', 'CD180' ] 

    genes_CD4pT = [ 'H3F3B', 'DNAJB1', 'ARL6IP5', 'GIMAP7', 'IL7R', 'TAGAP', 'RP11-51J9.5', 'S100A11', 'LEPROTL1', 'MYBL1', 'LPAR6', 'BTG2', 'RPL15', 'SELK', 'NFKBIA', 'TRAT1', 'ZNF394', 'PIK3IP1', 'RP11-18H21.1', 'DDX17', 'RAB30-AS1', 'ID3', 'PRRC2C', 'P2RY10', 'RHOC', 'RBM7', 'CYCS', 'FUNDC1', 'FTH1', 'IGLC3', 'RCSD1', 'UBC', 'SBDS', 'S100A12', 'LINC01481', 'LTB', 'ESYT1', 'TBCCD1', 'UXS1', 'RGCC', 'LSM12', 'SELPLG', 'RPS2', 'C10orf88', 'GORASP2', 'EEF2', 'DDX6', 'FRAT2', 'CMC1', 'RSRC2', 'VMO1', 'PCNA', 'COA1', 'LYPD2', 'ENO1', 'METTL5', 'PMAIP1', 'IGHA1', 'CLEC2D', 'RPL10', 'PDE7A', 'RP11-347P5.1', 'ZNF672', 'MTG1', 'HOPX', 'SETD7', 'IGHA2' ]

    genes_IL7RCD4pT = [ 'RP11-386I14.4', 'SEP15', 'RPL17', 'PYCR2', 'GLRX', 'LEF1', 'ATP6V0D1', 'DDX5', 'LRRN3', 'GIMAP1', 'FBLN5', 'SDCCAG8' ]

    genes_pbmc = ['VPREB3', 'IGHM', 'LINC00926', 'CD24', 'IL7R', 'TRAT1', 'PIK3IP1', 'RGCC',  'EGR1',  'LILRB3', 'CHST2', 'CTSW', 'FCRL6', 'LEF1', 'LRRN3']

    FEATURE_GENES_SELECT = ['CD2', 'CD28', 'TIGIT', 'CD14', 'CD24', 'CD9', 'CD163', 'CLEC12A', 'CD19']
    FEATURE_GENES =  ['CD2', 'CD28', 'CX3CR1', 'CD200', 'TIGIT', 'CD14', 'CD83', 'CD24', 'CD36', 'CD72', 'CD9', 'CD27', 'CD163', 'CD69', 'CLEC12A', 'CD19', 'CD68', 'CCR10', 'CD226', 'CD93', 'CD40', 'CD70', 'CD22', 'CD34']
    #FEATURE_GENES =  ['CD2', 'CD28', 'CX3CR1', 'CD200', 'TIGIT', 'CD14', 'CD83', 'CD24', 'CD36', 'CD72', 'CD9', 'CD27', 'CD163', 'CD69', 'CLEC12A', 'CD19', 'CD68', 'CCR10', 'CD226', 'CD93', 'CD40', 'CD70', 'CD22', 'CD34']
    # for atac_multi_rna
    #FEATURE_GENES =  ['DPYD', 'PLXDC2', 'NAMPT', 'NEAT1', 'ARHGAP26', 'FHIT', 'LEF1', 'BCL11B', 'CAMK4', 'BACH2', 'PDE3B', 'NELL2', 'THEMIS', 'INPP4B', 'ANK3']
    # for atac_multi_atac
    #FEATURE_GENES = ['PLXDC2', 'TREM1', 'C10orf11', 'RBM47', 'RAB31', 'FHIT', 'BACH2', 'LEF1', 'INPP4B', 'BCL11B', 'CD8A', 'NELL2', 'CD8B', 'AFAP1', 'ST8SIA1', 'RORA', 'EPHA4', 'GLB1', 'GALM']
    # for atac_multi_both
    #FEATURE_GENES = ['PLXDC2', 'FHIT', 'BACH2', 'NELL2']

    MARKERS_DIC_RNA = {'B': ['CD79A', 'MS4A1', 'CD74', 'BANK1', 'CD37', 'RALGPS2', 'CD79B', 'HLA-DRA', 'HLA-DPB1', 'HLA-DQA1'], 'CD4 T': ['IL7R', 'RPS12', 'RPS27', 'RPL13', 'LDHB', 'LTB', 'RPS18', 'TRAC', 'CD3D', 'CD3G'], 'CD8 T': ['CD8B', 'CD8A', 'CD3D', 'CD3G', 'TRAC', 'CD3E', 'IL32', 'MALAT1', 'TRBC2', 'RPS12'], 'DC': ['CD74', 'HLA-DPA1', 'HLA-DQA1', 'HLA-DPB1', 'HLA-DRA', 'CCDC88A', 'HLA-DMA', 'CST3', 'HLA-DRB1', 'FCER1A'], 'Mono': ['CTSS', 'FTL', 'FCN1', 'PSAP', 'LYZ', 'AIF1', 'SERPINA1', 'MNDA', 'CST3', 'LST1'], 'NK': ['NKG7', 'PRF1', 'KLRD1', 'GNLY', 'CST7', 'GZMB', 'KLRF1', 'GZMA', 'CD247', 'CTSW'], 'other': ['H3F3A', 'NRGN', 'CAVIN2', 'LIMS1', 'GNG11', 'PPBP', 'HIST1H2AC', 'TUBB1', 'PF4', 'CLU'], 'other T': ['ARL4C', 'DUSP2', 'CCL5', 'IL32', 'ZFP36L2', 'GZMA', 'LYAR', 'KLRB1', 'NKG7', 'CD3G']}

    MARKERS_DIC_ADT = {'B': ['CD19', 'CD268', 'CD22', 'CD21', 'CD20', 'CD72', 'CD196', 'CD275-2', 'HLA-DR', 'CD79b'], 'CD4 T': ['CD4-1', 'CD4-2', 'CD109', 'CD28', 'CD3-2', 'CD3-1', 'CD127', 'CD278', 'TCR-2', 'CD27'], 'CD8 T': ['CD8', 'CD8a', 'CD314', 'CD2', 'CD3-1', 'CD3-2', 'CD45RB', 'TCR-2', 'CD96', 'CD73'], 'DC': ['CD71', 'HLA-DR', 'CD123', 'CD54', 'CD271', 'CD49d', 'CD195', 'CLEC12A', 'CD205', 'CD141'], 'Mono': ['CD86', 'CLEC12A', 'CD64', 'CD11c', 'CD155', 'CD11b-2', 'CD31', 'CD172a', 'CD36', 'CD14'], 'NK': ['CD56-2', 'CD16', 'CD122', 'CD335', 'CD244', 'CD43', 'CD56-1', 'CD337', 'CD45RA', 'CD38-2'], 'other': ['CD110', 'CD102', 'CD69', 'CD49b', 'CLEC2', 'CD41', 'CD42b', 'CD61', 'CD226', 'CD9'], 'other T': ['CD3-1', 'CD195', 'CD3-2', 'CD161', 'CD81', 'CD99', 'CD45-2', 'CD2', 'CD48', 'CD45RB']}

class DataPath(object):

    CIT_HAO = '/home/groups/precepts/chhy/lib_samples/integration/ds_cit_hao_info/pbmc_multimodal.h5seurat'

    DEFAULT_PARAMS = '/home/groups/precepts/chhy/kci/kci_20200321_multi_modal_vae/scvt/scvt/defaults.json'
    DEFAULT_PARAMS_MULTI = '/home/groups/precepts/chhy/kci/kci_20200321_multi_modal_vae/scvt/scvt/defaults_multi.json'


class CellType(object):
    '''

dic_assign = {
    0: 'DC',
    1: 'B',
    2: 'CD8 T',
    3: 'DC',
    4: 'other',
    5: 'Mono',
    6: 'CD4 T',
    7: 'NK'
}

K = 8
kmeans = KMeans(K)
pred_init = kmeans.fit_predict(ad.X)
#ad.obs['y_pred'] = pd.Categorical(y_pred)

pred_C = Lists.list_to_hash_value(pred_init, dic_assign)

y_true_C = ad.obs['level1']
y_true = Lists.list_to_hash_value(y_true_C, dic_cellcode)
y_pred = Lists.list_to_hash_value(pred_C, dic_cellcode)

sankey.sankey(y_true_C, pred_C)


    '''
    dic_cellcode = {
    'B': 0,
    'CD4 T': 1,
    'CD8 T': 2,
    'DC': 3,
    'Mono': 4,
    'NK': 5,
    'other': 6,
    'other T': 7
    }

class ColorIdent(object):
    ident_hao = ['#1f77b4',
 '#ff7f0e',
 '#2ca02c',
 '#d62728',
 '#9467bd',
 '#8c564b',
 '#e377c2',
 '#7f7f7f']

    level2_hao = ['#FFFF00',
 '#1CE6FF',
 '#FF34FF',
 '#FF4A46',
 '#008941',
 '#006FA6',
 '#A30059',
 '#FFDBE5',
 '#7A4900',
 '#0000A6',
 '#63FFAC',
 '#B79762',
 '#004D43',
 '#8FB0FF',
 '#997D87',
 '#5A0007',
 '#809693',
 '#6A3A4C',
 '#1B4400',
 '#4FC601',
 '#3B5DFF',
 '#4A3B53',
 '#FF2F80',
 '#61615A',
 '#BA0900',
 '#6B7900',
 '#00C2A0',
 '#FFAA92',
 '#FF90C9',
 '#B903AA',
 '#D16100']

    level3_hao = ['#FFFF00',
 '#1CE6FF',
 '#FF34FF',
 '#FF4A46',
 '#008941',
 '#006FA6',
 '#A30059',
 '#FFDBE5',
 '#7A4900',
 '#0000A6',
 '#63FFAC',
 '#B79762',
 '#004D43',
 '#8FB0FF',
 '#997D87',
 '#5A0007',
 '#809693',
 '#6A3A4C',
 '#1B4400',
 '#4FC601',
 '#3B5DFF',
 '#4A3B53',
 '#FF2F80',
 '#61615A',
 '#BA0900',
 '#6B7900',
 '#00C2A0',
 '#FFAA92',
 '#FF90C9',
 '#B903AA',
 '#D16100',
 '#DDEFFF',
 '#000035',
 '#7B4F4B',
 '#A1C299',
 '#300018',
 '#0AA6D8',
 '#013349',
 '#00846F',
 '#372101',
 '#FFB500',
 '#C2FFED',
 '#A079BF',
 '#CC0744',
 '#C0B9B2',
 '#C2FF99',
 '#001E09',
 '#00489C',
 '#6F0062',
 '#0CBD66',
 '#EEC3FF',
 '#456D75',
 '#B77B68',
 '#7A87A1',
 '#788D66',
 '#885578',
 '#FAD09F',
 '#FF8A9A']


    ident_atac = ['#1f77b4',
 '#ff7f0e',
 '#279e68',
 '#d62728',
 '#aa40fc',
 '#8c564b',
 '#e377c2',
 '#b5bd61',
 '#17becf',
 '#aec7e8',
 '#ffbb78',
 '#98df8a',
 '#ff9896',
 '#c5b0d5',
 '#c49c94',
 '#f7b6d2',
 '#dbdb8d',
 '#9edae5']
