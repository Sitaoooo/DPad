import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name", type=str, default='', help="Test Name")

parser.add_argument("-t", "--task", type=str, default='gsm8k', help="Task name")
parser.add_argument("-l", "--gen_length", type=int, default=256, help="Response length")
parser.add_argument("-b", "--block_size", type=int, default=32, help="Block size")
parser.add_argument("-s", "--num_fewshot", type=int, default=0, help="Number of few-shot examples")
parser.add_argument("-th", "--threshold", type=float, default=0, help="Threshold value")
parser.add_argument("-m", "--model", type=str, default='instruct', help="Model name")

parser.add_argument("-d", "--dropout_strategy", dest='d', type=str, default='null', help="Pruning strategy ('gaussian' or 'random')")
parser.add_argument("-w", "--window", type=int, default=256, help="Dropout window size")
parser.add_argument("-k", "--k_sigma", type=int, default=4, help="The end of dropout window falls at k * sigma in Gaussian Distribution")
parser.add_argument("-sc", "--scale", type=float, default=2.0, help="Scale factor for Gaussian Pruning")
parser.add_argument("-nt", "--num_tokens", dest='nt', type=int, default=0, help="Number of reserved tokens for Random Pruning")

parser.add_argument("-c", "--use_cache", dest='c', help='Use cache', action='store_true')
parser.add_argument("-dc", "--dual_cache", dest='dc', help='Dual cache', action='store_true')
parser.add_argument("-re", "--from_scratch", dest='re', help='From scratch', action='store_true')
parser.add_argument("-e", "--early_termination", dest='e', help='From scratch', action='store_true')

# args = parser.parse_args()

# def get_tasks(type):
#     if type == 'overall':
#         # models = ['instruct', '1.5']
#         # model_paths = ['GSAI-ML/LLaDA-8B-Instruct', 'GSAI-ML/LLaDA-1.5']
#         # lengths = [512,256,512,256]
#         # block_sizes = [32,32,32,32]
#         # tasks = ['humaneval', 'gsm8k', 'mbpp', 'minerva_math']
#         # shots = [0, 4, 3, 4]
#         # strategies = [0,1]
#         # pruning_strategies = [[(3, 2.3), (4, 2.0), (3,2.3), (4,2.0)], [(3, 1.6), (3, 1.6), (3, 1.6), (3, 1.6)]]
#         # windows = [[512,128,128,128],[128,128,128,128]]
#         models = ['1.5']
#         model_paths = ['GSAI-ML/LLaDA-1.5']
#         lengths = [256]
#         block_sizes = [32]
#         tasks = ['minerva_math']
#         shots = [4]
#         strategies = [1]
#         pruning_strategies = [[(3, 1.6), (3, 1.6), (3, 1.6), (3, 1.6)]]
#         windows = [[256,128,128,256],[128,128,128,128]]

#     elif type == 'Fast-dLLM':
#         models = ['instruct']
#         model_paths = ['GSAI-ML/LLaDA-8B-Instruct']
#         lengths = [512,256]
#         block_sizes = [32,32]
#         tasks = ['humaneval', 'gsm8k']
#         shots = [0, 5]
#         strategies = [0,1,2,3]
#         pruning_strategies = [(3, 2.3), (4, 2.0)]

#     elif type == 'speedup':
#         models = ['instruct']
#         model_paths = ['GSAI-ML/LLaDA-8B-Instruct']
#         lengths = [256, 1024,512]
#         block_sizes = [32, 128,32]
#         tasks = ['gsm8k']
#         shots = [1]
#         strategies = [1,2,3]
#         pruning_strategies = [[(4,2.0)]]

#     return models, model_paths, lengths, block_sizes, tasks, shots, strategies, pruning_strategies, windows