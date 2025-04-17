
## cuda environment
import warnings, logging, os, sys, yaml
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM']='false'

from global_var import *
sys.path.append(utils_dir)

from config import config
from writer import JsonFile
from utils_processor import Processor, set_rng_seed


def run(args):
    ## 0. init seed
    set_rng_seed(args.train['seed']) #
    if 'wandb' in args.train and args.train['wandb']:
        import wandb
        wandb.init(
            project=f"project: {'-'.join(args.train['tasks'])}",
            name=f"{'-'.join(args.train['tasks'])}-seed-{args.train['seed']}",
        )
    
    ## 1. import model and dataset
    if args.model['name']=='simrp': from models.SimRP import import_model
    model, dataset = import_model(args)

    ## 2. train or eval the model
    processor = Processor(args, model, dataset)
    args.logger['process'].warning(args.model['data'])
    if args.train['inference']:
        result = processor._evaluate(stage='test')
        print(result.test)
    else: result = processor._train()
    result = processor._evaluate(stage='test')
    if args.train['wandb']: wandb.finish()

    ## 2. output results
    record = {
        'params': {
            'e':       args.train['epochs'],
            'es':      args.train['e_stop'],
            'lr':      args.train['lr'],
            'lr_pre':  args.train['lr_plm'],
            'bz':      args.train['batch_size'],
            'dr':      args.model['drop_rate'],
            'seed':    args.train['seed'],
        },
        'metric': {
            # 'stop':    result['valid']['epoch'],
            #'tr_mf1':  result['train'][dataset.met],
            #'tv_mf1':  result['valid'][dataset.met],
            'te_mf1':  result['test']['f1'],
        },
    }
    return record


if __name__ == '__main__':
    datasets = ['rest15', 'rest16', 'acos_lap16', 'acos_rest16']
    params = {
    'plm_dir': plm_dir, 'utils_dir': utils_dir, 'data_dir': data_dir, 'cache_dir': cache_dir,
    'framework': None, 'task': '', 'dataset': datasets[0], 'model': 'simrp',
    } 
    args = config(**params)

    ## update parameters
    with open(f"./configs/{args.model['name']}.yaml", 'r') as f:
        run_config = yaml.safe_load(f)
    args.train.update(run_config['train'])
    args.model.update(run_config['model'])
    args.logger['display'].extend(['arch', 'scale'])

    seeds = [2025]
    if seeds or args.train['inference']:
        recoed_path = f"{args.file['record']}{args.model['name']}_best.jsonl"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        if not seeds: seeds = [args.train['seed']]
        for seed in seeds:
            args.train['seed'] = seed
            record = run(args)
            record_show.write(record, space=False) 
            