import os
import math
import torch
import random
import logging as log
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from .common.train import train
from .common.chem import mol_to_dgl
from .common.utils import print_mols
from .datasets.utils import load_mols
from .datasets.datasets import ImitationDataset, \
                               GraphClassificationDataset

class Sampler():
    def __init__(self, config, proposal, estimator):
        self.proposal = proposal
        self.estimator = estimator
        
        self.writer = None
        self.run_dir = None
        
        ### for sampling
        self.step = None
        self.PATIENCE = 100
        self.patience = 10
        self.best_eval_res = 0.
        self.best_avg_score = 0.
        self.last_avg_size = None
        self.train = config['train']
        self.num_mols = config['num_mols']
        self.num_step = config['num_step']
        self.log_every = config['log_every']
        self.batch_size = config['batch_size']
        self.score_wght = {k: v for k, v in zip(config['objectives'], config['score_wght'])}
        self.score_succ = {k: v for k, v in zip(config['objectives'], config['score_succ'])}
        self.score_clip = {k: v for k, v in zip(config['objectives'], config['score_clip'])}
        self.fps_ref = [AllChem.GetMorganFingerprintAsBitVect(x, 3, 2048) 
                        for x in config['mols_ref']] if config['mols_ref'] else None

        ### for training editor
        if self.train:
            self.dataset = None
            self.DATASET_MAX_SIZE = config['dataset_size']
            self.optimizer = torch.optim.Adam(self.proposal.editor.parameters(), lr=config['lr'])

    def scores_from_dicts(self, dicts):
        '''
        @params:
            dicts (list): list of score dictionaries
        @return:
            scores (list): sum of property scores of each molecule after clipping
        '''
        scores = []
        score_norm = sum(self.score_wght.values())
        for score_dict in dicts: # one molecule 
            score = 0.
            for k, v in score_dict.items(): # objective: score 
                if self.score_clip[k] > 0.:
                    #print(v, self.score_clip[k])
                    v = min(v, self.score_clip[k])
                    
                score += self.score_wght[k] * v
            score /= score_norm
            score = max(score, 0.)
            scores.append(score)
        return scores

    def record(self, step, old_mols, old_dicts, acc_rates):
        ### average score and size
        old_scores = self.scores_from_dicts(old_dicts)
        #print(old_dicts[0]['Bcl-2'])
        avg_score = 1. * sum(old_scores) / len(old_scores)
        sizes = [mol.GetNumAtoms() for mol in old_mols]
        avg_size = sum(sizes) / len(old_mols)
        self.last_avg_size = avg_size

        ### successful rate and uniqueness
        fps_mols, unique = [], set()
        success_dict = {k: 0. for k in old_dicts[0].keys()}
        success, novelty, diversity = 0., 0., 0.
        for i, score_dict in enumerate(old_dicts):
            all_success = True
            for k, v in score_dict.items():
                if v >= self.score_succ[k]:
                    success_dict[k] += 1.
                else: all_success = False
            success += all_success
            if all_success:
                fps_mols.append(old_mols[i])
                unique.add(Chem.MolToSmiles(old_mols[i]))
        success_dict = {k: v / len(old_mols) for k, v in success_dict.items()}
        success = 1. * success / len(old_mols)
        unique = 1. * len(unique) / (len(fps_mols) + 1e-6)

        ### novelty and diversity
        fps_mols = [AllChem.GetMorganFingerprintAsBitVect(
            x, 3, 2048) for x in fps_mols]
        
        if self.fps_ref:
            n_sim = 0.
            for i in range(len(fps_mols)):
                sims = DataStructs.BulkTanimotoSimilarity(
                    fps_mols[i], self.fps_ref)
                if max(sims) >= 0.4: n_sim += 1
            novelty = 1. - 1. * n_sim / (len(fps_mols) + 1e-6)
        else: novelty = 1.
        
        similarity = 0.
        for i in range(len(fps_mols)):
            sims = DataStructs.BulkTanimotoSimilarity(
                fps_mols[i], fps_mols[:i])
            similarity += sum(sims)
        n = len(fps_mols)
        n_pairs = n * (n - 1) / 2
        diversity = 1 - similarity / (n_pairs + 1e-6)
        
        diversity = min(diversity, 1.)
        novelty = min(novelty, 1.)
        evaluation = {
            'success': success,
            'unique': unique,
            'novelty': novelty,
            'diversity': diversity,
            'prod': success * novelty * diversity
        }

        ### logging and writing tensorboard
        log.info('Step: {:02d},\tScore: {:.7f}'.format(step, avg_score))
        self.writer.add_scalar('score_avg', avg_score, step)
        self.writer.add_scalar('size_avg', avg_size, step)
        self.writer.add_scalars('success_dict', success_dict, step)
        self.writer.add_scalars('evaluation', evaluation, step)
        self.writer.add_histogram('acc_rates', torch.tensor(acc_rates), step)
        self.writer.add_histogram('scores', torch.tensor(old_scores), step)
        for k in old_dicts[0].keys():
            scores = [score_dict[k] for score_dict in old_dicts]
            self.writer.add_histogram(k, torch.tensor(scores), step)
        print_mols(self.run_dir, step, old_mols, old_scores, old_dicts)
        
        ### early stop
        if evaluation['prod'] > .1 and evaluation['prod'] < self.best_eval_res  + .01 and \
                    avg_score > .1 and          avg_score < self.best_avg_score + .01:
            self.patience -= 1
        else: 
            self.patience = self.PATIENCE
            self.best_eval_res  = max(self.best_eval_res, evaluation['prod'])
            self.best_avg_score = max(self.best_avg_score, avg_score)
        
    def acc_rates(self, new_scores, old_scores, fixings):
        '''
        compute sampling acceptance rates
        @params:
            new_scores : scores of new proposed molecules
            old_scores : scores of old molcules
            fixings    : acceptance rate fixing propotions for each proposal
        '''
        raise NotImplementedError

    def sample(self, run_dir, mols_init):
        '''
        sample molecules from initial ones
        @params:
            mols_init : initial molecules
        '''
        self.run_dir = run_dir
        self.writer = SummaryWriter(log_dir=run_dir)
        
        ### sample
        old_mols = [mol for mol in mols_init]
        old_dicts = self.estimator.get_scores(old_mols)
        old_scores = self.scores_from_dicts(old_dicts)
        acc_rates = [0. for _ in old_mols]
        self.record(-1, old_mols, old_dicts, acc_rates)

        for step in range(self.num_step):
            #print(step)
            if self.patience <= 0: break
            self.step = step
            new_mols, fixings = self.proposal.propose(old_mols) 
            new_dicts = self.estimator.get_scores(new_mols)
            new_scores = self.scores_from_dicts(new_dicts)
            
            print("old_scores", old_scores[:20])
            print('-----------------------------------------------------------------------------------')
            print("new_scores", new_scores[:20])

            indices = [i for i in range(len(old_mols)) if new_scores[i] > old_scores[i]]
            if len(indices) == 0:
                self.patience -= 1
                continue
            print("", indices[:20])
            
            with open(os.path.join(self.run_dir, 'edits.txt'), 'a') as f:
                f.write('edits at step %i\n' % step)
                f.write('improve\tact\tarm\n')
                for i, item in enumerate(self.proposal.dataset):
                    _, edit = item
                    improve = new_scores[i] > old_scores[i]
                    f.write('%i\t%i\t%i\n' % (improve, edit['act'], edit['arm']))
            
            acc_rates = self.acc_rates(new_scores, old_scores, fixings)
            acc_rates = [min(1., max(0., A)) for A in acc_rates]
            for i in range(self.num_mols):
                A = acc_rates[i] # A = p(x') * g(x|x') / p(x) / g(x'|x)
                if random.random() > A: continue
                old_mols[i] = new_mols[i]
                old_scores[i] = new_scores[i]
                old_dicts[i] = new_dicts[i]
            if step % self.log_every == 0:
                self.record(step, old_mols, old_dicts, acc_rates)

            ### train editor
            if self.train:
                dataset = self.proposal.dataset
                ####
                
                #print("000", indices)
                dataset = data.Subset(dataset, indices)
                
                #print(type(dataset))      
                if self.dataset: 
                    self.dataset.merge_(dataset)
                else: self.dataset = ImitationDataset.reconstruct(dataset)
                n_sample = len(self.dataset)
                if n_sample > 2 * self.DATASET_MAX_SIZE:
                    indices = [i for i in range(n_sample)]
                    random.shuffle(indices)
                    indices = indices[:self.DATASET_MAX_SIZE]
                    self.dataset = data.Subset(self.dataset, indices)
                    self.dataset = ImitationDataset.reconstruct(self.dataset)
                batch_size = int(self.batch_size * 20 / self.last_avg_size)
                log.info('formed a imitation dataset of size %i' % len(self.dataset))
                loader = data.DataLoader(self.dataset,
                    batch_size=batch_size, shuffle=True,
                    collate_fn=ImitationDataset.collate_fn
                )
                
                train(
                    model=self.proposal.editor, 
                    loaders={'dev': loader}, 
                    optimizer=self.optimizer,
                    n_epoch=1,
                    log_every=10,
                    max_step=25,
                    metrics=[
                        'loss', 
                        'loss_del', 'prob_del',
                        'loss_add', 'prob_add',
                        'loss_arm', 'prob_arm'
                    ]
                )
                
                if not self.proposal.editor.device == \
                    torch.device('cpu'):
                    torch.cuda.empty_cache()


class Sampler_SA(Sampler):
    def __init__(self, config, proposal, estimator):
        super().__init__(config, proposal, estimator)
        self.k = 0
        self.step_cur_T = 0
        self.T = Sampler_SA.T_k(self.k)

    @staticmethod
    def T_k(k):
        T_0 = 1. #.1
        BETA = .05
        ALPHA = .95
        
        # return 1. * T_0 / (math.log(k + 1) + 1e-6)
        # return max(1e-6, T_0 - k * BETA)
        return ALPHA ** k * T_0

    def update_T(self):
        STEP_PER_T = 5
        if self.step_cur_T == STEP_PER_T:
            self.k += 1
            self.step_cur_T = 0
            self.T = Sampler_SA.T_k(self.k)
        else: self.step_cur_T += 1
        self.T = max(self.T, 1e-2)
        return self.T
        
    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        T = self.update_T()
        # T = 1. / (4. * math.log(self.step + 8.))
        for i in range(self.num_mols):
            # A = min(1., math.exp(1. * (new_scores[i] - old_scores[i]) / T))
            A = min(1., 1. * new_scores[i] / max(old_scores[i], 1e-6))
            A = min(1., A ** (1. / T))
            acc_rates.append(A)
        return acc_rates


class Sampler_MH(Sampler):
    def __init__(self, config, proposal, estimator):
        super().__init__(config, proposal, estimator)
        self.power = 30.
        
    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        for i in range(self.num_mols):
            old_score = max(old_scores[i], 1e-5)
            A = ((new_scores[i] / old_score) ** self.power) * fixings[i]
            acc_rates.append(A)
        return acc_rates
    

class Sampler_Recursive(Sampler):
    def __init__(self, config, proposal, estimator):
        super().__init__(config, proposal, estimator)
        
    def acc_rates(self, new_scores, old_scores, fixings):
        acc_rates = []
        for i in range(self.num_mols):
            A = 1.
            acc_rates.append(A)
        return acc_rates

